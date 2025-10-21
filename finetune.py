import argparse
import gc
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from pathlib import Path
import shutil
import sys
import torch
import random
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_constant_schedule_with_warmup,
)
from configure import USE_CUDA
from corpora import MixtureOfBitexts, MultifileBitext, TokenizedMixtureOfBitexts, load_tokenizer
from permutations import (
    create_random_permutation_with_fixed_points,
    save_permutation_map,
)
from validate import translate_tokenized_mixture_of_bitexts, evaluate_translations

# memory things
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# prepare model for training according to experiment specifications
def prepare_model(base_model: str, freeze_decoder: bool, freeze_encoder: bool, should_finetune: bool):

    # loading pretrained model for us to finetune
    if should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 
        print('loaded pretrained model')
    # loads model architecture and random weights
    else: 
        model_config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print('loaded architecture only')

    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")

    if freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")
    if freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")

    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model

# returns average loss across batches (no gradients)
def evaluate(model, dev_data, batches: int = 100):
    model.eval()
    dev_losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y, _, _ = dev_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            loss = model(**x, labels=y.input_ids).loss
            dev_losses.append(loss.item())
    return np.mean(dev_losses)

# make dev and training graph
def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)

# training setup + training loop
def finetune(
    train_data,
    dev_data,
    tokenizer_len,
    base_model: str,
    model_dir: str,
    training_steps: int,
    report_every: int = 500,
    validate_every: int = 500,
    patience: int = 5,
    freeze_decoder: bool = False,
    freeze_encoder: bool = False,
    should_finetune: bool = True
):
    print(f"Training {model_dir}")
    model = prepare_model(base_model, freeze_decoder, freeze_encoder, should_finetune)

    # resize embeddings matrix (add embeddings from new lang codes)
    model.resize_token_embeddings(tokenizer_len) 

    # initialize optimizer (for GD)
    if should_finetune:
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    else:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            lr=None,  # Required when using relative_step
            clip_threshold=1.0,
            weight_decay=0.01,  
        )
        scheduler = None
        
    cleanup()
    train_losses, train_plot_x, train_plot_y = [], [], []
    dev_plot_x, dev_plot_y = [], []
    best_dev_loss, steps_since_best = None, 0

    # training loop!
    for i in tqdm(range(training_steps)):
        try:
            model.train()
            x, y, _, _ = train_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM. Cleaning up.")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e

        # calculating loss so far
        if i > 0 and i % report_every == 0:
            avg_train_loss = np.mean(train_losses[-report_every:])
            print(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
            sys.stdout.flush()

        # validating (after specific num steps)
        # start by calculating loss
        if i > 0 and i % validate_every == 0:
            print("Validating...")
            dev_loss = evaluate(model, dev_data)
            print(f"Dev loss: {dev_loss:.4f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss)
            sys.stdout.flush()

            # graph loss
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )

            # achieved new lower dev loss --> save best model
            if best_dev_loss is None or dev_loss < best_dev_loss:
                print("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)  # causes warning?

            # dev loss didn't improve --> early stopping if happened enough times
            else:
                steps_since_best += 1
                print(f"No improvement. Patience: {patience - steps_since_best}")
                if steps_since_best >= patience:
                    print("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()

    with open(args.config) as reader:
        config = json.load(reader)

    all_corpora = config["corpora"]
    params = config["finetuning_parameters"]
    should_finetune = params["finetune"] if "finetune" in params else True
    
    # Create unique model directory
    base_dir = config["model_dir"]
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)
    shutil.copy(args.config, Path(model_dir) / Path(args.config).name)

    lang_codes = dict()        
    for corpus in config['corpora']:
        for key in config['corpora'][corpus]:
            lang_codes[(corpus, key)] = config['corpora'][corpus][key]['lang_code']

    
    

    # get training and dev corpora
    train_data = MixtureOfBitexts.create_from_config(config, "train", only_once_thru=False)    
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=False)
    model_name = params["base_model"]
    tokenizer = load_tokenizer(model_name)


    # == SW addition: add new language codes to tokenizer ==
    # define references to new lang tokens / current special toks
    existing_special_tokens = tokenizer.additional_special_tokens

    # isolate langs not in the tokenizer already
    langs_in_current_task = set(lang_codes.values())
    langs_to_add = list(langs_in_current_task - set(existing_special_tokens))
    print(langs_to_add)

    # add new langs to the tokenizer
    new_specials = existing_special_tokens + langs_to_add
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_specials})
    print(tokenizer.additional_special_tokens)
    

    # Create the permutations
    permutations = dict()
    pmap = dict()
    for corpus in all_corpora:
        for language in all_corpora[corpus]:
            permutation_index = all_corpora[corpus][language]["permutation"]
            if permutation_index > 0:
                if permutation_index not in permutations:
                    permutations[permutation_index] = (
                        create_random_permutation_with_fixed_points(
                            len(tokenizer), tokenizer.all_special_ids
                        )
                    )
                pmap[(corpus, language)] = permutations[permutation_index]
        
    save_permutation_map(pmap, Path(model_dir) / "permutations.json")

    # tokenize training and dev data using (updated) tokenizer
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap
    )

    # train model
    finetune(
        tokenized_train,
        tokenized_dev,
        len(tokenizer),
        model_name,
        model_dir,
        params['num_steps'],
        freeze_decoder=params['freeze_decoder'] if 'freeze_decoder' in params else False,
        freeze_encoder=params['freeze_encoder'] if 'freeze_encoder' in params else False,        
        should_finetune=should_finetune
    )




    # ITERATIVE BACK TRANSLATION!
    num_iterations = 1

    # helper method to do the actual translation
    def translate_batch(sentences, target_lang, model_dir, batch_size=8):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer.src_lang = "eng_Latn"

        # create parallel example files
        working_dir = "/mnt/storage/swexler/thesis-starting-point-fall-2025/examples/data/creole_model_10_21_25/iterbacktrans/"
        eng_file_path = working_dir+target_lang+"-eng.eng"
        creole_file_path = working_dir+target_lang+"-eng.creole"

        # flush files since we open them in append
        open(eng_file_path, "w").close()
        open(creole_file_path, "w").close()

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size] # translate multiple sentences at once

            # translating the sentences
            tokenized_inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            generated_tokens = model.generate(**tokenized_inputs.to(model.device), forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
            batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # add all example pairs for this lang into files
            with open(eng_file_path, "a") as eng_file, open(creole_file_path, "a") as creole_file:
              for eng_example in batch:
                eng_file.write(eng_example + "\n")
              for creole_example in batch_translations:
                creole_file.write(creole_example + "\n")

        # create Bitext of regular training data + the synthetic data we just created
        target_lang_abbrev = target_lang.split("_")[0]

        regular_train_file_eng = corpus_dir+target_lang_abbrev+"-eng/train.eng"
        regular_train_file_creole = corpus_dir+target_lang_abbrev+"-eng/train."+target_lang_abbrev

        return MultifileBitext([eng_file_path, regular_train_file_eng], [creole_file_path, regular_train_file_creole])


    # get Eng -> each creole dialect
    target_langs = lang_codes.values()
    corpus_dir = "/mnt/storage/swexler/thesis-starting-point-fall-2025/examples/data/creole_model_10_21_25/"

    # Read and combine all English sentences
    eng_files = glob(os.path.join(corpus_dir, "**/train.eng"), recursive=True) # Find all train.eng files recursively

    english_sentences = []
    for fpath in eng_files:
        with open(fpath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            english_sentences.extend(lines)


    # choose random examples from all English examples
    sample_size = 20
    random_subset = random.sample(english_sentences, sample_size)
    print(random_subset)

    # split examples into each target lang 
    chunk_size = sample_size // len(target_langs)

    # define where MultiFileBitexts (w/ synthetic and training files) will end up
    bitexts = {}

    # translate batches and create Bitexts in the process
    i = 0
    for lang in target_langs:
        start = i * chunk_size
        end = start + chunk_size
        print(f"Translating to {lang}...")
        bitexts[lang] = translate_batch(random_subset[start:end], lang, model_dir)
        # for src, tgt in bitexts[lang]:
        #   print(src, "->", tgt)
        print()
        i += 1

    print("=====")
    print()
    print(bitexts)

   
    #transform bitexts into format that works with MixtureOfBitexts
    transformed_bitexts = {("eng_Latn", lang): bitext for lang, bitext in bitexts.items()}
    # {(eng_Latn, acf_Latn): Bitext(eng_Latn examples, acf_Latn examples),... }

    train_data_ibt = MixtureOfBitexts(transformed_bitexts, 8) # batch size = 8

    tokenized_train_ibt = TokenizedMixtureOfBitexts(
            train_data_ibt, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap
        )

    finetune(
        tokenized_train_ibt,
        tokenized_dev,
        len(tokenizer),
        model_name,
        model_dir,
        params['num_steps'],
        freeze_decoder=params['freeze_decoder'] if 'freeze_decoder' in params else False,
        freeze_encoder=params['freeze_encoder'] if 'freeze_encoder' in params else False,        
        should_finetune=should_finetune
    )







    # evaluation: tokenize test data and predict translations
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)    
    tokenized_test = TokenizedMixtureOfBitexts(test_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, lang_codes, pmap
    )
    with open(Path(model_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    print("Translations complete.")

    # evaluation: package reference sentences nicely across all examples
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)    
    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_lang, tgt_lang = batch
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = '->'.join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open(Path(model_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    print("References complete.")

    # evaluation: evaluate model's translations and write scores
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(model_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    print("Evaluation complete.")


    


if __name__ == "__main__":
    main()
