import json
import os
from pathlib import Path
import random

source_folder = "../data"
out_folder = "../configs"

#language pairs from AmericasNLP 2025
language_pairs = [
    "shipibo_konibo-spanish"
]
extension_dictionary ={
    "ashaninka-spanish": "cni",
    "awajun-spanish": "agr",
    "aymara-spanish": "aym",
    "bribri-spanish": "bzd",
    "chatino-spanish": "ctp",
    "guarani-spanish": "gn",
    "nahuatl-spanish": "nah",
    "otomi-spanish": "oto",
    "quechua-spanish": "quy",
    "raramuri-spanish": "tar",
    "shipibo_konibo-spanish": "shp",
    "wayuu-spanish": "guc",
    "wixarika-spanish": "hch"
}

dev_split_ratio = 0.15
random.seed(42)

config = {
    "model_dir": "models/shp_ibt",
    "data_dir": "data",
    "finetuning_parameters": {
        "base_model": "facebook/nllb-200-distilled-600M",
        "finetune": True,
        "freeze_encoder": False,
        "freeze_decoder": False,
        "batch_size": 64,
        "num_steps": 10000,
        "report_every": 500, 
        "validate_every": 500,
        "patience": 5,
        "add_new_lang_codes": True,
        "add_new_tokens": False,
        "new_lang_size": 2**7,
        "IBT": False,
        "IBT_iterations": 1,
        "IBT_training_steps": 1000,
        "IBT_Langs": ["shipibo_konibo-spanish"],
        "lang_extensions": extension_dictionary
    },
    "corpora": {},
    "bitexts": []
}
batch_size = config["finetuning_parameters"]["batch_size"]
for label_pair in language_pairs:
    lang_folder = f"{source_folder}/{label_pair}"
    
    #write out the training
    with open(f"{lang_folder}/train_filtered_processed.{extension_dictionary[label_pair]}", "r", encoding="utf-8") as f_indigenous, \
        open(f"{lang_folder}/train_filtered.es", "r") as f_es:
            src_lines = f_indigenous.readlines()
            tgt_lines = f_es.readlines()
    
    #split traing into train and test
    paired_data = list(zip(src_lines, tgt_lines))
    random.shuffle(paired_data)
    
     # Split into train and dev
    total_lines = len(paired_data)
    dev_size = int(total_lines * dev_split_ratio)
    train_size = total_lines - dev_size
    
    train_data = paired_data[:train_size]
    dev_data = paired_data[train_size:]

    with open(f"{lang_folder}/train_split_filtered.{extension_dictionary[label_pair]}", "w", encoding="utf-8") as f_indigenous, \
        open(f"{lang_folder}/train_split_filtered.es", "w") as f_es:
            for src_line, tgt_line in train_data:
                  f_indigenous.write(src_line)
                  f_es.write(tgt_line)
    
    with open(f"{lang_folder}/dev_split_filtered.{extension_dictionary[label_pair]}", "w", encoding="utf-8") as f_indigenous, \
        open(f"{lang_folder}/dev_split_filtered.es", "w") as f_es:
            for src_line, tgt_line in dev_data:
                  f_indigenous.write(src_line)
                  f_es.write(tgt_line)
    
    _, tgt_folder = lang_folder.split('/')
    src, _ = label_pair.split("-")
    tgt = "spanish"
    corpus_key = f"{label_pair}"
    config["corpora"][corpus_key] = {
        src: {
            "mono_data": [],
            "lang_code": f"{extension_dictionary[label_pair]}_Latn",
            "train":[f"{tgt_folder}/train_split_filtered_processed.{extension_dictionary[label_pair]}"],
            "dev": [f"{tgt_folder}/dev_split_filtered_processed.{extension_dictionary[label_pair]}"],
            "test": [f"{tgt_folder}/dev.{extension_dictionary[label_pair]}"],
            "permutation": 0
        },
        "spanish": {
            "lang_code": "spa_Latn",
            "train":[f"{tgt_folder}/train_split_filtered_processed.es"],
            "dev": [f"{tgt_folder}/dev_split_filtered_processed.es"],
            "test": [f"{tgt_folder}/dev.es"],
            "permutation": 0
        }
    }

    config["bitexts"].append({
        "corpus" : f"{label_pair}",
        "src" : src, 
        "tgt" : tgt,
        "train_lines": None,
    })
    config["bitexts"].append({
        "corpus" : f"{label_pair}",
        "src" : tgt, 
        "tgt" : src,
        "train_lines": None,
        })
    

with open(f"{out_folder}/spa-shp.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)