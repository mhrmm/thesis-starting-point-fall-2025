from tqdm import tqdm
from typing import Dict, List

class iterativeBackTranslation():
    def __init__(self, lang_to_path: Dict[str, List[str]], model_dir: str, lang_codes: Dict[str, str], pivot_lang = "spa_Latn") -> None:
        self.langs_to_path = lang_to_path
        self.data_dir = model_dir
        self.lang_codes = lang_codes
        self.pivot_lang = pivot_lang
    def translate(
        self, 
        src_tokenized,
        tokenizer,
        model,
        tgt_lang,
        permutation=None,
        a=32,
        b=3,
        num_beams=4,
        **kwargs
    ):
        model.eval()
        result = model.generate(
            **src_tokenized.to(model.device),
            forced_bos_token_id=tokenizer.get_special_tokens()[tgt_lang],
            max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
            num_beams=num_beams,
            **kwargs
        )
        result = result.to('cpu')
        if permutation is not None:
            result.apply_(permutation.get_inverse())
        return tokenizer.batch_decode(result)

    def backTranslate(self, in_model, target_languages, tokenizer):
        for lang in target_languages:
            mono_files = self.langs_to_path.get(lang, [])

            translation_lines = []
            for mono_file in mono_files:
                with open(mono_file, "r", encoding="utf-8") as f:
                    lines = [l.rstrip("\n") for l in f if l.strip()]
                    translation_lines.extend(lines)

            with open(f"{self.data_dir}/{lang}/trainBackTrans.es","w") as es, \
                open(f"{self.data_dir}/{lang}/trainBackTrans.tsv","w") as dialect:
                    for line in tqdm(translation_lines,desc=f"Translating: {lang}", total=len(translation_lines)):
                        inputs = tokenizer([line])
                        inputs = inputs.to(in_model.device)
                        translated_tokens = self.translate(src_tokenized=inputs, model=in_model,tokenizer=tokenizer, tgt_lang=self.pivot_lang)
                        es.write(translated_tokens[0]+ "\n")
                        dialect.write(line + "\n")
                            
