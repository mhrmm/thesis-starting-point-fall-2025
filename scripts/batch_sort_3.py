from pathlib import Path
import glob
import argparse
from transformers import AutoTokenizer 
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from permutations import create_random_permutation_with_fixed_points
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

data_dir = Path(args.data_dir)
lang_ext = {
    # Main low-resource languages
    "ashaninka": "cni",
    "awajun": "agr",
    "aymara": "aym",
    "bribri": "bzd",
    "chatino": "ctp",
    "guarani": "gn",
    "nahuatl": "nah",
    "otomi": "oto",
    "quechua": "quy",  
    "raramuri": "tar",
    "shipibo_konibo": "shp",
    "wayuu": "guc",
    "wixarika": "hch"}

def batch_sort(batch_size=64, tgt_file: Path = None, es_file: Path = None):
    num_lines = 0
    line_list_es = []

    with es_file.open("r") as f1:
        line_list_es = f1.readlines()
        for line in line_list_es:
            num_lines +=1
    
    number_of_batches = num_lines//batch_size
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.src_lang = "es_Latn"
    line_length_dict = {} #Key: line number; Value: line length
    i = 0
    for line in line_list_es:
        tokenized_line = tokenizer(line)['input_ids']
        line_length_dict[i] = len(tokenized_line)
        i+=1
    sorted_lines = sorted(line_length_dict.items(), key=lambda item: item[1])
    order_of_lines = []
    for pair in sorted_lines:
        order_of_lines.append(pair[0])
    batch_list = []
    for i in range(number_of_batches):
        this_batch = []
        for j in range(batch_size):
            this_batch.append(order_of_lines[i*batch_size+j])
        batch_list.append(this_batch)
    pmap_batches = create_random_permutation_with_fixed_points(number_of_batches,[]) 
    reshuffled_batches = []
    for k in range(number_of_batches):
        reshuffled_batches.append(batch_list[pmap_batches(k)])

    out_es = es_file.with_name(es_file.stem + "_processed" + es_file.suffix)
    out_tgt = tgt_file.with_name(tgt_file.stem + "_processed" + tgt_file.suffix)
    
    with out_es.open("w") as f1:
        for i in range(number_of_batches):
            for j in range(batch_size):
                f1.write(line_list_es[reshuffled_batches[i][j]])
    
    
    line_list = []
    with tgt_file.open("r") as f2:
        line_list = f2.readlines()
    
    with  out_tgt.open("w") as f:
            for i in range(number_of_batches):
                for j in range(batch_size):
                    f.write(line_list[reshuffled_batches[i][j]])

for lang_dir in data_dir.iterdir():
    language, _ = Path(lang_dir).name.split("-")
    ext = lang_ext[language]
    print("__________________________________")
    print(language)
    for file in lang_dir.glob(f"*_filtered.{ext}"):
        if "nllb_" in file.name:
            continue
        es_file = file.with_suffix(".es")
        print(f"batching tgt_file:{file.name} src_file{es_file.name}")
        batch_sort(tgt_file = file, es_file=es_file)