from pathlib import Path
from transformers import AutoTokenizer 
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from permutations import create_random_permutation_with_fixed_points
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def batch_sort(batch_size=64, cur_directory="", language=""):
    print("\n__________________________________")
    print(f"Starting batch_sort for: {language}")
    print(f"Directory: {cur_directory}")
    num_lines = 0
    line_list_en = []
    
    with open(f"{cur_directory}/processedTrain.es", "r", encoding="utf-8") as f:
        line_list_en = f.readlines()
        for line in line_list_en:
            num_lines += 1

    number_of_batches = num_lines//batch_size
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.src_lang = "es_Latn"
    line_length_dict = {} #Key: line number; Value: line length
    i = 0
    for line in line_list_en:
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
    
    with open(f"{cur_directory}/optimized_train_{batch_size}.es","w") as file:
        for i in range(number_of_batches):
            for j in range(batch_size):
                file.write(line_list_en[reshuffled_batches[i][j]])

    extension_dictionary = {
    "ashaninka": "cni",
    "awajun": "agr",
    "aymara": "aym",
    "bribri": "bzd",
    "chatino": "ctp",
    "guarani": "gn",
    "nahuatl": "nah",
    "otomi": "oto",
    "quechua_south_bolivian": "quy",  # what your files use
    "raramuri": "tar",
    "shipibo_konibo": "shp",
    "wayuu": "guc",
    "wixarika": "hch"}
    
    lang_code = extension_dictionary[language]
    line_list = []
    with open(f"{cur_directory}/processedTrain.{lang_code}", "r") as f:
        line_list = f.readlines()
    
    with open(f"{cur_directory}/optimized_train_{batch_size}.{lang_code}","w") as file:
        for i in range(number_of_batches):
            for j in range(batch_size):
                file.write(line_list[reshuffled_batches[i][j]])

def organize_data(batch_size, data_dir):
    data_dir = Path(data_dir)
    for corpus_dir in data_dir.iterdir():
        _,language = (str(corpus_dir)).split("/")
        batch_sort(batch_size=batch_size, cur_directory=corpus_dir, language=language)