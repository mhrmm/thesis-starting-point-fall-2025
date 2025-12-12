from pathlib import Path
import glob
import argparse

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

for lang_dir in data_dir.iterdir():
    language, _ = Path(lang_dir).name.split("-")
    ext = lang_ext[language]
    print(language)

    for file in lang_dir.glob(f"*.{ext}"):
        if "nllb_" in file.name:
            es_file = file.with_suffix(".eng")
        else:
            es_file = file.with_suffix(".es")

        with file.open("r") as f1, \
            es_file.open("r") as f2:
                first_tgt = f2.readline().strip()
                lines_1 = f1.readlines()
                lines_2 = f2.readlines()

                out_src = file.with_name(file.stem + "_filtered" + file.suffix)
                out_tgt = es_file.with_name(es_file.stem + "_filtered" + es_file.suffix)

                with out_src.open("w") as outf1, \
                        out_tgt.open("w") as outf2:
                        for x, y in zip(lines_1, lines_2):
                            if x.strip() == '' or y.strip() == '':
                                continue
                            outf1.write(x)
                            outf2.write(y)