from collections import defaultdict
from datetime import date
from datetime import timedelta
from jsonargparse import CLI
from pathlib import Path
from typing import List
import fasttext
import json 
import os
import re  
import string

langid_model_path = "/home/ptsolaki/scratch/matos_pipeline/lid.176.bin" 
# download from https://fasttext.cc/docs/en/language-identification.html 
langid_model = fasttext.load_model(langid_model_path)
print("Fasttext model loaded")

def get_lang(txt):
    txt = re.sub("\n", " ", txt)
    lang, score = langid_model.predict(txt)
    lang, score = lang[0][-2:], score[0]
    return lang, score

def main(tasks: List[dict]):
    languages = ["en", "fr"]
    modes = ["doc_0shot", "doc_1shot", "doc_2shot", "segment_0shot", "sentence_0shot"]
    for task in tasks:
        if task.get("name") == "postprocessing":             
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "langid": 
                    for lang in languages: 
                        exp_lang = lang 
                        postprocessed_path = subtask["postprocessed_path"]
                        postprocessed_path = string.Template(postprocessed_path)
                        postprocessed_path = postprocessed_path.safe_substitute(lang = exp_lang)
                        with open(Path(postprocessed_path) / "postprocessed.json", "r", encoding="utf-8") as f: 
                            data = json.load(f)
                        for k, v in data.items(): 
                            for mode in modes: 
                                langid_res, langid_score = get_lang(v[mode]["target_txt"])
                                v[mode]["lang"] = langid_res 
                                v[mode]["langid_score"] = langid_score
                        postprocessed_f = Path(postprocessed_path) / "postprocessed.json"
                        with open(postprocessed_f, "w", encoding="utf-8") as f: 
                            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CLI(main, description=__doc__)