from collections import defaultdict
from datetime import date, datetime, timedelta
from jsonargparse import CLI
from pathlib import Path
from typing import List
import fasttext
import json 
import os
import re  
import string

# language identification
langid_model_path = "/home/ptsolaki/scratch/matos_prod/lid.176.bin" # download from https://fasttext.cc/docs/en/language-identification.html 
langid_model = fasttext.load_model(langid_model_path)
print("Fasttext model loaded")

def create_timestamp(data_date): 
    # offline mode
    if data_date: 
        dt_object = datetime.strptime(data_date, "%d_%m_%Y")
        datestamp = dt_object.strftime("%d_%m_%Y")
        return datestamp 
    # online mode
    if data_date is None: 
        today = date.today() 
        yesterday = today - timedelta(days = 1)
        datestamp = yesterday.strftime("%d_%m_%Y")
        return datestamp 

def get_lang(txt):
    txt = re.sub("\n", " ", txt)
    lang, score = langid_model.predict(txt)
    lang, score = lang[0][-2:], score[0]
    return lang, score

def main(tasks: List[dict], models: List[dict] = None):
    languages = ["en", "fr"]
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks:
        if task.get("name") == "postprocessing":            
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "langid": 
                    for model in models: 
                        model_name = model["name"]
                        for lang in languages: 
                            postprocessed_path = task.get("postprocessed_path") 
                            tgt_lang = lang 
                            postprocessed_path = string.Template(postprocessed_path).safe_substitute(model_name = model_name, lang = tgt_lang)
                            with open(Path(postprocessed_path) / f"postprocessed_{datestamp}.json", "r", encoding="utf-8") as f: 
                                data = json.load(f)
                            for k, v in data.items(): 
                                langid_res, langid_score = get_lang(v["tgt_abstract"])
                                v["lang"] = langid_res 
                                v["langid_score"] = langid_score
                            postprocessed_f = Path(postprocessed_path) / f"postprocessed_{datestamp}.json"
                            with open(postprocessed_f, "w", encoding="utf-8") as f: 
                                json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CLI(main, description=__doc__)