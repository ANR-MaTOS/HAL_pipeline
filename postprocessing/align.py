from bertalign.aligner import Bertalign
from collections import defaultdict
from datetime import date, timedelta, datetime
from jsonargparse import CLI
from pathlib import Path
from typing import List
import json 
import os
import re  
import string 
import time 

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

def normalise(text:str) -> str:
    return re.sub(r"\n", " ", text)

def main(tasks: List[dict], models: List[dict] = None):
    languages = ["en", "fr"]
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks:
        if task.get("name") == "postprocessing":             
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "alignment":
                    print("Alignment begins")
                    postprocessed_path = task["postprocessed_path"]
                    for model in models: 
                        model_name = model["name"] 
                        for tgt_lang in languages: 
                            src_lang = "en" if tgt_lang == "fr" else "fr"
                            postprocessed_lang_path = string.Template(postprocessed_path)
                            postprocessed_lang_path = postprocessed_lang_path.safe_substitute(model_name = model_name, lang = tgt_lang)
                            postprocessed_f = Path(postprocessed_lang_path) / f"postprocessed_{datestamp}.json"
                            data = json.load(open(postprocessed_f, "r", encoding="utf-8"))
                            print(f"{len(data)} entries loaded")
                            for docid, entry in data.items(): 
                                src_abstract = entry["src_abstract"]
                                tgt_abstract = entry["tgt_abstract"]
                                entry["alignments"] = defaultdict(dict)     
                                aligner = Bertalign(src=src_abstract, tgt=tgt_abstract, src_lang=src_lang, tgt_lang=tgt_lang)
                                aligner.align_sents() 
                                alignments = aligner.get_alignments() 
                                entry["alignments"] = alignments
                            with open(postprocessed_f, "w", encoding="utf-8") as f: 
                                json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CLI(main, description=__doc__)