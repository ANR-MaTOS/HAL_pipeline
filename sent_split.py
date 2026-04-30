from datetime import date, timedelta, datetime
from jsonargparse import CLI
from pathlib import Path
from trankit import Pipeline 
from typing import List
import json 
import os

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

def get_sentences(txt, pipeline):
    sentences = [] 
    res = pipeline.ssplit(txt) 
    res_sentences = res["sentences"]
    for rank, res_s in enumerate(res_sentences): 
        src_txt = res_s["text"]
        res_data = {"rank":rank, "src_txt":src_txt}
        sentences.append(res_data)
    return sentences 

def main(tasks: List[dict], models: List[dict] = None):
    print("Sentence preprocessing begins")
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks:
        if task.get("name") == "sent_split":       
            for subtask in task.get("subtasks",[]): 
                print(type(subtask), subtask)  
                src_lang = subtask["src_lang"]
                # lang_full = subtask["lang_full"]
                src_path = subtask["src_path"]
                tgt_path = subtask["tgt_path"]

                datefile = f"checked_{src_lang}_{datestamp}.json"
                source_f = Path(src_path) / datefile
                with open(source_f, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data: 
                    full_lang = "french" if src_lang == "fr" else "english"
                    p = Pipeline(lang=full_lang, gpu=True, cache_dir="/home/ptsolaki/scratch/matos_pipeline/cache")
                    preprocessed = [] 
                    for entry in data: 
                        abstract_field = f"{src_lang}_abstract_s"
                        abstract_list = entry.get(abstract_field)
                        if not isinstance(abstract_list, list) or len(abstract_list) == 0:
                            continue
                        abstract_txt = abstract_list[0]
                        sentences = get_sentences(abstract_txt, p)
                        entry["sentences"] = sentences 
                
                target_f = Path(tgt_path) / f"sentences_{datestamp}.json"
                target_f.parent.mkdir(parents=True, exist_ok=True)
                with open(target_f, "w", encoding="utf-8") as tgt_file: 
                    json.dump(data, tgt_file, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)