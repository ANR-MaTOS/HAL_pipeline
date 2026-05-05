from datetime import date, timedelta, datetime
from itertools import islice
from jsonargparse import CLI
from pathlib import Path
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

def get_segments(sentences, num):     
    segments = []
    if isinstance(sentences,list): 
        texts = [sent["src_txt"] for sent in sentences]
        segments = [" ".join(texts[i:i+num]) for i in range(0,len(texts),num)]
    return segments 

def main(tasks: List[dict], models: List[dict] = None):
    print("Segment split begins")
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks:
        if task.get("name") == "segment_split":       
            for subtask in task.get("subtasks",[]): 
                lang = subtask["src_lang"]
                src_path = subtask["src_path"]
                tgt_path = subtask["tgt_path"]

                datefile = f"sentences_{datestamp}.json"
                source_f = Path(src_path) / datefile
                with open(source_f, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if data:
                    for entry in data: 
                        sentences = entry.get("sentences")
                        if sentences: 
                            segments = get_segments(sentences, 3)
                            entry["segments"] = segments 
                            entry["segments"] = [{"rank":rank, "src_txt":src_txt} for rank, src_txt in enumerate(segments)]

                target_f = Path(tgt_path) / f"segments_{datestamp}.json"
                target_f.parent.mkdir(parents=True, exist_ok=True)
                with open(target_f, "w", encoding="utf-8") as tgt_file: 
                    json.dump(data, tgt_file, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)

                        




