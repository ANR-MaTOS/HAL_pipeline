from collections import defaultdict
from comet import download_model, load_from_checkpoint
from datetime import date, timedelta, datetime
from jsonargparse import CLI
from merge_alignments import align
from pathlib import Path
from typing import List
import json 
import os
import re  
import torch 
import string 
import sys 

mtqe_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
mtqe_model = load_from_checkpoint(mtqe_model_path)
print("MTQE model loaded")

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

def get_mtqe_scores(mtqe_data): 
    if not mtqe_data:
        return []
    if not isinstance(mtqe_data, list):
        return []
    mtqe_model_output = mtqe_model.predict(mtqe_data, batch_size=8, gpus=1)
    scores = mtqe_model_output["scores"]
    return scores

def main(tasks: List[dict], models: List[dict] = None):
    languages = ["fr", "en"]
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks: 
        if task.get("name") == "postprocessing":           
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "mtqe":
                    print("MTQE begins")
                    postprocessed_template = task["postprocessed_path"]
                    for model in models: 
                        model_name = model["name"]
                        for tgt_lang in languages: 
                            postprocessed_path = string.Template(postprocessed_template)
                            postprocessed_path = postprocessed_path.safe_substitute(model_name = model_name, lang = tgt_lang) 
                            postprocessed_f = Path(postprocessed_path) / f"postprocessed_{datestamp}.json"
                            with open(postprocessed_f, "r", encoding="utf-8") as f: 
                                data = json.load(f) 
                            print(f"Fetched {len(data)} abstracts from {postprocessed_f}")
                            # post_alignment = align(data)
                            # print(post_alignment)
                            # print(f"{len(post_alignment)} abstracts after alignment merging")
                            # assert len(post_alignment) == len(data) 
                            
                            # MTQE 
                            scores = [] 
                            mtqe_data = []
                            for docid, entry in data.items(): 
                                for alignment in entry["alignments"]: 
                                    src_txt = alignment["src_sent"]
                                    tgt_txt = alignment["tgt_sent"]
                                    mtqe_data.append({"src": src_txt, "mt": tgt_txt})
                            print(len(mtqe_data), "alignments loaded")
                            scores = get_mtqe_scores(mtqe_data) 
                            assert len(mtqe_data) == len(scores)
                            counter = 0    
                            for docid, entry in data.items():
                                for alignment in entry["alignments"]: 
                                    alignment["comet_kiwi_score"] = scores[counter]
                                    counter += 1 
                            assert len(scores) == counter
                            with open(postprocessed_f, "w", encoding="utf-8") as output_f: 
                                json.dump(data, output_f, ensure_ascii=False, indent=2)
                            
if __name__ == "__main__":
    CLI(main, description=__doc__)