from datetime import date, timedelta, datetime
from jsonargparse import CLI
from pathlib import Path
from typing import List
import duckdb 
import json 
import os 
import string 

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

def create_db_date(data_date):
    dt_object = datetime.strptime(data_date, "%d_%m_%Y")
    db_date = dt_object.strftime("%Y-%m-%d")
    return db_date

def eval_entry(entry, length_ratio_threshold, tgt_lang, langid_threshold, mtqe_threshold):
    logs = ""
    length_ratio = entry.get("length_ratio")
    langid_res = entry.get("lang")
    langid_score = entry.get("langid_score") 
    alignments = entry.get("alignments")
    if alignments: 
        mtqe_total = 0
        for a in alignments: 
            if a.get("comet_kiwi_score"): 
                mtqe_total += a["comet_kiwi_score"]
        mtqe_average = mtqe_total / len(alignments)
    if not length_ratio or \
       not langid_res or \
       not langid_score or \
       not alignments: 
        return False, "" 
    if length_ratio < length_ratio_threshold and \
       langid_res == tgt_lang and \
       langid_score >= langid_threshold and \
       mtqe_average >= mtqe_threshold: 
        eval = True 
    else: 
        if length_ratio >= length_ratio_threshold: 
            logs += "Length ratio higher than threshold. "
        eval = False 
        if langid_res != tgt_lang: 
            logs += "Output language does not match target language. "
        if langid_score < langid_threshold: 
            logs += "LangID threshold lower than threshold. "
        if mtqe_average < mtqe_threshold: 
            logs += "MTQE score lower than threshold."
    return eval, logs, mtqe_average 

def main(tasks: List[dict], models: List[dict] = None):
    src_languages = ["en", "fr"]
    tgt_languages = ["fr", "en"]
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for task in tasks: 
        if task.get("name") == "postprocessing":           
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "filter":
                    print("Filtering begins")
                    length_ratio_threshold = subtask.get("length_ratio_threshold")
                    langid_threshold = subtask.get("langid_threshold")
                    mtqe_threshold = subtask.get("mtqe_threshold")

                    for model in models: 
                        model_name = model["name"]
                        for src_lang, tgt_lang in zip(src_languages, tgt_languages): 
                            postprocessed_path = task.get("postprocessed_path")
                            postprocessed_path = string.Template(postprocessed_path).safe_substitute(model_name = model_name, lang = tgt_lang)
                            postprocessed_f = Path(postprocessed_path) / f"postprocessed_{datestamp}.json"
                            accepted_path = subtask.get("accepted_path")
                            accepted_path = string.Template(accepted_path).safe_substitute(model_name = model_name, lang = tgt_lang)
                            Path(accepted_path).mkdir(exist_ok=True, parents=True)
                            accepted_f = Path(accepted_path) / f"accepted_{datestamp}.json"
                            rejected_path = subtask.get("rejected_path")
                            rejected_path = string.Template(rejected_path).safe_substitute(model_name = model_name, lang = tgt_lang)
                            Path(rejected_path).mkdir(exist_ok=True, parents=True)
                            rejected_f = Path(rejected_path) / f"rejected_{datestamp}.json"
                            with open(postprocessed_f, "r", encoding="utf-8") as f: 
                                data = json.load(f) 
                            print(f"Fetched {len(data)} abstracts from {postprocessed_f}")
                            accepted = {}
                            rejected = {} 
                            data_to_insert = []
                            publication_date = create_db_date(data_date)
                            processing_date = date.today()
                            for docid, entry in data.items(): 
                                notif_ready, logs, mtqe_average = eval_entry(entry, length_ratio_threshold, tgt_lang, langid_threshold, mtqe_threshold) 
                                entry["notif_ready"] = notif_ready 
                                entry["logs"] = logs 
                                if notif_ready == True: 
                                    accepted[docid] = entry 
                                elif notif_ready == False: 
                                    rejected[docid] = entry
                                
                                data_to_insert.append(
                                    (
                                        docid, 
                                        entry["url"],
                                        entry["domain"], 
                                        entry["title"], 
                                        src_lang,  
                                        entry["src_abstract"], 
                                        tgt_lang,
                                        entry["tgt_abstract"], 
                                        model_name, 
                                        entry["lang"], 
                                        entry["langid_score"],
                                        entry["length_ratio"], 
                                        mtqe_average, 
                                        publication_date,
                                        processing_date,
                                        notif_ready, 
                                        logs
                                    )
                                )
                            print(f"{len(accepted)} abstracts are notification ready")
                            print(f"{len(rejected)} abstracts were rejected")
                            with open(accepted_f, "w", encoding="utf-8") as f: 
                                json.dump(accepted, f, ensure_ascii = False, indent = 2) 
                            with open(rejected_f , "w", encoding="utf-8") as f:
                                json.dump(rejected, f, ensure_ascii = False, indent = 2) 

                            # populate database 
                            with duckdb.connect("matos.duckdb") as con: 
                                con.execute(
                                    """
                                    CREATE TABLE IF NOT EXISTS publications (
                                        docid VARCHAR, 
                                        url VARCHAR, 
                                        domain VARCHAR, 
                                        title VARCHAR, 
                                        src_lang VARCHAR,  
                                        src_abstract VARCHAR, 
                                        tgt_lang VARCHAR,  
                                        tgt_abstract VARCHAR, 
                                        model VARCHAR, 
                                        tgt_langid VARCHAR, 
                                        tgt_langid_score DOUBLE, 
                                        length_ratio DOUBLE, 
                                        cometkiwi DOUBLE, 
                                        publication_date TIMESTAMP, 
                                        processing_date TIMESTAMP, 
                                        notif_ready BOOLEAN, 
                                        logs VARCHAR 
                                    )
                                    """
                                )
                                con.executemany(
                                    f"""
                                    INSERT INTO publications (
                                        docid, 
                                        url, 
                                        domain, 
                                        title, 
                                        src_lang, 
                                        src_abstract,
                                        tgt_lang,
                                        tgt_abstract,
                                        model, 
                                        tgt_langid, 
                                        tgt_langid_score,
                                        length_ratio, 
                                        cometkiwi, 
                                        publication_date,
                                        processing_date,
                                        notif_ready,
                                        logs
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    
                                    """, 
                                    data_to_insert
                                )

if __name__ == "__main__":
    CLI(main, description=__doc__)