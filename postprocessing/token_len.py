from collections import defaultdict
from jsonargparse import CLI
from pathlib import Path
from transformers import AutoTokenizer 
from typing import List
import glob
import json 
import os
import re  
import string 
import torch 
from huggingface_hub import login
from datetime import date, timedelta, datetime

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")
login(token=hf_token)

# tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-22B-Instruct")
# print("Tokenizer loaded")

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

def get_length(tokenizer, txt):
    token_len = None 
    if txt and isinstance(txt, str):
        tokens = tokenizer.encode(txt)
        token_len = len(tokens)
    return token_len 

def get_translations(filepath, datestamp):
    translations = dict() 
    translation_f = Path(filepath) / f"output_{datestamp}.json"
    translations = json.load(open(translation_f, "r", encoding="utf-8")) 
    translations = {entry["docid"]: entry for entry in translations}
    return translations

"""
def get_target_txt(mode, entry): 
    if mode.startswith("doc"): 
        target_txt =  entry["tgt_abstract"]
    elif mode == "segment_0shot": 
        target_txt = " ".join([seg["tgt_txt"] for seg in entry["segments"]])
    elif mode == "sentence_0shot": 
        target_txt = " ".join([sent["tgt_txt"] for sent in entry["sentences"]])
    else: 
        target_txt = None 
    return target_txt 
""" 

def main(tasks: List[dict], models: List[dict] = None):
    print("Token counting begins")
    languages = ["en", "fr"]
    for task in tasks: 
        if task.get("name") == "postprocessing":        
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "length_ratio": 
                    print("Length ratio calculation")
                    for model in models: 
                        model_name = model["name"] 
                        tokenizer = AutoTokenizer.from_pretrained(model["llm_arguments"]["model"])   
                        data = dict() 
                        data_date = os.getenv("DATE")
                        datestamp = create_timestamp(data_date)
                        for lang in languages: 
                            translation_path = task.get("translation_path")
                            postprocessed_path = task.get("postprocessed_path") 
                            expected_lang = lang 
                            
                            translation_path = string.Template(translation_path).safe_substitute(model_name = model_name, lang = expected_lang)
                            translations = get_translations(translation_path, datestamp)

                            postprocessed_path = string.Template(postprocessed_path)
                            postprocessed_path = postprocessed_path.safe_substitute(model_name = model_name, lang = expected_lang)
                            Path(postprocessed_path).mkdir(exist_ok=True, parents=True)

                            # translations = dict() 
                            for docid, entry in translations.items(): 
                                source_len = entry.get("src_len")
                                target_len = get_length(tokenizer, entry["tgt_abstract"])
                                if source_len and target_len: 
                                    length_ratio = target_len / source_len 
                                    entry["length_ratio"] = length_ratio 

                            postprocessed_f = Path(postprocessed_path) / f"postprocessed_{datestamp}.json"
                            with open(postprocessed_f, "w", encoding="utf-8") as output_f: 
                                json.dump(translations, output_f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main, description=__doc__)