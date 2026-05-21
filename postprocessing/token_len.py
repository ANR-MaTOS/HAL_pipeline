from collections import defaultdict
from datetime import date
from datetime import timedelta
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
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-9B-Instruct")
print("Tokenizer loaded")

def get_length(txt):
    token_len = None 
    if txt and isinstance(txt, str):
        tokens = tokenizer.encode(txt)
        token_len = len(tokens)
    return token_len 

def get_translations(filepath):
    translations = dict() 
    translation_files = glob.glob(filepath + "/output*.json")
    for f in translation_files: 
        trans = json.load(open(f, "r", encoding="utf-8")) 
        trans = {entry["docid"]: entry for entry in trans}
        translations.update(trans)
    return translations 

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

def main(tasks: List[dict]):
    print("Token counting begins")
    languages = ["en", "fr"]
    for task in tasks: 
        if task.get("name") == "postprocessing":           
            for subtask in task.get("subtasks",{}): 
                if subtask.get("name") == "length_ratio": 
                    print("Length ratio calculation")
                    data = dict() 
                    for lang in languages: 
                        exp_lang = lang 
                        
                        doc0_path = subtask["doc0_path"]
                        doc0_path = string.Template(doc0_path)
                        doc0_path = doc0_path.safe_substitute(lang = exp_lang)
                        doc0_translations = get_translations(doc0_path)

                        doc1_path = subtask["doc1_path"]
                        doc1_path = string.Template(doc1_path)
                        doc1_path = doc1_path.safe_substitute(lang = exp_lang)
                        doc1_translations = get_translations(doc1_path)

                        doc2_path = subtask["doc2_path"]
                        doc2_path = string.Template(doc2_path)
                        doc2_path = doc2_path.safe_substitute(lang = exp_lang)
                        doc2_translations = get_translations(doc2_path)

                        seg_path = subtask["segment_path"]
                        seg_path = string.Template(seg_path)
                        seg_path = seg_path.safe_substitute(lang = exp_lang)
                        seg_translations = get_translations(seg_path)

                        sent_path = subtask["sent_path"]
                        sent_path = string.Template(sent_path)
                        sent_path = sent_path.safe_substitute(lang = exp_lang)
                        sent_translations = get_translations(sent_path)

                        output_path = subtask["postprocessed_path"]
                        output_path = string.Template(output_path)
                        output_path = output_path.safe_substitute(lang = exp_lang)

                        translations = dict() 
                        modes = {"doc_0shot":doc0_translations, "doc_1shot":doc1_translations, "doc_2shot":doc2_translations, "segment_0shot":seg_translations, "sentence_0shot":sent_translations}
                        # only retain abstracts that have been translated in all modes 
                        for docid in doc0_translations.keys(): 
                            if (docid in doc1_translations.keys()) and (docid in doc2_translations.keys()) and (docid in seg_translations.keys()) and (docid in sent_translations.keys()): 
                                translations[docid] = defaultdict(dict)
                                translations[docid]["source"] = doc0_translations[docid]["src_abstract"]
                                translations[docid]["source_len"] = doc0_translations[docid]["src_len"]
                                for k, v in modes.items(): 
                                    target_text = get_target_txt(k, v[docid])
                                    translations[docid][k]["target_txt"] = target_text 
                                    target_len = get_length(target_text)
                                    translations[docid][k]["target_len"] = target_len
                                    translations[docid][k]["length_ratio"] = target_len / doc0_translations[docid]["src_len"]
                        postprocessed_f = Path(output_path) / "postprocessed.json"
                        with open(postprocessed_f, "w", encoding="utf-8") as output_f: 
                            json.dump(translations, output_f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main, description=__doc__)