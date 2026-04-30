from pathlib import Path
from typing import Optional, List
from jsonargparse import CLI
import json
import sys
import string
from transformers import AutoTokenizer
import torch
import os
from huggingface_hub import login
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=hf_token)
from string import Template
from datetime import date, timedelta, datetime
from string import Template

MT_SYS_MESSAGE = Template("You are a professional translator of scientific documents. Translate the following text from $src_lang into $tgt_lang. The target text must have the same number of paragraphs as the source text. Reply only with the translated text.")

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

def make_prompt(src, template_string, src_lang, tgt_lang, tokenizer=None, no_sys_message = False):
    source_lang = "French" if src_lang == "fr" else "English"
    target_lang = "English" if tgt_lang == "en" else "French"
    template = string.Template(template_string)
    prompt = template.safe_substitute(src_lang_full = source_lang, src_txt = src, tgt_lang_full = target_lang)
    sys_message = MT_SYS_MESSAGE.substitute({"src_lang":source_lang, "tgt_lang":target_lang})
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": prompt},
        ]
        if no_sys_message:
            messages = [{"role": "user", "content": prompt},]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt 

def make_prompts_for_inference(src_path, datestamp, src_lang, tgt_lang, template_string, tokenizer = None, no_sys_message = False):    
    datefile = f"sentences_{datestamp}.json"
    with open(Path(src_path) / datefile, "r", encoding="utf-8") as f: 
        data = json.load(f) 
    src_list = []
    abstract_field = f"{src_lang}_abstract_s"
    for entry in data: 
        if "docid" in entry.keys() and "sentences" in entry.keys(): 
            src_list.append({"docid":entry["docid"], "src_abstract":entry[abstract_field][0], "sentences":entry["sentences"]})
    res = []
    for src in src_list:
        docid = src["docid"]
        src_abstract = src["src_abstract"] 
        prompts = []
        sentences = src["sentences"]
        for sent in sentences:
            prompt_entry = dict()
            rank = sent["rank"]
            src_txt = sent["src_txt"]
            src_tokens = tokenizer.encode(src_txt)
            src_len = len(src_tokens)
            prompt = make_prompt(src_txt, template_string, src_lang, tgt_lang, tokenizer = tokenizer, no_sys_message = no_sys_message)
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            status = "standard" if prompt_len <= 4096 else "too_long"
            prompt_entry = {"rank":rank, "src_txt":src_txt, "src_len":src_len, "prompt":prompt, "prompt_len":prompt_len, "status":status}
            prompts.append(prompt_entry)
        # what to keep? 
        res.append({"docid":docid, "src_abstract":src_abstract, "prompts":prompts})
    return res
    
def main(tasks: List[dict], models: List[dict] = None):
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for model in models:
        model_name = model["name"]
        print(model_name)
        print("llm_arguments", model["llm_arguments"]) 
        tokenizer = AutoTokenizer.from_pretrained(model["llm_arguments"]["model"])   

        for task in tasks:
            if task.get("name") == "prompt_preparation": 
                for subtask in task.get("subtasks",[]):
                    src_lang = subtask["src_lang"]
                    tgt_lang = subtask["tgt_lang"]
                    src_path = subtask["src_path"]
                    print(f"Source path = {src_path}")
                    tgt_path = subtask["tgt_path"]
                    tgt_path = Template(tgt_path)
                    tgt_path = tgt_path.safe_substitute(model_name=model_name)
                    Path(tgt_path).mkdir(exist_ok=True, parents=True)
                    print(f"Target path = {tgt_path}")
                    # make prompt
                    if model.get("chat_template", False):
                        instructions = make_prompts_for_inference(
                            src_path, datestamp, src_lang, tgt_lang, template_string = model["template"], tokenizer = tokenizer, no_sys_message = model.get("no_sys_message", False))
                    else:
                        instructions = make_prompts_for_inference(
                            src_path, datestamp, src_lang, tgt_lang, template_string = model["template"], tokenizer = tokenizer)
                    prompt_file = Path(tgt_path) / f"input_{datestamp}.json"
                    with open(prompt_file, "w", encoding="utf-8") as f:
                        json.dump(instructions, f, ensure_ascii=False, indent=2)
                        
if __name__ == '__main__':
    CLI(main, description=__doc__)
