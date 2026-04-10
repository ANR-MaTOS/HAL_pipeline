from pathlib import Path
from typing import Optional, List
from jsonargparse import CLI
import json
import sys
import string
import os 
from transformers import AutoTokenizer
import torch
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
from string import Template
from datetime import date
from datetime import timedelta
from demonstrations import get_examples
import re 

today = date.today() 
yesterday = today - timedelta(days = 1)

MT_SYS_MESSAGE = Template("You are a professional translator of scientific documents. Translate the following text from $src_lang into $tgt_lang. The target text must have the same number of paragraphs as the source text. Reply only with the translated text.")

def make_prompt(src, template_string, src_lang, tgt_lang, num_examples=0, tokenizer=None, no_sys_message=False):
    src_lang_full = "French" if src_lang == "fr" else "English"
    tgt_lang_full = "English" if tgt_lang == "en" else "French"
    sys_message = MT_SYS_MESSAGE.substitute({"src_lang":src_lang_full, "tgt_lang":tgt_lang_full})
    template = string.Template(template_string)
    prompt = template.safe_substitute(src_lang_full=src_lang_full, src_txt=src, tgt_lang_full=tgt_lang_full)
    demonstrations = []
    if num_examples > 0: 
        demonstrations = get_examples(src_lang, src, num_examples)
    if tokenizer is not None:
        messages = []
        messages.append({"role": "system", "content": sys_message})
        if demonstrations:
            for d in demonstrations: 
                d_src = d[src_lang]
                d_tgt = d[tgt_lang]
                messages.append({"role": "user", "content": template.safe_substitute(src_lang_full=src_lang_full, src_txt=d_src, tgt_lang_full=tgt_lang_full)})
                messages.append({"role": "assistant", "content": d_tgt})
        messages.append({"role": "user", "content": prompt}) 
        if no_sys_message:
            if demonstrations: 
                for d in demonstrations: 
                    d_src = d[src_lang]
                    d_tgt = d[tgt_lang]
                    messages.append({"role": "user", "content": template.safe_substitute(src_lang_full=src_lang_full, src_txt=d_src, tgt_lang_full=tgt_lang_full)})
                    messages.append({"role": "assistant", "content": d_tgt})
            messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt 
    return None 

def make_prompts_for_inference(src_path, src_lang, tgt_lang, template_string, mode = "0-shot", tokenizer = None, no_sys_message = False):    
    datefile = f"checked_{src_lang}_{yesterday.day:02d}_{yesterday.month:02d}_{yesterday.year}.json"
    with open(Path(src_path) / datefile, "r", encoding="utf-8") as f: 
        data = json.load(f) 
    src_list = []
    abstract_field = f"{src_lang}_abstract_s"
    for entry in data: 
        if "docid" in entry.keys() and abstract_field in entry.keys(): 
            src_list.append({"docid":entry["docid"], "src_abstract":entry[abstract_field][0]})
    res = []
    for src in src_list:
        docid = src["docid"]
        abstract_s = src["src_abstract"]

        if mode == "0-shot":
            num_examples = 0 
        elif mode == "1-shot":
            num_examples = 1 
        elif mode == "2-shot": 
            num_examples = 2
        print(f"Making {mode} {src_lang}->{tgt_lang} prompt for {docid}")

        prompt = make_prompt(abstract_s, template_string, src_lang, tgt_lang, num_examples = num_examples, tokenizer = tokenizer, no_sys_message = no_sys_message)
        if prompt is not None: 
            src_tokens = tokenizer.encode(abstract_s)
            src_len = len(src_tokens) 
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            max_len = tokenizer.model_max_length
            status = "standard" if prompt_len <= max_len else "too_long"
            res.append({"docid":docid, "src_abstract":abstract_s, "src_len":src_len, "prompt":prompt, "prompt_len":prompt_len, "status":status})
    return res
    
def main(tasks: List[dict], models: List[dict] = None):
    for model in models:
        model_name = model["name"]
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model["llm_arguments"]["model"])   

        for task in tasks:
            if task.get("name") == "prompt_preparation": 
                for subtask in task.get("subtasks",[]): 
                    mode = subtask["mode"]
                    print(mode)
                    src_lang = subtask["src_lang"]
                    tgt_lang = subtask["tgt_lang"]

                    src_path = subtask["src_path"]
                    print(f"Source path = {src_path}")
                    
                    tgt_path = subtask["tgt_path"]
                    tgt_path = string.Template(tgt_path)
                    tgt_path = tgt_path.safe_substitute(model_name = model_name)
                    print(f"Target path = {tgt_path}")
                    Path(tgt_path).mkdir(exist_ok=True, parents=True)
                    
                    # make prompt
                    if model.get("chat_template", False):
                        instructions = make_prompts_for_inference(
                            src_path, src_lang, tgt_lang, template_string = model["template"], mode = mode, tokenizer = tokenizer, 
                            no_sys_message = model.get("no_sys_message", False))
                    else:
                        instructions = make_prompts_for_inference(
                            src_path, src_lang, tgt_lang, template_string = model["template"], mode = mode, tokenizer = tokenizer)
                        
                    prompt_filename = Path(tgt_path) / f"input_{yesterday.day:02d}_{yesterday.month:02d}_{yesterday.year}.json"
                    with open(prompt_filename, "w", encoding="utf-8") as f:
                        json.dump(instructions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CLI(main, description=__doc__)