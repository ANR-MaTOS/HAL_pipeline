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
from demonstrations import get_examples
import re 
from transformers import AutoConfig
from datetime import date, timedelta, datetime
from phrase_matcher import detect_matches
import pickle 
from functools import cache
import duckdb 

MT_SYS_MESSAGE = Template("You are a professional translator of scientific documents. Translate the following text from $src_lang into $tgt_lang. $domain_instruction The target text must have the same number of paragraphs as the source text. Reply only with the translated text.")

@cache 
def get_HAL_domains(dico="hal_domains.json"): 
    hal_domains = {} 
    with open(dico, "r", encoding="utf-8") as f: 
        hal_domains = json.load(f) 
    return hal_domains 

def create_domain_instruction(domain):
    hal_domains = get_HAL_domains()
    domain_parts = domain.split(".")
    subdomains = [
        ".".join(domain_parts[:i])
        for i in range(1, len(domain_parts) + 1)
    ]
    matching_domains = [
        subdomain
        for subdomain in subdomains
        if subdomain in hal_domains
    ]
    if matching_domains:
        longest_domain = max(matching_domains, key=len)
        full_domain = hal_domains[longest_domain]
        return (
            f"The text is an abstract of a scholarly publication "
            f"in the field of {full_domain}."
        )
    return ""
        
def get_term_translations(matches, src_lang, domain): 
    term_translations = ""
    tgt_lang = "en" if src_lang == "fr" else "fr"

    domain = "Natural Language Processing" if domain in ["info.info-cl", "info.info-ir"] else domain
    domain = "Computer Science" if domain.startswith("info") else domain 
    domain = "Mathematics" if domain.startswith("math") else domain 
    domain = "Psychology" if domain in ["scco.psyc", "sdv.mhep.psm", "sdv.neu.pc", "shs.psy"] else domain 
    for src_term in matches: 
        print(f"{src_term} matched for the domain of {domain}")
        with duckdb.connect("glossary.duckdb") as con:
            res = con.sql(f"SELECT {tgt_lang} FROM terms WHERE {src_lang} = ? AND domain = ?", params=[src_term, domain]).fetchone() 
            print(f"SELECT {tgt_lang} FROM glossary WHERE {src_lang} = {src_term} AND domain = {domain}")
            if res: 
                term_translations += f"\"{src_term}\" = \"{res[0]}\"\n"
    return term_translations 

def get_glossary(src_text, src_lang, domain):
    glossary_content = "" 
    matches = detect_matches(src_text, src_lang, domain)
    if len(matches) > 0: 
        # glossary = read_glossary()
        term_translations = get_term_translations(matches, src_lang, domain) 
        if term_translations.strip(): 
            glossary_content = "\nThe following glossary could be helpful:\n"
            glossary_content += term_translations 
    return glossary_content  

def make_prompt(src, template_string, src_lang, tgt_lang, domain, glossary_use, few_shot_k, tokenizer=None, no_sys_message=False):
    src_lang_full = "French" if src_lang == "fr" else "English"
    tgt_lang_full = "English" if tgt_lang == "en" else "French"
    domain_instruction = create_domain_instruction(domain)
    sys_message = MT_SYS_MESSAGE.substitute({"src_lang":src_lang_full, "tgt_lang":tgt_lang_full, "domain_instruction":domain_instruction})
    sys_message = " ".join(sys_message.split())
    template = string.Template(template_string)
    glossary_content = ""
    if glossary_use == True: 
        glossary_content = get_glossary(src, src_lang, domain)
    prompt = template.safe_substitute(src_lang_full=src_lang_full, src_txt=src, glossary=glossary_content, tgt_lang_full=tgt_lang_full)
    demonstrations = []
    if few_shot_k > 0: 
        demonstrations = get_examples(src_lang, src, few_shot_k)
    if tokenizer is not None:
        messages = []
        messages.append({"role": "system", "content": sys_message})
        if demonstrations:
            for d in demonstrations: 
                d_src = d[src_lang]
                d_tgt = d[tgt_lang]
                messages.append({"role": "user", "content": template.safe_substitute(src_lang_full=src_lang_full, src_txt=d_src, glossary="", tgt_lang_full=tgt_lang_full)})
                messages.append({"role": "assistant", "content": d_tgt})
        messages.append({"role": "user", "content": prompt})  

        if no_sys_message:
            if demonstrations: 
                for d in demonstrations: 
                    d_src = d[src_lang]
                    d_tgt = d[tgt_lang]
                    messages.append({"role": "user", "content": template.safe_substitute(src_lang_full=src_lang_full, src_txt=d_src, glossary="", tgt_lang_full=tgt_lang_full)})
                    messages.append({"role": "assistant", "content": d_tgt})
            messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt 
    return None 

def make_prompts_for_inference(src_path, datestamp, src_lang, tgt_lang, template_string, max_len, glossary_use = True, few_shot_k = 1, tokenizer = None, no_sys_message = False):    
    datefile = f"checked_{src_lang}_{datestamp}.json"
    with open(Path(src_path) / datefile, "r", encoding="utf-8") as f: 
        data = json.load(f) 
    src_list = []
    abstract_field = f"{src_lang}_abstract_s"
    for entry in data: 
        if "docid" in entry.keys() and abstract_field in entry.keys(): 
            src_list.append({"docid":entry["docid"], "url": entry["uri_s"], "domain":entry["primaryDomain_s"], "title":entry["title_s"][0], "src_abstract":entry[abstract_field][0]})
    res = []
    for src in src_list:
        docid = src["docid"]
        url = src["url"]
        domain = src["domain"]
        title = src["title"]
        abstract_s = src["src_abstract"] 
        print(f"Preparing {src_lang}->{tgt_lang} prompt for {docid}")
        prompt = make_prompt(abstract_s, template_string, src_lang, tgt_lang, domain, glossary_use, few_shot_k, tokenizer = tokenizer, no_sys_message = no_sys_message)
        if prompt is not None: 
            src_tokens = tokenizer.encode(abstract_s)
            src_len = len(src_tokens) 
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            status = "standard" if prompt_len <= max_len else "too_long"
            res.append({"docid":docid, "url":url, "domain": domain, "title":title, "src_abstract":abstract_s, "src_len":src_len, "prompt":prompt, "prompt_len":prompt_len, "status":status})
    return res

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

def main(tasks: List[dict], models: List[dict] = None):
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for model in models:
        model_name = model["name"]
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model["llm_arguments"]["model"])   
        config = AutoConfig.from_pretrained(model["llm_arguments"]["model"])
        max_len = getattr(config, "max_position_embeddings", None)
        print("Maximum length:", max_len)

        for task in tasks:
            if task.get("name") == "prompt_preparation": 
                for subtask in task.get("subtasks",[]): 
                    glossary_use = subtask["glossary"]
                    few_shot_k = subtask["few_shot"]
                    src_lang = subtask["src_lang"]
                    tgt_lang = subtask["tgt_lang"]
                    print(f"Preparing tokens for {src_lang}>{tgt_lang} translation")
                    print(f"Glossary mode: {glossary_use}")
                    print(f"Number of demonstrations in prompts: {few_shot_k}")
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
                            src_path, datestamp, src_lang, tgt_lang, template_string = model["template"], max_len = max_len, glossary_use = glossary_use, few_shot_k = few_shot_k, tokenizer = tokenizer, 
                            no_sys_message = model.get("no_sys_message", False))
                    else:
                        instructions = make_prompts_for_inference(
                            src_path, datestamp, src_lang, tgt_lang, template_string = model["template"], max_len = max_len, glossary_use = glossary_use, few_shot_k = few_shot_k, tokenizer = tokenizer)
                        
                    prompt_filename = Path(tgt_path) / f"input_{datestamp}.json"
                    with open(prompt_filename, "w", encoding="utf-8") as f:
                        json.dump(instructions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CLI(main, description=__doc__)