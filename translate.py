import json
import time
import os
import torch
import vllm
print("vLLM location:", vllm.__file__)
from vllm import LLM, SamplingParams
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Torch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from jsonargparse import CLI
from datetime import date
from datetime import timedelta
from typing import List
from pathlib import Path
from string import Template
from datetime import date, timedelta, datetime

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
        print("LLM arguments:", model["llm_arguments"])
        llm = LLM(**model["llm_arguments"])
        sampling_params = SamplingParams(**model["sampling_arguments"])  

        for task in tasks:
            if task.get("name") == "translation":             
                for subtask in task.get("subtasks",{}):
                    src_lang = subtask["src_lang"]
                    tgt_lang = subtask["tgt_lang"]
                
                    print(f"{src_lang}>{tgt_lang} translation")

                    src_path = subtask["src_path"]
                    src_path = Template(src_path)
                    src_path = src_path.safe_substitute(model_name=model_name)
                    input_datefile = f"input_{datestamp}.json"
                    src_file = Path(src_path) / input_datefile 
                    print(f"Source file = {src_file}")

                    tgt_path = subtask["tgt_path"]
                    tgt_path = Template(tgt_path)
                    tgt_path = tgt_path.safe_substitute(model_name=model_name)
                    Path(tgt_path).mkdir(parents=True, exist_ok=True)
                    out_datefile = f"output_{datestamp}.json"
                    tgt_file = Path(tgt_path) / out_datefile
                    print(f"Target file = {tgt_file}")
                
                    with open(src_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # prepare input
                    mt_content = []                    
                    for pub in data:
                        if pub["status"] == "standard":
                            entry = dict() 
                            entry["docid"] = pub["docid"]
                            entry["url"] = pub["url"]
                            entry["domain"] = pub["domain"]
                            entry["title"] = pub["title"]
                            entry["src_abstract"] = pub["src_abstract"]
                            entry["src_len"] = pub["src_len"]
                            entry["prompt"] = pub["prompt"]
                            entry["prompt_len"] = pub["prompt_len"]
                            mt_content.append(entry)
                        else: 
                            print(f"{pub['docid']} left out because the prompt is too long")     

                    print(f"{len(mt_content)} prompts loaded")                                        
                    instructions = [entry["prompt"] for entry in mt_content]
                
                    # generate 
                    t0 = time.time()
                    outputs = llm.generate(instructions, sampling_params)
                    print("Generation finished in", time.time() - t0, "seconds")

                    # store
                    output_texts = [output.outputs[0].text for output in outputs]
                    assert len(mt_content) == len(output_texts) 
                    for mt_entry, output in zip(mt_content, output_texts):
                        mt_entry["tgt_abstract"] = output 

                    print(f"{len(mt_content)} abstracts were translated")

                    print(tgt_file)
                        
                    with open(tgt_file, "w", encoding="utf-8") as f:
                        json.dump(mt_content, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)