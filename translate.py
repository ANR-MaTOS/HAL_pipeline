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

def create_timestamp(date): 
    # offline mode
    if date: 
        dt_object = datetime.strptime(date, "%d_%m_%Y")
        datestamp = dt_object.strftime("%d_%m_%Y")
        return datestamp 
    # online mode
    if date is None: 
        today = date.today() 
        yesterday = today - timedelta(days = 1)
        datestamp = yesterday.strftime("%d_%m_%Y")
        return datestamp 

def main(tasks: List[dict], models: List[dict] = None):
    date = os.getenv("DATE")
    datestamp = create_timestamp(date)

    for model in models:
        model_name = model["name"]
        print(model_name)
        print("LLM arguments:", model["llm_arguments"])
        llm = LLM(**model["llm_arguments"])
        sampling_params = SamplingParams(**model["sampling_arguments"])  

        for task in tasks:
            if task.get("name") == "mt":             
                for subtask in task.get("subtasks",{}):
                    modes = subtask["modes"]
                    src_lang = subtask["src_lang"]
                    tgt_lang = subtask["tgt_lang"]
                    for mode in modes: 
                        print(f"{src_lang}>{tgt_lang} {mode} translation")

                        src_path = subtask["src_path"]
                        src_path = Template(src_path)
                        src_path = src_path.safe_substitute(model_name=model_name, mode=mode)
                        input_datefile = f"input_{datestamp}.json"
                        src_file = Path(src_path) / input_datefile 
                        print(f"Source file = {src_file}")

                        tgt_path = subtask["tgt_path"]
                        tgt_path = Template(tgt_path)
                        tgt_path = tgt_path.safe_substitute(model_name=model_name, mode=mode)
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
                                mt_input = dict() 
                                mt_input["docid"] = pub["docid"]
                                mt_input["src_abstract"] = pub["src_abstract"]
                                mt_input["src_len"] = pub["src_len"]
                                mt_input["prompt"] = pub["prompt"]
                                mt_input["prompt_len"] = pub["prompt_len"]
                                mt_content.append(mt_input)
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
                            
                        with open(tgt_file, "w", encoding="utf-8") as f:
                            json.dump(mt_content, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)