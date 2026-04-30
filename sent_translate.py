import json
import time
import torch
import vllm
print("vLLM location:", vllm.__file__)
from vllm import LLM, SamplingParams
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Torch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from jsonargparse import CLI
from datetime import date, timedelta, datetime
from typing import List
from pathlib import Path
from string import Template
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

def main(tasks: List[dict], models: List[dict] = None):
    print("Execution begins")
    
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)

    for model in models:
        model_name = model["name"]
        print(model_name)
        # print("llm_arguments", model["llm_arguments"])
        llm = LLM(**model['llm_arguments'])
        sampling_params = SamplingParams(**model['sampling_arguments'])  

        for task in tasks:
            if task.get("name") == "translation":             
                for subtask in task.get("subtasks",{}):    
                    src_path = subtask["src_path"]
                    src_path = Template(src_path)
                    src_path = src_path.safe_substitute(model_name=model_name)
                    datefile = f"input_{datestamp}.json"
                    src_file = Path(src_path) / datefile 
                    print(f"Source file = {src_file}")
                    
                    # Load input 
                    with open(src_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # verify that all publications have a docid and prompts 
                    mt_data = []
                    for pub in data:
                        entry = dict() 
                        if pub.get("docid") is not None and pub.get("prompts") is not None: 
                            entry["docid"] = pub["docid"]
                            entry["src_abstract"] = pub.get("src_abstract") 
                            entry["sentences"] = pub["prompts"]
                            mt_data.append(entry)
                    
                    # flattened list of prompts 
                    instructions = [] 
                    for entry in mt_data: 
                        sentences = entry["sentences"]
                        for sent in sentences: 
                            instructions.append(sent["prompt"])
                        
                    print(f"{len(instructions)} prompts loaded")
                    
                    tgt_path = subtask["tgt_path"]
                    tgt_path = Template(tgt_path)
                    tgt_path = tgt_path.safe_substitute(model_name=model_name)

                    Path(tgt_path).mkdir(parents=True, exist_ok=True)
                    print(f"Target path = {tgt_path}")
                                        
                    # generate 
                    t0 = time.time()
                    outputs = llm.generate(instructions, sampling_params)
                    print(f"Generation finished in {time.time() - t0} seconds")

                    # store
                    output_texts = [output.outputs[0].text for output in outputs]
                    assert len(instructions) == len(output_texts) 
                    cursor = 0
                    for entry in mt_data:
                        for sent in entry["sentences"]:
                            sent["tgt_txt"] = output_texts[cursor]
                            cursor += 1
                    assert cursor == len(output_texts)

                    # write output file 
                    output_file = Path(tgt_path) / f"output_{datestamp}.json"
                    with open(output_file, "w", encoding="utf-8") as file:
                        json.dump(mt_data, file, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)