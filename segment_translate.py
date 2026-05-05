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
from typing import List
from pathlib import Path
from datetime import date, timedelta, datetime
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
    print("Segment translation begins")
    data_date = os.getenv("DATE")
    datestamp = create_timestamp(data_date)
    for model in models:
        model_name = model["name"]
        print(model_name)
        llm = LLM(**model['llm_arguments'])
        sampling_params = SamplingParams(**model['sampling_arguments'])  

        for task in tasks:
            if task.get("name") == "translation":             
                for subtask in task.get("subtasks",{}):
                    
                    src_path = subtask["src_path"]
                    input_datefile = f"input_{datestamp}.json"
                    src_file = Path(src_path) / input_datefile 
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
                            entry["segments"] = pub["prompts"]
                            mt_data.append(entry)
                    
                    # flattened list of prompts 
                    instructions = [] 
                    for entry in mt_data: 
                        segments = entry["segments"]
                        for seg in segments: 
                            instructions.append(seg["prompt"])
                        
                    print(f"{len(instructions)} prompts loaded")
                    
                    tgt_path = subtask["tgt_path"]
                    Path(tgt_path).mkdir(parents=True, exist_ok=True)
                    print(f"Target path = {tgt_path}")
                                        
                    # generate 
                    t0 = time.time()
                    outputs = llm.generate(instructions, sampling_params)
                    print("Generation finished in", time.time() - t0, "seconds")

                    # store
                    output_texts = [output.outputs[0].text for output in outputs]
                    assert len(instructions) == len(output_texts) 
                    cursor = 0
                    for entry in mt_data:
                        for segment in entry["segments"]:
                            segment["tgt_txt"] = output_texts[cursor]
                            cursor += 1
                    assert cursor == len(output_texts)

                    output_datefile = f"output_{datestamp}.json"
                    tgt_f = Path(tgt_path) / output_datefile
                    # write output file 
                    with open(tgt_f, "w", encoding="utf-8") as file:
                        json.dump(mt_data, file, ensure_ascii=False, indent=2)

if __name__=="__main__":
    CLI(main,description=__doc__)