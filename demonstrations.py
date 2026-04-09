from collections import defaultdict
from functools import cache
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle 
import re 
import spacy 
import time 
import bm25s

@cache 
def read_corpus(corpus="acadata.csv"): 
    cache_path = "acadata_tokenized.pkl"
    if Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    data = defaultdict(list)
    with open(corpus, "r", encoding="utf-8") as corpus_f:
        lines = corpus_f.readlines()
    for line in lines: 
        en_txt, fr_txt = re.split("\t", line)
        en_txt = en_txt.strip() 
        data["en_texts"].append(en_txt)
        fr_txt = fr_txt.strip()
        data["fr_texts"].append(fr_txt)
        # en_tokenized = lemmatize("en", en_txt)
        # data["en_tokenized"].append(en_tokenized)
        # fr_tokenized = lemmatize("fr", fr_txt)
        # data["fr_tokenized"].append(fr_tokenized)
    data["en_tokenized"] = lemmatize("en", data["en_texts"])
    data["fr_tokenized"] = lemmatize("fr", data["fr_texts"])
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data 

@cache 
def load_model(lang):
    if lang == "en": 
        model = "en_core_web_sm"
    elif lang == "fr": 
        model = "fr_core_news_sm"
    else: 
        raise ValueError(f"Unsupported language: {lang}")
    return spacy.load(model, exclude=["parser", "ner", "entity_linker", "entity_ruler", "textcat", "textcat_multilabel"])

def lemmatize(lang, text):
    nlp = load_model(lang)
    print("Spacy model loaded")
    # lemmatizer = nlp.get_pipe("lemmatizer")
    if isinstance(text, str):
        doc = nlp(text)
        return [token.lemma_ for token in doc]
    elif isinstance(text, list): 
        return [[token.lemma_ for token in doc] for doc in tqdm(nlp.pipe(text), total=len(text), desc="Lemmatizing")]

@cache 
def get_retriever(src_lang): 
    data = read_corpus()
    if src_lang == "en": 
        tokenized_corpus = data["en_tokenized"]
    elif src_lang == "fr": 
        tokenized_corpus = data["fr_tokenized"]
    else: 
        raise ValueError(f"Unsupported language: {src_lang}")
    retriever = bm25s.BM25()
    print(f"Indexing BM25s for {src_lang}...")
    retriever.index(tokenized_corpus) 
    # retriever.save(f"bm25s_index_{src_lang}", corpus=data[f"{src_lang}_texts"])
    return retriever  

def get_examples(src_lang, src_txt, num_examples):
    data = read_corpus()
    retriever = get_retriever(src_lang)
    tokenized_query = lemmatize(src_lang, src_txt)
    results, scores = retriever.retrieve([tokenized_query], k=num_examples)
    # indices = results.indices[0]
    indices = results[0]
    doc_scores = scores[0]
    best_examples = []
    for i, idx in enumerate(indices): 
        best_examples.append({
            "en": data["en_texts"][idx], 
            "fr": data["fr_texts"][idx], 
            "score": float(doc_scores[i]),
        })
    return best_examples 