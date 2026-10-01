import spacy
from pathlib import Path
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin

nlp_en = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)

nlp_fr = spacy.load(
    "fr_core_news_sm",
    disable=["ner", "parser"]
)
 
def create_matcher(docbin_path, nlp): 
    doc_bin = DocBin().from_disk(docbin_path)
    patterns = list(doc_bin.get_docs(nlp.vocab))
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    for i, term in enumerate(patterns): 
        matcher.add(f"TERM_{i}", [term]) 
    return matcher 

TERMINOLOGY_CONFIGS = [
    ("nlp_en", "patterns/nlp_en.spacy", nlp_en),
    ("nlp_fr", "patterns/nlp_fr.spacy", nlp_fr),
    ("maths_en", "patterns/maths_en.spacy", nlp_en),
    ("maths_fr", "patterns/maths_fr.spacy", nlp_fr),
    ("computer_science_en", "patterns/computer_science_en.spacy", nlp_en),
    ("computer_science_fr", "patterns/computer_science_fr.spacy", nlp_fr),
    ("psychology_en", "patterns/psychology_en.spacy", nlp_en),
    ("psychology_fr", "patterns/psychology_fr.spacy", nlp_fr), 
]

matchers = {
    key: create_matcher(docbin_path=path, nlp=nlp)
    for key, path, nlp in TERMINOLOGY_CONFIGS
    }

def detect_matches(src_text, src_lang, domain):
    if src_lang == "en": 
        nlp = nlp_en  
    elif src_lang == "fr": 
        nlp = nlp_fr 
    else: 
        raise ValueError("Unsupported language")
    doc = nlp(src_text)
    matched_terms = set() 
    matcher_key = ""
    if domain in ["info.info-cl", "info.info-ir"]: 
        matcher_key = f"nlp_{src_lang}"
    elif domain.startswith("info"): 
        matcher_key = f"computer_science_{src_lang}"
    elif domain.startswith("math"): 
        matcher_key = f"maths_{src_lang}" 
    elif domain in ["scco.psyc", "sdv.mhep.psm", "sdv.neu.pc", "shs.psy"]: 
        matcher_key = f"psychology_{src_lang}"
    if matcher_key: 
        matcher = matchers[matcher_key] 
        matches = matcher(doc)
        if matches:  
            print(f"Matches found: {matches}")
            for match_id, start, end in matches: 
                matched_terms.add(doc[start:end].text)
    return matched_terms
