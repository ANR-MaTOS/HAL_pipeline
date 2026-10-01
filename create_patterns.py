from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin
import duckdb 
import pickle 
import spacy 
import json 

nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp_fr = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

def create_patterns(terminology): 
    """
    Creates phrase patterns based on an available terminology
    """
    en_terms = [term["en"] for term in terminology.values()]
    fr_terms = [term["fr"] for term in terminology.values()]
    patterns_en = list(nlp_en.pipe(en_terms))
    patterns_fr = list(nlp_fr.pipe(fr_terms))
    return patterns_en, patterns_fr 

def store_patterns(patterns_en, patterns_fr, domain): 
    """
    Serializes patterns 
    """
    docbin_en = DocBin()
    for en_doc in patterns_en: 
        docbin_en.add(en_doc)
    docbin_en.to_disk(f"patterns/{domain}_en.spacy")
    docbin_fr = DocBin()
    for fr_doc in patterns_fr: 
        docbin_fr.add(fr_doc)
    docbin_fr.to_disk(f"patterns/{domain}_fr.spacy")

if __name__ == "__main__": 

    # load terminologies 
    with open("terms/informatique.json", "r", encoding="utf-8") as info_f: 
        info_terms = json.load(info_f) 
    with open("terms/maths.json", "r", encoding="utf-8") as maths_f: 
        maths_terms = json.load(maths_f)
    with open("terms/psychologie.json", "r", encoding="utf-8") as psy_f: 
        psy_terms = json.load(psy_f)
    with open("terms/tal.json", "r", encoding="utf-8") as tal_f: 
        tal_terms = json.load(tal_f)

    # create and store patterns 
    terminologies = [info_terms, maths_terms, psy_terms, tal_terms]
    domains = ["computer_science", "maths", "psychology", "nlp"]
    for terminology, domain in zip (terminologies, domains): 
        patterns_en, patterns_fr = create_patterns(terminology) 
        store_patterns(patterns_en, patterns_fr, domain) 

    # create and populate database 
    DISCIPLINE_MAP = {
        "Informatique": "Computer Science",
        "maths": "Mathematics",
        "tal": "Natural Language Processing",
        "Psychologie": "Psychology",
        }
    data_to_insert = []
    for terminology in terminologies:
        for item in terminology.values():
            domain_name = DISCIPLINE_MAP[item["discipline"]]
            data_to_insert.append((item["en"], item["fr"], domain_name))
    print(f"{len(data_to_insert)} entries to be inserted in the database")
    with duckdb.connect("glossary.duckdb") as con:
    # Create the target table if it doesn't already exist
        con.execute(
            """
            CREATE TABLE terms (
                en VARCHAR,
                fr VARCHAR,
                domain VARCHAR
            )
        """
        ) 
        con.executemany(
        """
        INSERT INTO terms (en, fr, domain)
        VALUES (?, ?, ?)
        """,
            data_to_insert,
        )