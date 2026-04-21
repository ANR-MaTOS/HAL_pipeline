from datetime import date, datetime, timedelta
from pathlib import Path
from source_filtering import get_lang, normalise
import json 
import requests 
import os 
import re 

def get_pubs(fields:list, date=None, coll_name="INRIA"):
    date = os.getenv("DATE")
    if date is None: 
        datefield = "[NOW-1DAY/DAY TO NOW/DAY]"
    else: 
        dt_object = datetime.strptime(date, "%d_%m_%Y")
        start = dt_object.strftime("%Y-%m-%dT00:00:00Z")
        end = (dt_object + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        datefield = f"[{start} TO {end}]"
    cursor = "*"
    docs = []
    while True: 
        url = f"https://api.archives-ouvertes.fr/search/?q=collCode_s:{coll_name}&fq=submittedDate_tdate:{datefield}&fl={",".join(fields)}&rows=100&sort=docid%20asc&cursorMark={cursor}"
        res = requests.get(url).json() 
        docs.extend(res["response"]["docs"])
        next_cursor = res["nextCursorMark"]
        if cursor == next_cursor: 
            break 
        else: 
            cursor = next_cursor
    return docs  
    
def save_publications(content, filepath): 
    directory = Path(filepath).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f: 
        json.dump(content, f, ensure_ascii=False, indent=2)

def filter_abstracts(publications:list):
    bilingual = []
    only_english = []
    only_french = []
    for pub in publications: 
        en_abstract = pub.get("en_abstract_s")
        if en_abstract and isinstance(en_abstract, list): 
            en_abstract = en_abstract[0]
        fr_abstract = pub.get("fr_abstract_s")
        if fr_abstract and isinstance(fr_abstract, list): 
            fr_abstract = fr_abstract[0]
        en_len = bool(en_abstract and len(en_abstract) >= 40)
        fr_len = bool(fr_abstract and len(fr_abstract) >= 40) 
        if en_len and fr_len:
            bilingual.append(pub)
        elif en_len and not fr_len: 
            only_english.append(pub)
        elif fr_len and not en_len: 
            only_french.append(pub)
    return [bilingual, only_english, only_french]

def normalise_langid(expected_lang:str, publications:list, langid_threshold=0.8, length_threshold=40):
    accepted = []
    rejected = [] 
    abstract_field = f"{expected_lang}_abstract_s"
    for pub in publications: 
        abstract = pub[abstract_field][0]
        abstract = normalise(abstract)
        pub[abstract_field][0] = abstract 
        abstract_len = len(abstract)
        lang, langid_score = get_lang(abstract)
        if lang == expected_lang and langid_score >= langid_threshold and abstract_len >= length_threshold: 
            accepted.append(pub)
        else: 
            rejected.append(pub)
    return [accepted, rejected]

if __name__=="__main__":
    fields = [
        "docid",
        "label_s",
        "uri_s",
        "title_s",
        "en_title_s",
        "fr_title_s",
        "keyword_s",
        "en_keyword_s",
        "fr_keyword_s",
        "abstract_s",
        "en_abstract_s",
        "fr_abstract_s",
        "authIdFormPerson_s",
        "authIdForm_i",
        "authFullName_s",
        "submittedDate_tdate",
        "primaryDomain_s",
        "collCode_s",
        ]
    res = get_pubs(fields)
    print(f"Fetched {len(res)} publications")
    datestamp = os.getenv("DATE")
    if datestamp is None: 
        today = date.today()
        yesterday = today - timedelta(days=1)
        datestamp = yesterday.strftime("%d_%m_%Y")
    res = [pub for pub in res if pub.get("docid") and (pub.get("en_abstract_s") or pub.get("fr_abstract_s"))]
    save_publications(res, f"metadata/inria_{datestamp}.json")

    # filter by abstract fields and save 
    bilingual, only_english, only_french = filter_abstracts(res)
    save_publications(bilingual, f"bilingual/all/{datestamp}.json")
    save_publications(only_english, f"source/en/all/{datestamp}.json")
    save_publications(only_french, f"source/fr/all/{datestamp}.json")

    en_accepted, en_rejected = normalise_langid("en", only_english)
    fr_accepted, fr_rejected = normalise_langid("fr", only_french)
    save_publications(en_accepted, f"source/en/accepted/checked_en_{datestamp}.json")
    save_publications(en_rejected, f"source/en/rejected/rejected_en_{datestamp}.json")
    save_publications(fr_accepted, f"source/fr/accepted/checked_fr_{datestamp}.json")
    save_publications(fr_rejected, f"source/fr/rejected/rejected_fr_{datestamp}.json")



        












    