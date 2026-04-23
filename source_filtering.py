from itertools import batched
import xml.etree.ElementTree as ET
import fasttext
import json  
import re 

def get_lang(text:str): 
    model_path = "lid.176.bin" # download from https://fasttext.cc/docs/en/language-identification.html 
    model = fasttext.load_model(model_path)
    """
    if model:
        res = model.predict(text, 1)
        if res[0][0].strip() == "__label__fr": 
            lang.add("fr")
        elif res[0][0].strip() == "__label__en": 
            lang.add("en")
    return lang 
    """ 
    res = model.predict(text, 1)
    lang = res[0][0].strip()[-2:]
    score = res[1][0]
    return lang, score

SUBSCRIPT_MAP = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ", "h": "ₕ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "p": "ₚ",
    "s": "ₛ", "t": "ₜ"
})

SUPERSCRIPT_MAP = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
    "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
    "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
    "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
    "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    "-":"⁻"
})

def convert_subscripts(text: str) -> str:
    def repl(match):
        content = match.group(1)
        return content.translate(SUBSCRIPT_MAP)
    return re.sub(r"<sub>(.*?)</sub>", repl, text)

def convert_superscripts(text: str) -> str:
    def repl(match):
        content = match.group(1)
        return content.translate(SUPERSCRIPT_MAP)
    return re.sub(r"<sup>(.*?)</sup>", repl, text)

def normalise_mmultiscripts(match): 
    def convert_subscripts(text) -> str: 
        if text == None: 
            return None
        return text.translate(SUBSCRIPT_MAP)
    def convert_superscripts(text) -> str: 
        if text == None: 
            return None
        return text.translate(SUPERSCRIPT_MAP)
    text = match.group(0)
    result = ""
    root = ET.fromstring(text)
    # mmultiscripts = root.find("mmultiscripts")
    mmultiscript_tags = [child.tag for child in root]
    mmultiscript_texts = [child.text for child in root]
    base = ""
    if mmultiscript_tags and mmultiscript_tags[0] == "mi": 
        base = mmultiscript_texts[0]
    if base and "mprescripts" in mmultiscript_tags: 
        idx = mmultiscript_tags.index("mprescripts")
        postsubscripts = []
        postsuperscripts = []
        presubscripts = []
        presuperscripts = []
        postsubscripts_txt = ""
        postsuperscripts_txt = "" 
        presubscripts_txt = "" 
        presuperscripts_txt = ""

        # get post-script subscript-superscript pairs 
        postscript_tags = mmultiscript_tags[1:idx]
        postscript_texts = mmultiscript_texts[1:idx]
        if len(postscript_texts) % 2 == 0: 
            for postscript in list(batched(postscript_texts, 2)):
                postsubscripts.append(postscript[0])
                postsuperscripts.append(postscript[1])
            postsubscripts = ["" if i is None else i for i in postsubscripts]
            postsubscripts_txt = "".join(postsubscripts)
            postsuperscripts = ["" if i is None else i for i in postsuperscripts] 
            postsuperscripts_txt = "".join(postsuperscripts)
                        
        # get pre-script subscript-superscript pairs 
        prescript_tags = mmultiscript_tags[idx+1:]
        prescript_texts = mmultiscript_texts[idx+1:]
        if len(prescript_texts) % 2 == 0: 
            for prescript in list(batched(prescript_texts, 2)):
                presubscripts.append(prescript[0])
                presuperscripts.append(prescript[1])
            presubscripts = ["" if i is None else i for i in presubscripts]
            presubscripts_txt = "".join(presubscripts)
            presuperscripts = ["" if i is None else i for i in presuperscripts]
            presuperscripts_txt = "".join(presuperscripts)

        result = convert_subscripts(presubscripts_txt) + convert_superscripts(presuperscripts_txt) + base + convert_subscripts(postsubscripts_txt) + convert_superscripts(postsuperscripts_txt) 

    elif base and "mprescripts" not in mmultiscript_tags: 
        subscripts = []
        superscripts = []
        # get subscript-superscript pairs 
        if len(mmultiscript_texts[1:]) % 2 == 0: 
            for postscript in list(batched(mmultiscript_texts[1:], 2)): 
                subscripts.append(postscript[0])
                superscripts.append(postscript[1])
            subscripts = ["" if i is None else i for i in subscripts]
            superscripts = ["" if i is None else i for i in superscripts]
            subscripts_txt = "".join(subscripts)
            superscripts_txt = "".join(superscripts)
            result = base + convert_subscripts(subscripts_txt) + convert_superscripts(superscripts_txt)            
    return result  

def normalise_mathml(text): 
    def replace_msup(match):
        mi_content = match.group(1)
        mn_content = match.group(2)
        return mi_content + convert_superscripts(mn_content)
    def replace_msub(match):
        mi_content = match.group(1)
        mn_content = match.group(2)
        return mi_content + convert_subscripts(mn_content)
    text = re.sub("<(/)?(math|mrow|mo)>", "", text)
    if "<msup>" in text:
        text = re.sub(r"<msup><mi>(.*)</mi><mn>(.*)</mn></msup>", replace_msup, text)
    if "<msub>"in text: 
        text = re.sub(r"<msub><mi>(.*)</mi><mn>(.*)</mn></msub>", replace_msub, text)
    if "<mmultiscripts>" in text: 
        text = re.sub(r"<mmultiscripts>([\s\S]+?)</mmultiscripts>", normalise_mmultiscripts, text) 
    text = re.sub(r"<(/)?(math|mi|mn|msub|msup|mtext|mmultiscripts|none|mrow)(/)?>", "", text)
    return text 

def normalise(text:str): 
    text = text.strip() 
    if "<math>" in text: 
        text = normalise_mathml(text)
    if "<sub>" in text:
        text = convert_subscripts(text)
    if "<sup>" in text: 
        text = convert_superscripts(text)  
    # collapse simple HTML and multiple spaces 
    text = re.sub(r"<(/)?(b|i|br|div .*)>", "", text) 
    text = re.sub(r"(\s)+", " ", text) 
    return text 