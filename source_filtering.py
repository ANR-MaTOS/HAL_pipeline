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
    text = re.sub(r"<(/)?(mi|mn|msub|msup|mtext)>", "", text)
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