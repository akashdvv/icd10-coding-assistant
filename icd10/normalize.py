"""Clinical text normalisation: abbreviations and shorthand that notes use but the code set doesn't."""
from __future__ import annotations

import re

ABBREVIATIONS = {
    "htn": "hypertension", "dm": "diabetes mellitus", "dm2": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus", "t1dm": "type 1 diabetes mellitus", "dm1": "type 1 diabetes mellitus",
    "chf": "heart failure", "hf": "heart failure", "hfref": "systolic heart failure", "hfpef": "diastolic heart failure",
    "cad": "atherosclerotic heart disease coronary artery", "mi": "myocardial infarction",
    "stemi": "st elevation myocardial infarction", "nstemi": "non st elevation myocardial infarction",
    "afib": "atrial fibrillation", "a-fib": "atrial fibrillation", "af": "atrial fibrillation",
    "copd": "chronic obstructive pulmonary disease", "ckd": "chronic kidney disease", "esrd": "end stage renal disease",
    "aki": "acute kidney failure", "uti": "urinary tract infection", "uri": "acute upper respiratory infection",
    "gerd": "gastro-esophageal reflux disease", "dvt": "deep vein thrombosis embolism", "pe": "pulmonary embolism",
    "cva": "cerebral infarction", "tia": "transient cerebral ischemic attack", "osa": "obstructive sleep apnea",
    "bph": "benign prostatic hyperplasia", "ra": "rheumatoid arthritis", "oa": "osteoarthritis",
    "mdd": "major depressive disorder", "gad": "generalized anxiety disorder", "adhd": "attention-deficit hyperactivity disorder",
    "ptsd": "post-traumatic stress disorder", "hld": "hyperlipidemia", "hlp": "hyperlipidemia",
    "ibs": "irritable bowel syndrome", "sob": "shortness of breath", "cp": "chest pain", "lbp": "low back pain",
    "hx": "personal history", "h/o": "personal history", "fx": "fracture", "sz": "seizure", "ha": "headache",
    "n/v": "nausea vomiting", "abd": "abdominal", "pna": "pneumonia", "cap": "pneumonia", "rsv": "respiratory syncytial virus",
    "hiv": "human immunodeficiency virus", "hcv": "hepatitis c", "hbv": "hepatitis b", "etoh": "alcohol",
    "obs": "observation", "w/": "with", "w/o": "without", "b/l": "bilateral", "bilat": "bilateral",
    "r": "right", "l": "left", "rt": "right", "lt": "left", "lle": "left lower limb", "rle": "right lower limb",
    "lue": "left upper limb", "rue": "right upper limb", "init": "initial encounter", "f/u": "follow-up",
    "s/p": "status post", "yo": "year old", "pt": "patient", "dx": "diagnosis",
}
_SPLIT = re.compile(r"[\s,;]+")


def expand(text: str) -> str:
    out = []
    for tok in _SPLIT.split(text.strip()):
        key = tok.lower().strip(".:()")
        out.append(ABBREVIATIONS.get(key, tok))
    return " ".join(out)


STAGE = re.compile(r"\bstage\s*(\d|[iv]+)([ab])?\b", re.I)
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}


def normalise(text: str) -> str:
    t = expand(text)
    t = STAGE.sub(lambda m: f"stage {ROMAN.get(m.group(1).lower(), m.group(1))}{(m.group(2) or '').lower()}", t)
    return t
