"""
Specificity-aware re-ranking.

A coder only assigns a detail (side of body, encounter type, "with complication") when the
documentation supports it. Text search ignores that and happily ranks "Effusion, left hand"
for "peritoneal effusion". This step re-scores the top candidates: every qualifier in a code's
description that the query does not mention costs a penalty, so documented detail wins and
undocumented detail loses.
"""
from __future__ import annotations

import re

from icd10.codeset import Code
from icd10.retriever import words

QUALIFIERS = {
    "sequela": r"\bsequela\b", "subsequent encounter": r"\bsubsequent encounter\b",
    "initial encounter": r"\binitial encounter\b", "right": r"\bright\b", "left": r"\bleft\b",
    "bilateral": r"\bbilateral\b", "upper": r"\bupper\b", "lower": r"\blower\b",
    "trimester": r"\btrimester\b", "pregnancy": r"\bpregnancy|childbirth|puerperium\b",
    "newborn": r"\bnewborn|perinatal\b", "in remission": r"\bin remission\b",
    "with complication": r"\bwith (?!out)\w+", "hand": r"\bhand\b", "foot": r"\bfoot\b",
    "ankle": r"\bankle\b", "knee": r"\bknee\b", "hip": r"\bhip\b", "shoulder": r"\bshoulder\b",
    "eye": r"\beye\b", "ear": r"\bear\b",
}
_PATTERNS = {k: re.compile(v, re.I) for k, v in QUALIFIERS.items()}


def undocumented_qualifiers(code: Code, query: str) -> list[str]:
    q = query.lower()
    q_words = set(words(query, keep_stop=True))
    missing = []
    q_without = re.search(r"\bwithout\b", q) is not None
    q_with = re.search(r"\bwith\b", q) is not None
    for name, pat in _PATTERNS.items():
        if pat.search(code.description) and not pat.search(q):
            if name == "with complication" and q_with and not q_without:
                continue
            missing.append(name)
    # "without X" in the code but the text documents "with X" (and never says "without")
    if re.search(r"\bwithout\b", code.description, re.I) and q_with and not q_without:
        missing.append("without (text says with)")
    return missing


def rerank(candidates: list[tuple[Code, float]], query: str, penalty: float = 0.25) -> list[tuple[Code, float, list[str]]]:
    if not candidates:
        return []
    top = candidates[0][1] or 1.0
    out = []
    for code, score in candidates:
        miss = undocumented_qualifiers(code, query)
        out.append((code, score / top - penalty * len(miss), miss))
    return sorted(out, key=lambda x: -x[1])
