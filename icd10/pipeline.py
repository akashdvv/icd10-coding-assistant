"""
The coding pipeline: free text -> ranked ICD-10-CM suggestions with coding guidance.

    1. normalise    expand clinical abbreviations (HTN, T2DM, CHF ...)
    2. rewrite      (optional, LLM) translate clinician wording / eponyms into ICD terminology
    3. retrieve     hybrid word + character search over 74,714 billable codes; when a rewrite
                    exists, results for the original and rewritten query are fused
    4. rerank       penalise codes asserting detail the text doesn't document
    5. select       (optional, LLM) choose the best code from the shortlist
    6. guidance     attach Excludes1 conflicts, "code first" and "use additional code" notes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from icd10.codeset import Code, load_codes
from icd10.normalize import normalise
from icd10.rerank import rerank
from icd10.retriever import HybridRetriever, top_k

CODE_REF = re.compile(r"\b([A-TV-Z][0-9][0-9AB])(?:\.([0-9A-Z]{1,4}))?(-?)")


@dataclass
class Suggestion:
    code: str
    description: str
    score: float
    undocumented_detail: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)


def referenced_codes(note_text: str) -> list[str]:
    """Code references inside an instructional note, e.g. 'E11.22' or 'I13.-' (prefix)."""
    return [m.group(1) + ("." + m.group(2) if m.group(2) else "") for m in CODE_REF.finditer(note_text)]


def excludes1_conflicts(codes: list[Code]) -> list[str]:
    """Excludes1 means 'never code together'. Flag any pair in the selection that violates it."""
    msgs, seen = [], set()
    for a in codes:
        for note in a.excludes1:
            for ref in referenced_codes(note):
                for b in codes:
                    pair = frozenset((a.code, b.code))
                    if b.code != a.code and b.code.startswith(ref) and pair not in seen:
                        seen.add(pair)
                        msgs.append(f"Excludes1: {a.code} and {b.code} should not be coded together ({note.splitlines()[0]})")
    return msgs


class Coder:
    def __init__(self, llm=None, k: int = 10, depth: int = 50, with_inclusion_terms: bool = True):
        """with_inclusion_terms=False is for evaluation only: the benchmark's queries ARE inclusion
        terms, so they must not be in the index being tested."""
        self.codes = list(load_codes().values())
        self.index = {c.code: c for c in self.codes}
        self.retriever = HybridRetriever(self.codes, with_inclusion_terms=with_inclusion_terms)
        self.llm, self.k, self.depth = llm, k, depth

    def candidates(self, text: str, rewrite: str | None = None, depth: int | None = None) -> list[tuple[Code, float, list[str]]]:
        depth = depth or self.depth
        queries = [normalise(text)] + ([rewrite] if rewrite else [])
        fused = np.zeros(len(self.codes))
        for q in queries:
            s = self.retriever.scores(q)
            for rank, i in enumerate(top_k(s, depth), 1):
                fused[i] += 1.0 / (60 + rank)
        cands = [(self.codes[i], float(fused[i])) for i in top_k(fused, depth)]
        return rerank(cands, " ".join(queries))

    @staticmethod
    def _note(n: str, limit: int = 160) -> str:
        t = " ".join(x.strip() for x in n.splitlines() if x.strip()).replace(": ", ": ", 1)
        return t if len(t) <= limit else t[: limit - 1] + "…"

    def guidance(self, code: Code) -> list[str]:
        g = [f"Code first: {self._note(n)}" for n in code.code_first[:2]]
        g += [f"Use additional code: {self._note(n)}" for n in code.use_additional[:2]]
        return g

    def suggest(self, text: str) -> list[Suggestion]:
        rewrite = self.llm.rewrite(text) if self.llm else None
        ranked = self.candidates(text, rewrite)[: max(self.k, 20)]
        if self.llm:
            choice = self.llm.select(text, [(c.code, c.description) for c, _, _ in ranked])
            ranked.sort(key=lambda x: x[0].code != choice)
        return [Suggestion(c.code, c.description, round(s, 3), miss, self.guidance(c)) for c, s, miss in ranked[: self.k]]

    def code_note(self, note: str) -> dict:
        """Code a short assessment / problem list: one top suggestion per line, then check
        Excludes1 conflicts across the whole set."""
        problems = [p.strip(" -•\t") for p in re.split(r"[\n;]+", note) if len(p.strip(" -•\t")) > 2]
        picks = [(p, self.suggest(p)) for p in problems]
        chosen = [self.index[s[0].code] for _, s in picks if s]
        return {"problems": [{"text": p, "suggestions": s} for p, s in picks], "conflicts": excludes1_conflicts(chosen)}
