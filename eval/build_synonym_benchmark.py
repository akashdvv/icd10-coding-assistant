"""
Build the official-synonym benchmark.

ICD-10-CM lists "inclusion terms" under many codes - alternative names a clinician might write,
e.g. "Stein-Leventhal syndrome" under E28.2 Polycystic ovarian syndrome. We use each inclusion
term as a query and the code it belongs to as the answer. The search index used for this test
contains descriptions only (no inclusion terms), so the system must generalise to wording it has
never seen. Gold answers come from the official code set, not from the author.

Terms that are near-copies of their code's description (e.g. "Acute bronchitis due to RSV") are
kept but tagged, so results can be reported separately for the hard, genuinely different synonyms.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icd10.codeset import load_codes  # noqa: E402
from icd10.retriever import words  # noqa: E402


def overlap(a: str, b: str) -> float:
    A, B = set(words(a)), set(words(b))
    return len(A & B) / max(len(A), 1)


def main(n: int = 1500, seed: int = 7) -> None:
    rows = []
    for c in load_codes().values():
        for term in c.inclusion_terms:
            if len(term) < 4 or term.lower() == c.description.lower():
                continue
            ov = overlap(term, c.description + " " + " ".join(c.parents[-2:]))
            rows.append({"query": term, "gold": c.code, "description": c.description,
                         "difficulty": "hard" if ov < 0.5 else "easy"})
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out = ROOT / "eval" / "synonym_benchmark.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    hard = sum(r["difficulty"] == "hard" for r in rows)
    print(f"{len(rows)} queries ({hard} hard, {len(rows) - hard} easy) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
