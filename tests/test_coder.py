import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icd10.codeset import load_codes  # noqa: E402
from icd10.llm import ClaudeCodingLLM  # noqa: E402
from icd10.normalize import normalise  # noqa: E402
from icd10.pipeline import Coder, excludes1_conflicts, referenced_codes  # noqa: E402
from icd10.rerank import undocumented_qualifiers  # noqa: E402


@pytest.fixture(scope="module")
def coder():
    return Coder(k=5)


def test_codeset_is_billable_only_and_parses_instructions():
    codes = load_codes()
    assert len(codes) > 70_000
    assert "E11" not in codes and "E11.22" in codes           # categories are not billable
    assert any("N18.1-N18.6" in n for n in codes["E11.22"].use_additional)
    assert all(isinstance(n, str) and len(n) > 3 for n in codes["E11.22"].use_additional)


def test_abbreviations_are_expanded():
    t = normalise("T2DM w/ CKD stage IIIa, HTN")
    assert "type 2 diabetes mellitus" in t and "hypertension" in t and "chronic kidney disease" in t


def test_specificity_penalises_undocumented_detail():
    codes = load_codes()
    assert "left" in undocumented_qualifiers(codes["M25.562"], "knee pain")
    assert undocumented_qualifiers(codes["M25.562"], "left knee pain") == []
    assert "with complication" in undocumented_qualifiers(codes["E10.69"], "type 1 diabetes without complications")


@pytest.mark.parametrize("text,expected", [
    ("type 1 diabetes mellitus without complications", "E10.9"),
    ("acute on chronic systolic CHF", "I50.23"),
    ("COPD with acute exacerbation", "J44.1"),
    ("pain in left knee", "M25.562"),
    ("essential hypertension", "I10"),
])
def test_common_problems_rank_first(coder, text, expected):
    assert coder.suggest(text)[0].code == expected


def test_excludes1_conflict_is_flagged_once():
    codes = load_codes()
    msgs = excludes1_conflicts([codes["E11.22"], codes["E10.9"]])
    assert len(msgs) == 1 and "Excludes1" in msgs[0]


def test_referenced_codes_parses_ranges_and_prefixes():
    assert referenced_codes("type 1 diabetes mellitus (E10.-)") == ["E10"]
    assert "I13" in referenced_codes("hypertensive chronic kidney disease (I12.-, I13.-)")


def test_llm_select_cannot_return_a_code_outside_the_shortlist():
    class Fake:
        def __init__(self, text):
            self.text = text

        def create(self, **kw):
            assert kw["temperature"] == 0
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])

    shortlist = [("I45.6", "Pre-excitation syndrome"), ("I45.9", "Conduction disorder, unspecified")]
    ok = ClaudeCodingLLM(model="t", client=SimpleNamespace(messages=Fake('{"code": "I45.6", "reason": "WPW"}')))
    assert ok.select("WPW", shortlist) == "I45.6"
    invented = ClaudeCodingLLM(model="t", client=SimpleNamespace(messages=Fake('{"code": "I49.9", "reason": "x"}')))
    assert invented.select("WPW", shortlist) == "NONE"
    garbage = ClaudeCodingLLM(model="t", client=SimpleNamespace(messages=Fake("not json")))
    assert garbage.select("WPW", shortlist) == "NONE"


def test_frozen_llm_answers_cover_the_sample_and_stay_in_shortlists():
    cands = {json.loads(l)["qid"]: [c[0] for c in json.loads(l)["candidates"]]
             for l in (ROOT / "eval/llm_sample_candidates.jsonl").read_text().splitlines()}
    picks = [json.loads(l) for l in (ROOT / "eval/llm_answers/step2_selections.jsonl").read_text().splitlines()]
    assert len(picks) == 100
    assert all(p["selected"] == "NONE" or p["selected"] in cands[p["qid"]] for p in picks)
