"""
Suggest ICD-10-CM codes for a short note or problem list (one problem per line or separated by ';').

    python code_note.py "T2DM with diabetic CKD stage 3a; HTN; acute on chronic systolic CHF"
    python code_note.py --llm "Wolff-Parkinson-White syndrome"      # needs ANTHROPIC_API_KEY
"""
import argparse

from icd10.pipeline import Coder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("note")
    ap.add_argument("--llm", action="store_true", help="use Claude to rewrite and select (needs ANTHROPIC_API_KEY)")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    llm = None
    if args.llm:
        from icd10.llm import ClaudeCodingLLM
        llm = ClaudeCodingLLM()
    coder = Coder(llm=llm, k=args.k)
    result = coder.code_note(args.note)
    for p in result["problems"]:
        print(f"\n▸ {p['text']}")
        for i, s in enumerate(p["suggestions"], 1):
            flag = f"   (undocumented detail: {', '.join(s.undocumented_detail)})" if s.undocumented_detail else ""
            print(f"   {i}. {s.code:<9} {s.description}{flag}")
        for g in p["suggestions"][0].guidance if p["suggestions"] else []:
            print(f"      ⚑ {g}")
    for c in result["conflicts"]:
        print(f"\n⚠ {c}")


if __name__ == "__main__":
    main()
