"""
Score the coding assistant.

Part A - retrieval only, all 1,500 official-synonym queries (descriptions-only index).
Part B - full pipeline on 100 held-out queries (50 hard, 50 easy):
         retrieval only  ->  + LLM rewrite  ->  + LLM rewrite + LLM selection
         The LLM outputs were committed to git before this script was run.

    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from icd10.pipeline import Coder  # noqa: E402
from icd10.rerank import rerank  # noqa: E402
from icd10.retriever import CharRetriever, WordRetriever, search  # noqa: E402

EVAL = ROOT / "eval"
RESULTS, IMAGES = ROOT / "results", ROOT / "images"
BLUE, ORANGE, AQUA, YELLOW, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(ranked: list[list[str]], gold: list[str]) -> dict:
    n = len(gold)
    return {"top1": sum(r[:1] == [g] for r, g in zip(ranked, gold)) / n,
            "top5": sum(g in r[:5] for r, g in zip(ranked, gold)) / n,
            "top10": sum(g in r[:10] for r, g in zip(ranked, gold)) / n,
            "category_top1": sum(bool(r) and r[0][:3] == g[:3] for r, g in zip(ranked, gold)) / n}


def part_a(coder: Coder, qs: list[dict]) -> pd.DataFrame:
    word, char = WordRetriever(coder.codes, False), CharRetriever(coder.codes, False)
    systems = {
        "BM25 (words)": lambda q: [c.code for c, _ in search(word, q, 10)],
        "TF-IDF (char n-grams)": lambda q: [c.code for c, _ in search(char, q, 10)],
        "Hybrid": lambda q: [c.code for c, _ in search(coder.retriever, q, 10)],
        "Hybrid + specificity rerank": lambda q: [c.code for c, _, _ in rerank(search(coder.retriever, q, 50), q)][:10],
    }
    rows = []
    for name, fn in systems.items():
        ranked = [fn(q["query"]) for q in qs]
        for diff in ["all", "easy", "hard"]:
            idx = [i for i, q in enumerate(qs) if diff == "all" or q["difficulty"] == diff]
            m = metrics([ranked[i] for i in idx], [qs[i]["gold"] for i in idx])
            rows.append({"system": name, "subset": diff, "n": len(idx), **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def part_b(coder: Coder) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = {r["qid"]: r for r in jl(EVAL / "llm_sample_gold.jsonl")}
    # step-1 rewrites are already reflected in the frozen shortlists below
    picks = {r["qid"]: r["selected"] for r in jl(EVAL / "llm_answers" / "step2_selections.jsonl")}
    # the exact shortlists the LLM saw (frozen in git together with its answers)
    shortlists = {r["qid"]: [c[0] for c in r["candidates"]] for r in jl(EVAL / "llm_sample_candidates.jsonl")}
    per_q = []
    for qid, g in gold.items():
        base = [c.code for c, _, _ in rerank(search(coder.retriever, g["query"], 50), g["query"])]
        with_rw = shortlists[qid]
        per_q.append({"qid": qid, "query": g["query"], "gold": g["gold"], "difficulty": g["difficulty"],
                      "retrieval_top1": base[0], "gold_in_retrieval_top15": g["gold"] in base[:15],
                      "rewrite_top1": with_rw[0], "gold_in_shortlist": g["gold"] in with_rw[:15],
                      "llm_selected": picks[qid]})
    d = pd.DataFrame(per_q)
    rows = []
    for diff in ["all", "easy", "hard"]:
        s = d if diff == "all" else d[d.difficulty == diff]
        auto = s[s.llm_selected != "NONE"]
        rows.append({
            "subset": diff, "n": len(s),
            "retrieval_only_top1": (s.retrieval_top1 == s.gold).mean(),
            "retrieval_only_recall15": s.gold_in_retrieval_top15.mean(),
            "with_rewrite_top1": (s.rewrite_top1 == s.gold).mean(),
            "with_rewrite_recall15": s.gold_in_shortlist.mean(),
            "full_pipeline_accuracy": (s.llm_selected == s.gold).mean(),
            "auto_coded_share": len(auto) / len(s),
            "precision_when_auto_coded": (auto.llm_selected == auto.gold).mean() if len(auto) else float("nan"),
            "category_accuracy_when_auto_coded": (auto.llm_selected.str[:3] == auto.gold.str[:3]).mean() if len(auto) else float("nan"),
        })
    return pd.DataFrame(rows).round(3), d


def charts(a: pd.DataFrame, b: pd.DataFrame) -> None:
    IMAGES.mkdir(exist_ok=True)
    # Part A: top-1 / top-10 by system (all queries)
    s = a[a.subset == "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = range(len(s))[::-1]
    ax.barh([v + 0.2 for v in y], s.top10 * 100, height=0.38, color=AQUA, label="Correct code in top 10")
    ax.barh([v - 0.2 for v in y], s.top1 * 100, height=0.38, color=BLUE, label="Correct code ranked #1")
    for v, (t1, t10) in zip(y, zip(s.top1, s.top10)):
        ax.text(t10 * 100 + 0.8, v + 0.2, f"{t10:.0%}", va="center", fontsize=9, color=INK_2)
        ax.text(t1 * 100 + 0.8, v - 0.2, f"{t1:.0%}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(list(y), s.system)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 1,500 official synonym queries")
    ax.set_title("Search alone: finds the neighbourhood, rarely the exact code")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "retrieval_benchmark.png", dpi=150)
    plt.close(fig)

    # Part B: pipeline stages, easy vs hard
    stages = [("retrieval_only_top1", "Search only"), ("with_rewrite_top1", "+ LLM rewrite"),
              ("full_pipeline_accuracy", "+ LLM rewrite\n+ LLM selection")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.36
    for j, (subset, color) in enumerate([("easy", AQUA), ("hard", ORANGE)]):
        row = b[b.subset == subset].iloc[0]
        xs = [i + (j - 0.5) * width for i in range(len(stages))]
        vals = [row[k] * 100 for k, _ in stages]
        ax.bar(xs, vals, width=width - 0.03, color=color, label=f"{subset.capitalize()} synonyms (n={int(row.n)})")
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(stages)), [lbl for _, lbl in stages])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact code correct (%)")
    ax.set_title("What each pipeline stage adds (100 held-out queries)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "pipeline_stages.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    coder = Coder(with_inclusion_terms=False)
    qs = jl(EVAL / "synonym_benchmark.jsonl")
    a = part_a(coder, qs)
    a.to_csv(RESULTS / "retrieval_benchmark.csv", index=False)
    print("Part A - retrieval only (1,500 official synonym queries)\n", a.to_string(index=False))
    b, per_q = part_b(coder)
    b.to_csv(RESULTS / "pipeline_benchmark.csv", index=False)
    per_q.to_csv(RESULTS / "pipeline_per_query.csv", index=False)
    print("\nPart B - full pipeline (100 held-out queries)\n", b.T.to_string())
    charts(a, b)


if __name__ == "__main__":
    main()
