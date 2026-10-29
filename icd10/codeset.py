"""
The ICD-10-CM code set (April 2026 release, via the `simple-icd-10-cm` package, which bundles
the official CMS/CDC tabular XML).

Only billable (leaf) codes are ever suggested. Each code carries the text used for search
(its description plus its parent descriptions for context) and the coding instructions a
human coder checks: inclusion terms, Excludes1, "code first" and "use additional code".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import simple_icd_10_cm as cm

RELEASE = "ICD-10-CM April 1, 2026"


@dataclass
class Code:
    code: str
    description: str
    parents: list[str]                       # descriptions from category down to parent
    chapter: str
    inclusion_terms: list[str] = field(default_factory=list)
    excludes1: list[str] = field(default_factory=list)
    code_first: list[str] = field(default_factory=list)
    use_additional: list[str] = field(default_factory=list)

    def search_text(self, with_inclusion_terms: bool = True) -> str:
        parts = [self.description] + self.parents[-2:]
        if with_inclusion_terms:
            parts += self.inclusion_terms
        return " | ".join(parts)


def _inherited(code: str, getter) -> list[str]:
    """Instructional notes apply to a code and everything under the category/subcategory."""
    out = []
    for c in [code] + cm.get_ancestors(code):
        if cm.is_chapter_or_block(c):
            break
        v = getter(c)
        if isinstance(v, str):
            v = [v] if v.strip() else []
        out += [x.strip() for x in v if x and x.strip()]
    return out


@lru_cache(maxsize=1)
def load_codes() -> dict[str, Code]:
    codes = {}
    for c in cm.get_all_codes(with_dots=True):
        if not cm.is_leaf(c) or cm.is_chapter_or_block(c):
            continue
        anc = cm.get_ancestors(c)
        parents = [cm.get_description(a) for a in reversed(anc) if not cm.is_chapter_or_block(a)]
        chapter = next((cm.get_description(a) for a in anc if cm.is_chapter(a)), "")
        codes[c] = Code(
            code=c, description=cm.get_description(c), parents=parents, chapter=chapter,
            inclusion_terms=list(cm.get_inclusion_term(c)),
            excludes1=_inherited(c, cm.get_excludes1),
            code_first=_inherited(c, cm.get_code_first),
            use_additional=_inherited(c, cm.get_use_additional_code),
        )
    return codes
