"""
LLM stages of the pipeline (Anthropic Claude API).

rewrite(text)            -> ICD-10-CM terminology for clinician wording, eponyms and lay terms.
                            The model never outputs a code here, only words to search with, so it
                            cannot invent a code that doesn't exist.
select(text, shortlist)  -> one code from the retrieved shortlist, or "NONE" to route the case to a
                            human coder. The answer is validated against the shortlist.
"""
from __future__ import annotations

import json
import os

REWRITE_SYSTEM = (
    "You translate clinical wording into the terminology used by ICD-10-CM code titles. Expand "
    "abbreviations, replace eponyms and lay terms with the official condition name, and include "
    "category-level wording a coder would look up. Never output codes. Reply with one line of search "
    "terms separated by semicolons."
)
SELECT_SYSTEM = (
    "You are a certified medical coder. Choose the single ICD-10-CM code from the shortlist that "
    "matches the documentation. Only use codes from the shortlist. Do not assign detail (laterality, "
    "severity, encounter type) that is not documented. If no listed code fits, answer NONE so a human "
    'coder reviews it. Reply as JSON: {"code": "<code or NONE>", "reason": "<one sentence>"}'
)


class ClaudeCodingLLM:
    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("ICD_LLM_MODEL", "claude-haiku-5-5")
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.last_reason = ""

    def _ask(self, system: str, user: str, max_tokens: int = 200) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0,
                                          system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def rewrite(self, text: str) -> str:
        return self._ask(REWRITE_SYSTEM, f"Clinical text: {text}")

    def select(self, text: str, shortlist: list[tuple[str, str]]) -> str:
        listing = "\n".join(f"{code}: {desc}" for code, desc in shortlist)
        raw = self._ask(SELECT_SYSTEM, f"Documentation: {text}\n\nShortlist:\n{listing}")
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            return "NONE"
        self.last_reason = out.get("reason", "")
        code = str(out.get("code", "NONE")).strip().upper()
        return code if code in {c for c, _ in shortlist} else "NONE"   # never trust a code outside the list
