from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    applied_rules: list[str] = field(default_factory=list)


class QueryRewriter:
    FILLERS = (
        r"^can you (please )?", r"^could you (please )?", r"^please tell me ",
        r"^tell me ", r"^i want to know ", r"^do you know ",
    )
    ABBREVIATIONS = {
        "faq": "frequently asked questions", "docs": "documentation",
        "info": "information", "hr": "human resources",
    }

    def rewrite(self, query: str) -> RewrittenQuery:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Cannot rewrite an empty query.")
        rules: list[str] = []
        result = cleaned
        for pattern in self.FILLERS:
            updated = re.sub(pattern, "", result, flags=re.IGNORECASE)
            if updated != result:
                rules.append("stripped_filler_phrase")
            result = updated
        words = result.split()
        expanded = [self.ABBREVIATIONS.get(word.lower(), word) for word in words]
        if expanded != words:
            rules.append("expanded_abbreviations")
        rewritten = " ".join(expanded)
        rewritten = rewritten[:1].upper() + rewritten[1:]
        return RewrittenQuery(cleaned, rewritten, list(dict.fromkeys(rules)))

    def generate_multi_queries(self, query: str, max_queries: int = 3) -> list[str]:
        """Generate complementary retrieval queries without extra API calls."""
        base = self.rewrite(query).rewritten
        variants = [base]
        lower = base.lower()
        replacements = {
            "how many": "number of",
            "employees receive": "employee entitlement",
            "vacation": "annual leave",
            "working hours": "work schedule office hours",
            "policy": "rules requirements procedure",
        }
        for source, target in replacements.items():
            if source in lower:
                variants.append(re.sub(source, target, base, flags=re.IGNORECASE))
        keywords = " ".join(
            word for word in re.findall(r"\w+", lower)
            if word not in {"what", "how", "many", "do", "does", "the", "a", "an", "is", "are"}
        )
        if keywords:
            variants.append(keywords)
        return list(dict.fromkeys(variants))[:max_queries]
