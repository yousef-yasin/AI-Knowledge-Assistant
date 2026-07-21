from __future__ import annotations

import re
#هذه مكتبة Regular Expressions.
#نستخدمها للبحث عن أنماط معينة في السؤال.

class RetrievalDecisionAgent:
    """Decide whether a question needs document retrieval or memory only."""

    DOCUMENT_TERMS = {
        "policy", "document", "handbook", "leave", "vacation", "salary",
        "department", "working hours", "company", "rule", "procedure",
        "benefit", "employee", "annual", "sick", "maternity",
    }
    #هذه قائمة كلمات تدل أن السؤال يحتاج الرجوع للوثائق.

    FOLLOW_UP_PATTERNS = (
        r"^(and|what about|how about)\b",
        r"\b(it|that|those|this|previous answer|you said)\b",
    )
   # هذه أنماط تدل أن المستخدم يكمل سؤالًا سابقًا.


#هل هذا السؤال يحتاج أن أذهب للملفات (Knowledge Base)، 
# أم أستطيع الإجابة من المحادثة السابقة؟"
    def decide(self, question: str, history: list[dict]) -> str:
        #بتحتوي على السوال و المحادثة السابقة 
        cleaned = question.strip().lower()
        #لحتى يفعم انه لو كتبت الكلمة باحرف ضغيرة او كبية هي نفسها
        if not history:
            return "retrieval"
        #اذا مافي محادثة سابقة رح يروح فورا للملفات
        if any(term in cleaned for term in self.DOCUMENT_TERMS):
            return "retrieval"
        #هون اذا عندي كلمات محددة رح يرجع للملفات
        if any(re.search(pattern, cleaned) for pattern in self.FOLLOW_UP_PATTERNS):
            return "memory"
        return "retrieval"
    # هل السؤال عبارة عن سؤال تابع للمحادثة السابقة؟
    #  اذا بنقدر نطول الاجابة من الذاكرة ولا لازم نرجع للملفات


def needs_retrieval(question: str, history: list[dict]) -> bool:
    return RetrievalDecisionAgent().decide(question, history) == "retrieval"
#و needs_retrieval() :فقط يحول القرار إلى True/False
#  حتى يستخدمه main.py.