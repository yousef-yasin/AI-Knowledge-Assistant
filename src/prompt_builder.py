from typing import List


def format_history(history: List[dict]) -> str:
   
   # Converts conversation history into a readable string.
    

    if not history:
        return "No previous conversation."

    formatted = ""

    for message in history:
        role = message["role"].capitalize()
        content = message["content"]
        formatted += f"{role}: {content}\n"

    return formatted


def build_prompt(question: str, context: str, history: List[dict]) -> str:
    #هي دالة تبني الـ Prompt النهائي الذي سيتم إرساله إلى Gemini
    """
    Builds the final prompt for Gemini.
    """

    formatted_history = format_history(history)

    prompt = f"""

You are an AI Knowledge Assistant.

Answer the user's question using ONLY the retrieved context below.
Do not use your own knowledge.

Rules:

- Never invent information.
- Never use outside knowledge.
- If the answer does not exist in the context, reply:
  "The requested information is not available in the provided documents."

- Be concise.
- Use bullet points when appropriate.
- Include document citations at the end.

----------------------------
Conversation History
----------------------------

{formatted_history}
#هو سجل المحادثة السابقة بين المستخدم والمساعد.

----------------------------
Retrieved Context
----------------------------

{context}
#هو النص الذي استرجعه نظام الـ RAG من الملفات (Knowledge Base).

----------------------------
User Question
----------------------------

{question}
#هو سؤال المستخدم الحالي

----------------------------
Answer Format
----------------------------

Answer:
Sources:
Confidence:
"""

    return prompt

'''
هذا فقط لعمل testing
history = [
    {
        "role": "user",
        "content": "Tell me about annual leave."
    },
    {
        "role": "assistant",
        "content": "Employees receive 21 annual leave days."
    }
]

context = """
Employee Handbook.pdf
Page 15

Employees receive 21 annual leave days.
Emergency leave requires manager approval.
"""

question = "What about emergency leave?"

prompt = build_prompt(question, context, history)

print(prompt)
'''