from typing import List
#استوردنا List حتى نكتب نوع المتغيرات.


def format_history(history: List[dict]) -> str:
    #تحويل الـ History إلى نص مرتب.
    """
    Convert conversation history into a readable string.
    """

    if not history:
        return "No previous conversation."
    # اذا ما كان عندي محادثة من قبل
    formatted_messages = []

    for message in history:
        role = message.get(
            "role",
            "Unknown",
        ).capitalize()

        content = message.get(
            "content",
            "",
        )
        '''
        الهدف الرئيسي نحول المحادثة ل history
        الهدف انه تمر على كل رسالة بال history 
        تحدد اذا السوال من المستخدم او المساعد
        وتطلع على محتوى الرسالة
        وتحولها من قائمة الى نص لانه جيمني ما بفهم الا نص
        واذا ما لقى المطلوب بس برجع غير معروف 
        وهيك البرنامج بضل شغال وما بتوقف عن العمل
        اذا هو رح يوخذ السوال والمحتوى

        '''

        formatted_messages.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted_messages)
#join() تجمع عناصر القائمة في String واحد.

#هذه تبني Basic Prompt.
def build_prompt(
    question: str,
    context: str,
    history: List[dict],
) -> str:
    """
    Build a basic grounded prompt.
    """

    formatted_history = format_history(history)
# لحتى يضل نص وهون بلشنا نبني ال prompt ونحط القواعد
    prompt = f"""
You are an AI Knowledge Assistant.

Your task is to answer the user's question using ONLY
the retrieved context.

Rules:

- Use only information available in the retrieved context.
- Never use outside knowledge.
- Never invent information.
- If the answer is not available in the context, reply exactly:
"The requested information is not available in the retrieved documents."
- Keep the answer concise and professional.
- Do not include sources.
- Do not include citations.
- Do not include confidence scores.
- Return only the final answer text.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt.strip()
#تحذف الفراغات والأسطر الفارغة من البداية والنهاية.


# المتطور هو فقط لحتى يعطي اجابقة اقوى ونفس عمل العادي
def build_advanced_prompt(
    question: str,
    context: str,
    history: List[dict],
) -> str:
    """
    Build an advanced prompt for higher accuracy.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an advanced AI Knowledge Assistant.

Generate a precise and professional answer based ONLY
on the retrieved context.

Instructions:

- Answer ONLY using the retrieved context.
- Every factual statement must be supported by the context.
- Never hallucinate.
- Never use external knowledge.
- Use conversation history only to understand references
  such as "it", "that", or follow-up questions.
- Do not allow conversation history to override retrieved context.
- If the information is missing, reply exactly:
"The requested information is not available in the retrieved documents."
- Keep the answer concise.
- Use bullet points if they improve readability.
- Return ONLY the answer text.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt.strip()



"""
فنحن لا نرتب الإجابة فقط، وإنما ننظم المعلومات التي نعطيها لـ Gemini.
يعني كأننا نقول له:
هذه هي المحادثة السابقة:
...
وهذه هي المعلومات التي يجب أن تعتمد عليها:
...
وهذا هو سؤال المستخدم:
...
الآن ابدأ بالإجابة هنا:
Answer:
"""