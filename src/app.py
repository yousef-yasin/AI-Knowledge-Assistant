"""
Legacy prototype retained for reference.

Run the complete integrated application with:
    python -m src.main
"""

from memory import ConversationMemory
from decision_agent import needs_retrieval
from generator import generate_answer
from validator import validate_answer


# ----------------------------
# Initialize conversation memory
# ----------------------------

memory = ConversationMemory()


# ----------------------------
# Temporary retrieved context
# This will later be replaced by
# the Retriever module (Member 2)
# ----------------------------

knowledge_context = """
Source: Employee Handbook.pdf
Page: 15

Employees receive 21 annual leave days.
Emergency leave requires manager approval.
Working hours are from 9:00 AM to 5:00 PM.
"""


def ask_question(question: str):

    print("\nUser:")
    print(question)

    # Get conversation history
    history = memory.get_history()

    # ----------------------------
    # Retrieval Decision Agent
    # ----------------------------

    retrieve = needs_retrieval(
        question,
        history
    )

    if retrieve:

        print("\n[Retrieval Decision]: Searching documents...")

        # TODO:
        # Replace this with the real Retriever
        # context = retrieve_context(question)

        context = knowledge_context

    else:

        print("\n[Retrieval Decision]: Using conversation memory...")

        # Temporary context
        context = knowledge_context

    # ----------------------------
    # Generate Answer
    # ----------------------------

    answer = generate_answer(
        question=question,
        context=context,
        history=history,
        strategy="advanced"
    )

    # ----------------------------
    # Handle Gemini Errors
    # ----------------------------

    if isinstance(answer, str) and answer.startswith("Error"):
        print("\nAssistant:")
        print(answer)
        return

    # ----------------------------
    # Validate Answer
    # ----------------------------

    validation = validate_answer(
        question=question,
        context=context,
        answer=answer
    )

    print("\nValidation:")
    print(validation)

    # ----------------------------
    # Display Result
    # ----------------------------

    if validation["status"] == "PASS":

        print("\nAssistant:")
        print(answer)

        # Save conversation
        memory.add_user_message(question)
        memory.add_assistant_message(answer)

    else:

        print("\nAssistant:")
        print("The answer was rejected because it could not be verified.")
        print("Reason:", validation["reason"])


# ----------------------------
# Program Entry Point
# ----------------------------

if __name__ == "__main__":

    print("==============================")
    print(" AI Knowledge Assistant ")
    print("==============================")

    print("\nType 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in [
            "exit",
            "quit",
            "bye"
        ]:
            print("Goodbye!")
            break

        ask_question(question)