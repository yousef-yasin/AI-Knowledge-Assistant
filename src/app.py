
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
# Replace this with retriever output
# from Member 2
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

    # Get previous conversation
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

        # Here will be replaced by:
        #
        # context = retrieve_context(question)
        #
        # from retriever.py

        context = knowledge_context


    else:

        print("\n[Retrieval Decision]: Using conversation memory...")

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

        memory.add_user_message(question)

        memory.add_assistant_message(answer)


    else:

        print("\nAssistant:")
        print(
            "The answer was rejected because it could not be verified."
        )

        print(
            "Reason:",
            validation["reason"]
        )



# ----------------------------
# Program Entry Point
# ----------------------------

if __name__ == "__main__":


    print("==============================")
    print(" AI Knowledge Assistant ")
    print("==============================")


    while True:

        question = input("\nYou: ")

        if question.lower() in [
            "exit",
            "quit",
            "bye"
        ]:
            print("Goodbye!")
            break


        ask_question(question)


