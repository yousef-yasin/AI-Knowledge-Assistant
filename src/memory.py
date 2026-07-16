from typing import List, Dict


class ConversationMemory:
    """
    Stores conversation history during the current session.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append({
            "role": "user",
            "content": message
        })

    def add_assistant_message(self, message: str):
        self.history.append({
            "role": "assistant",
            "content": message
        })

    def get_history(self) -> List[Dict[str, str]]:
        return self.history[-5:]
       # return self.history

    def clear(self):
        self.history.clear()