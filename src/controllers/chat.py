from src.core import model
from src.db import vector_db
from src.services import ChatService


class ChatController:
    def __init__(self):
        self.chat_service = ChatService(
            vector_db,
            model
        )

    def load_pdf(self, file_path: str) -> bool:
        try:
            self.chat_service.add_pdf(file_path)
            return True
        except Exception as e:
            print(f"Erro ao carregar PDF: {e}")
            return False

    def list_pdf(self) -> list[str]:
        return self.chat_service.list_pdf()

    def chat(self,  message: str) -> str:
        return self.chat_service.send_message(message)

    def clear_session(self) -> None:
        self.chat_service.clear_session()
