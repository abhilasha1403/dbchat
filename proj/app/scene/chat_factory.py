from proj.app.scene.base_chat import BaseChat
from proj.core import PromptTemplate
from proj.util.singleton import Singleton
from proj.util.tracer import root_tracer


class ChatFactory(metaclass=Singleton):
    @staticmethod
    def get_implementation(chat_mode, **kwargs):
        # Lazy loading
        from proj.app.scene.chat_db.auto_execute.chat import ChatWithDbAutoExecute
        from proj.app.scene.chat_db.auto_execute.prompt import prompt
        from proj.app.scene.chat_db.professional_qa.chat import ChatWithDbQA
        from proj.app.scene.chat_db.professional_qa.prompt import prompt
        from proj.app.scene.chat_execution.chat import ChatWithPlugin
        from proj.app.scene.chat_execution.prompt import prompt

        chat_classes = BaseChat.__subclasses__()
        implementation = None
        for cls in chat_classes:
            if cls.chat_scene == chat_mode:
                metadata = {"cls": str(cls)}
                with root_tracer.start_span(
                    "get_implementation_of_chat", metadata=metadata
                ):
                    implementation = cls(**kwargs)
        if implementation == None:
            raise Exception(f"Invalid implementation name:{chat_mode}")
        return implementation
