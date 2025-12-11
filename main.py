from langchain.agents import create_agent
from langchain_ollama.chat_models import ChatOllama


chat = ChatOllama(
    name="gpt-oss:20b",

)

agent = create_agent(

)
