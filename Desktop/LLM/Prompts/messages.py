from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv("../.env")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
messages=[
    SystemMessage(content='you are a helpful assistent'),
    HumanMessage(content='tell me about langchain')
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)