from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv("../.env")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
chat_history=[]
print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        print("Goodbye!")
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:', result.content, '\n')

print(chat_history)