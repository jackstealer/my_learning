from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-models/Muse-Glimmer-30B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_TOKEN")
)

model=ChatHuggingFace(llm=llm)
result=model.invoke("what is lamma animal")

print(result.content)