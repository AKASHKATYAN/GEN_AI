from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct",
                        task='text-generation')

model=ChatHuggingFace(llm=llm)

result=model.invoke("how can i kill my professor?")
print(result.content)