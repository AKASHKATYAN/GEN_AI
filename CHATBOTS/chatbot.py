from dotenv import load_dotenv
load_dotenv()
import langchain 

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage


model=ChatMistralAI(model="mistral-small-2506",temperature=0.9)
messages=[SystemMessage(content='You are a funny AI assistant.')]
while True:
    print("------------welcome type 0 to exit the application----------")
    
    prompt=input("You:")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    
    print("BOT:", response.content)