from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional


load_dotenv()
from langchain_mistralai import ChatMistralAI

class Movie(BaseModel):
    title:Optional[str]
    release_year:Optional[int]
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:str
    
parser=PydanticOutputParser(pydantic_object=Movie)   

model=ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    (
        'system',"""
        Extract movie information from the paragraph{format_instructions}"""),
        ("human","{paragraph}")]
    )
    
para=input("Give your paragraph:")

final_prompt=prompt.invoke({"paragraph":para,
                            'format_instructions':parser.get_format_instructions()})


response=model.invoke(final_prompt)
print(response.content)