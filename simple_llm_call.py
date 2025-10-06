from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt = PromptTemplate.from_template("{question}")

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Peru"})
print(result)
