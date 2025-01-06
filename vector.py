import os 
import psycopg2 
import random 
import json 
from langchain_ollama import OllamaEmbeddings 
from langchain_core.documents import Document 
from langchain_postgres.vectorstores import PGVector
from langchain_ollama.llms import OllamaLLM
from format import resume_format


file_path = os.path.join(os.getcwd(),'data/resume_text')
file_paths = os.listdir(file_path)

llm = OllamaLLM(model='llama3.2')
resume_path = os.path.join(file_path, file_paths[0])
with open(resume_path, 'r') as f:
    resume = f.read()

prompt = f"""
            Please provide the output strictly in JSON format without any additional comments or explanations.
            From the given resume, extract the information in JSON format.
            If there are no field values, leave it empty.
            The given resume is {resume}.
            The response format is {resume_format}.
            """
response = llm.invoke(prompt)
js = response.replace('\n','')
print(json.loads(js))
