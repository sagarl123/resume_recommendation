import os 
import json 
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from PyPDF2 import PdfReader
from convertPdfToText import extract_text_from_pdf 
from extractResumeJsonFormat import extract_resume_data
from format import job_description_format
from aggregate_data import aggregate_job_description_data
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def job_description_json_format(job_description, format = job_description_format):
    model = OllamaLLM(model = 'llama3.2')
    prompt = f"""
                You are an expert in extracting key information from job description in json format.
                Prepare the job description in JSON format.
                The format is {job_description_format}
                The job description is {job_description}
                You should only provide the response in JSON format.   I
    """
    response = model.invoke(prompt).replace('\n','')
    json_response = json.loads(response)
    aggregate_content = aggregate_job_description_data(json_response)
    json_response['aggregate_content'] = aggregate_content
    return json_response

def query_similar_results(job_description_text, model_name, qdrant_host, qdrant_port, collection_name, top_k=7):
    """
    Retrieves similar results from a Qdrant collection.

    Args:
        query_text (str): The query text to find similar results.
        model_name (str): Name of the embedding model.
        qdrant_host (str): Host address for the Qdrant server.
        qdrant_port (int): Port number for the Qdrant server.
        collection_name (str): Name of the Qdrant collection to query.
        top_k (int): Number of similar results to retrieve.

    Returns:
        list: A list of similar documents with metadata.
    """
    try:
        # Initialize embeddings and Qdrant client
        embeddings = OllamaEmbeddings(model=model_name)
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize the vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        # Get embeddings for the query text
        query_embedding = embeddings.embed_query(job_description_text)
        # Query the vector store
        results = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k
        )

        # Return the results
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
job_description = """Job Title: Python Programmer (2 Years Experience)

Location: [Insert Location]

Job Type: [Full-Time/Part-Time/Contract]

Job Description:

We are seeking a talented and motivated Python Programmer with 2 years of experience to join our growing development team. As a Python Programmer, you will be responsible for writing clean, efficient, and maintainable code, working on a variety of exciting projects, and contributing to the development of new features and improvements. The ideal candidate will have hands-on experience with Python programming, strong problem-solving skills, and the ability to work in a collaborative environment.

Key Responsibilities:

Write clean, scalable, and efficient Python code for both backend and front-end solutions.
Collaborate with cross-functional teams to gather requirements, design technical solutions, and deliver projects on time.
Develop and maintain APIs, web applications, and data pipelines.
Debug and troubleshoot software issues, providing timely solutions.
Participate in code reviews to ensure code quality and adherence to best practices.
Continuously learn and apply new technologies to improve codebase and development processes.
Assist in testing, maintaining, and improving the software.
Write unit and integration tests to ensure software reliability.
Optimize applications for performance, scalability, and security.
Collaborate in Agile teams, following Scrum or Kanban methodologies.
Required Skills & Qualifications:

2+ years of professional experience in Python development.
Solid understanding of Python frameworks such as Django, Flask, or FastAPI.
Experience working with relational and NoSQL databases (e.g., PostgreSQL, MySQL, MongoDB).
Proficient in writing and consuming RESTful APIs.
Familiarity with version control systems like Git.
Strong knowledge of data structures, algorithms, and object-oriented programming.
Experience with testing frameworks (e.g., pytest, unittest).
Knowledge of Agile development methodologies (Scrum/Kanban).
Strong problem-solving and analytical skills.
Good understanding of code quality practices, including code reviews, test-driven development, and continuous integration.
Preferred Skills:

Knowledge of front-end technologies such as HTML, CSS, JavaScript (React or Vue.js).
Familiarity with containerization technologies (e.g., Docker).
Experience with cloud platforms (e.g., AWS, Google Cloud, Azure).
Experience with task automation and orchestration (e.g., Celery).
Familiarity with CI/CD pipelines and DevOps practices.
Exposure to machine learning libraries such as TensorFlow or PyTorch (optional, depending on the project).
Education:

A Bachelor’s degree in Computer Science, Software Engineering, or a related field, or equivalent experience.
Why Join Us?

Opportunity to work on exciting and impactful projects.
Collaborative and supportive work environment.
Flexible working hours and remote work options.
Career development opportunities and access to continuous learning resources.
Competitive salary and benefits package.
If you are passionate about Python programming and eager to contribute to innovative software solutions, we would love to hear from you! Apply today to be part of our dynamic team.

"""

if __name__ == "__main__":
    job_description_json = job_description_json_format(job_description)
    results = query_similar_results(
        job_description_text= job_description_json.get('aggregate_content'),
        model_name="llama3.2",          # Embedding model name
        qdrant_host="localhost",        # Host address for Qdrant
        qdrant_port=6333,               # Port for Qdrant
        collection_name="resume_collection",  # Name of the Qdrant collection
        top_k=7                         # Number of results to retrieve
    )

    # Print the results
    for idx, result in enumerate(results, start=1):
        print(f"Result {idx}:")
        print(f"Content: {result.page_content}")
        # print(f"Metadata: {result.metadata}")
        print("-" * 50)