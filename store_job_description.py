import os
import json
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from aggregate_data import aggregate_job_description_data

def process_job_description(json_file_path, model_name, qdrant_host, qdrant_port, collection_name):
    """
    Processes job descriptions from a JSON file and stores them in a Qdrant vector database.

    Args:
        json_file_path (str): Path to the job description JSON file.
        model_name (str): Name of the embedding model.
        qdrant_host (str): Host address for the Qdrant server.
        qdrant_port (int): Port number for the Qdrant server.
        collection_name (str): Name of the Qdrant collection.

    Returns:
        None
    """
    try:
        # Initialize embeddings and Qdrant client
        embeddings = OllamaEmbeddings(model=model_name)
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Load job descriptions from JSON file
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The file {json_file_path} does not exist.")

        with open(json_file_path, 'r') as f:
            job_description_json = json.load(f)

        if not isinstance(job_description_json, list):
            raise ValueError("The JSON file should contain a list of job descriptions.")

        # Check if collection exists, create if not
        if collection_name not in [collection.name for collection in client.get_collections().collections]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )

        # Initialize Qdrant vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        # Process job descriptions and add to vector store
        documents = [
            Document(page_content=aggregate_job_description_data(job), metadata=job)
            for job in job_description_json
        ]

        uuids = [str(uuid4()) for _ in range(len(job_description_json))]

        vector_store.add_documents(documents=documents, ids=uuids)

        print("Data successfully uploaded to the Qdrant vector database.")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")

    except ValueError as val_error:
        print(f"Error: {val_error}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_job_description(
        json_file_path='job_description.json',
        model_name="llama3.2",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name='jobdescription_collection'
    )
