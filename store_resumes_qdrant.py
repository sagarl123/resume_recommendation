import os
import json
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from aggregate_data import aggregate_resume_data

def process_resumes(json_file_path, model_name, qdrant_host, qdrant_port, collection_name):
    """
    Processes resumes from a JSON file and stores them in a Qdrant vector database.

    Args:
        json_file_path (str): Path to the resumes JSON file.
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

        # Load resumes from JSON file
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The file '{json_file_path}' does not exist.")

        with open(json_file_path, 'r') as f:
            resumes_json = json.load(f)

        if not isinstance(resumes_json, list):
            raise ValueError("Invalid data format. Expected a list of resumes.")

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

        # Process resumes and add to vector store
        documents = [
            Document(page_content=aggregate_resume_data(resume), metadata=resume)
            for resume in resumes_json
        ]

        uuids = [str(uuid4()) for _ in range(len(resumes_json))]

        vector_store.add_documents(documents=documents, ids=uuids)

        print("Resumes have been successfully uploaded to the vector store.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_resumes(
        json_file_path='resumes_json.json',
        model_name="llama3.2",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name='resume_collection'
    )
