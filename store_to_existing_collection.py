import os
import json
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def upload_to_existing_collection(json_file, model_name, qdrant_host, qdrant_port, collection_name):
    """
    Uploads new data to an existing Qdrant collection.

    Args:
        json_file_path (str): Path to the JSON file containing the data.
        model_name (str): Name of the embedding model.
        qdrant_host (str): Host address for the Qdrant server.
        qdrant_port (int): Port number for the Qdrant server.
        collection_name (str): Name of the existing Qdrant collection.

    Returns:
        None
    """
    try:
        # Initialize embeddings and Qdrant client
        embeddings = OllamaEmbeddings(model=model_name)
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Check if collection exists
        existing_collections = [collection.name for collection in client.get_collections().collections]
        if collection_name not in existing_collections:
            raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant.")

        if not isinstance(json_file, list):
            raise ValueError("Invalid data format. Expected a list of data entries.")

        # Initialize Qdrant vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        # Process data and add to vector store
        documents = [
            Document(page_content=entry.get('aggregate_content', ''), metadata=entry)
            for entry in json_file
        ]

        uuids = [str(uuid4()) for _ in range(len(json_file))]

        vector_store.add_documents(documents=documents, ids=uuids)

        print(f"Data has been successfully uploaded to the collection '{collection_name}'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    upload_to_existing_collection(
        json_file_path='new_data.json',  # Path to the JSON file containing new data
        model_name="llama3.2",          # Embedding model name
        qdrant_host="localhost",        # Host address for Qdrant
        qdrant_port=6333,               # Port for Qdrant
        collection_name='resume_collection'  # Existing collection name
    )
