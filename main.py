import os
import json
from fastapi import FastAPI, HTTPException, Query
from typing import List
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
from format import job_description_format
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Qdrant and embeddings settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "resume_collection"
EMBEDDING_MODEL = "llama3.2"

# Initialize clients (reuse connections)
@lru_cache()
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@lru_cache()
def get_ollama_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# Helper function: Extract job description and format into JSON
def job_description_json_format(job_description: str, format_template: str):
    model = OllamaLLM(model=EMBEDDING_MODEL)
    prompt = f"""
        Extract key information from the job description into JSON format:
        Template: {format_template}
        Job Description: {job_description}
    """
    response = model.invoke(prompt).replace("\n", "")
    json_response = json.loads(response)
    return json_response

# Query Qdrant for similar results
async def query_similar_results(job_description_text: str, top_k: int = 7):
    try:
        embeddings = get_ollama_embeddings()
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        # Get embeddings for the query text
        query_embedding = await run_in_threadpool(embeddings.embed_query, job_description_text)

        # Query the vector store
        results = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k,
        )

        # Include similarity scores
        return [
            {"content": result.page_content, "similarity": result.score}
            for result in sorted(results, key=lambda x: x.score, reverse=True)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying similar results: {e}")

# API endpoint for GET request
@app.get("/similar-resumes", response_model=List[dict])
async def get_similar_resumes(
    job_description: str = Query(..., description="Job description text to find similar resumes"),
    top_k: int = Query(7, description="Number of similar resumes to retrieve"),
):
    """
    GET endpoint to retrieve similar resumes based on the provided job description.
    """
    try:
        # Process job description JSON
        formatted_json = await run_in_threadpool(
            job_description_json_format,
            job_description,
            job_description_format,
        )
        aggregate_content = formatted_json.get("aggregate_content")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing job description: {e}"
        )

    # Query for similar results
    similar_results = await query_similar_results(aggregate_content, top_k=top_k)
    return similar_results
