# Resume Recommendation API

This project is a FastAPI application that integrates with Ollama for language model capabilities and Qdrant for vector storage and similarity search. It provides an API endpoint to retrieve similar resumes based on a job description.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the Repository**:
git clone https://github.com/yourusername/resume_recommendation.git
cd resume_recommendation

2. **Set Up a Virtual Environment** (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. **Install Dependencies**:
Create a `requirements.txt` file with the following content:
Then install the dependencies:

4. **Install Ollama**:
Follow the instructions from the [Ollama GitHub repository](https://github.com/ollama/ollama) to install Ollama and ensure that you have the `llama3.2` model available.

5. **Start Qdrant**:
Make sure you have Qdrant running locally on port 6333. You can run it using Docker:

## Running the FastAPI Server

After setting everything up, you can start the FastAPI server with:


The server will be available at `http://localhost:8000`.

## API Endpoint

### Get Similar Resumes

To retrieve similar resumes based on a job description, send a GET request to the following endpoint:


- **Parameters**:
  - `job_description`: The job description text to find similar resumes.
  - `top_k`: (Optional) The number of similar resumes to retrieve (default is 7).

### Example Request

You can test the API using curl or any HTTP client:


## Additional Notes

- Ensure that both Ollama and Qdrant are properly configured and running.
- The FastAPI server and Ollama must be running simultaneously to process requests successfully.
- For more details on FastAPI, refer to the [FastAPI documentation](https://fastapi.tiangolo.com/).

## License

This project is licensed under the MIT License.
