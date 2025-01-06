import json
import os
import logging
from langchain_ollama.llms import OllamaLLM
from format import job_description_format


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fields for which job descriptions are to be generated
FIELDS = ['software engineer', 'database', 'quality assurance', 'human resources', 'teacher', 'receptionist', 'project manager', 'chef', 'Business Analyst', 'Accountant']


def generate_job_description_for_field(model, field, job_description_format):
    """
    Generates job description for a specific field using the provided model.
    
    Args:
    - model: The language model to use for job description generation.
    - field: The specific field (department) for which the job description is being created.
    - job_description_format: The format in which the job description should be structured.
    
    Returns:
    - dict: The generated job description in JSON format.
    """
    prompt = f"""
    You are an expert in creating a job description for the following {field} department.
    Prepare the job description in JSON format.
    The format is {job_description_format}
    You should only provide the response in JSON format.
    """
    try:
        result = model.invoke(prompt)
        structured_results = json.loads(result.replace('\n', ''))
        return structured_results
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for {field} department.")
        return None
    except Exception as e:
        logging.error(f"Error occurred while generating job description for {field}: {e}")
        return None


def create_job_descriptions(model, fields, job_description_format):
    """
    Creates job descriptions for a list of fields using the provided model.
    
    Args:
    - model: The language model to use for job description generation.
    - fields: A list of fields for which job descriptions need to be created.
    - job_description_format: The format in which the job descriptions should be structured.
    
    Returns:
    - list: A list of generated job descriptions.
    """
    generated_job_descriptions = []
    number_example = 5
    for field in fields:
        for i in range(number_example): 
            logging.info(f"Generating job description for {field} department...")
            job_description = generate_job_description_for_field(model, field, job_description_format)
            if job_description:
                generated_job_descriptions.append(job_description)
    return generated_job_descriptions


def save_job_description_to_file(job_descriptions, filename='job_description.json'):
    """
    Saves the generated job descriptions to a JSON file.
    
    Args:
    - job_descriptions: The list of job descriptions to be saved.
    - filename: The name of the file where job descriptions will be saved.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(job_descriptions, f, indent=4)
        logging.info(f"Job descriptions successfully saved to {filename}.")
    except Exception as e:
        logging.error(f"Failed to save job descriptions to {filename}: {e}")

def main():
    """
    Main function to create and save job descriptions.
    """
    model = OllamaLLM(model='llama3.2')
    
    # Generate job descriptions for all fields
    job_descriptions = create_job_descriptions(model, FIELDS, job_description_format)
    
    # Save the generated job descriptions to a file
    if job_descriptions:
        save_job_description_to_file(job_descriptions)
    else:
        logging.warning("No job descriptions were generated.")


if __name__ == '__main__':
    main()
