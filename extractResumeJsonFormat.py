import os
import json
import random
import logging 
from langchain_ollama.llms import OllamaLLM
from langdetect import detect 
from format import resume_format 

def load_resumes(file_dir, sample_size):
    """
    Loads random sample size resumes from given directory and returns them as a list
    """
    try:
        file_paths = os.listdir(file_dir)
        if not file_paths:
            raise FileNotFoundError("No files found in the directory.")

        sample_size = min(sample_size, len(file_paths))
        random_sample = random.sample(file_paths, sample_size)

        resumes = []
        for path in random_sample:
            abs_path = os.path.join(file_dir, path)
            with open(abs_path, 'r', encoding='utf-8') as f:
                resumes.append(f.read())

        return resumes
    except Exception as e:
        raise Exception(f"Error loading resumes: {e}")

def extract_resume_data(model, resumes, resume_format):
    """
    Extracts structured data from resumes using llm model 
    """
    extracted_data = []

    for resume in resumes:
        try:
            detected_language = detect(resume)
             # Detect the language of the resume text
            logging.info(f"Detected language: {detected_language}")
            if detected_language != 'en':
                resume = convert_to_english(resume, detected_language = detected_language, model = model)
            prompt = f"""
            Please provide the output strictly in JSON format without any additional comments or explanations.
            From the given resume, extract the information in JSON format.
            If there are no field values, leave it empty.
            The given resume is {resume}.
            The response format is {resume_format}.
            """
            result = model.invoke(prompt)
            structured_result = json.loads(result.replace('\n', ''))
            extracted_data.append(structured_result)
        except json.JSONDecodeError as jde:
            print(f"JSON decoding error for resume: {resume[:50]}... Error: {jde}")
        except Exception as e:
            print(f"Error processing resume: {resume[:50]}... Error: {e}")

    return extracted_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_english(resume_text, detected_language ,model):
    """
    Translates a resume text to English if it's not already in English.
    
    Args:
    - resume_text: The resume text that needs to be translated.
    - model: The language model used for translation.
    
    Returns:
    - str: The translated resume text in English, or the original text if it is already in English.
    """
    try:
        if detected_language != 'en':
            # If the language is not English, translate the resume to English
            prompt = f"""
            You are an expert in language translation.
            The given language is {detected_language}.
            Please translate the following resume to English:
            {resume_text}
            """
            # Invoke the model to translate the resume
            translated_resume = model.invoke(prompt).replace('\n', '')
            logging.info("Translation successful.")
            return translated_resume
        else:
            logging.info("Resume is already in English.")
            return resume_text  # Return the original resume if it's already in English
    
    except Exception as e:
        logging.error(f"Error during language detection or translation: {e}")
        return resume_text  # Return the original text in case of error

def save_to_json(data, output_file):
    """
    Extracts data and stored in JSON file 
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        raise Exception(f"Error saving data to JSON file: {e}")


def main():
    file_dir = os.path.join(os.getcwd(), 'data/resume_text')
    output_file = 'resumes_json.json'
    sample_size = 250

    try:
        model = OllamaLLM(model='llama3.2')
        resumes = load_resumes(file_dir, sample_size)
        extracted_data = extract_resume_data(model, resumes, resume_format = resume_format)
        save_to_json(extracted_data, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
