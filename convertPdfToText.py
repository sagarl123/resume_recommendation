import os
from PyPDF2 import PdfReader


def get_resume_paths(base_dir):
    """
    Recursively fetch all PDF resume file paths from a base directory.

    Args:
        base_dir (str): Path to the directory containing resumes.

    Returns:
        list: List of absolute paths to PDF resume files.
    """
    resume_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                resume_paths.append(os.path.join(root, file))
    return resume_paths


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""


def save_text_to_file(text, output_path):
    """
    Save extracted text to a .txt file.

    Args:
        text (str): Text to be saved.
        output_path (str): Destination file path for the text.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}")


def process_resumes(input_dir, output_dir):
    """
    Process all PDF resumes: Extract text and save it to text files.

    Args:
        input_dir (str): Directory containing PDF resumes.
        output_dir (str): Directory to save extracted text files.
    """
    resume_paths = get_resume_paths(input_dir)
    if not resume_paths:
        print("No PDF files found in the input directory.")
        return

    for resume_path in resume_paths:
        text = extract_text_from_pdf(resume_path)
        if text:
            output_file = os.path.join(output_dir, os.path.basename(resume_path).replace('.pdf', '.txt'))
            save_text_to_file(text, output_file)
    print(f"Processed {len(resume_paths)} resumes. Extracted text files are stored in {output_dir}")


if __name__ == '__main__':
    input_directory = os.path.join(os.getcwd(), 'data/data')
    output_directory = os.path.join(os.getcwd(), 'data/resume_text')
    process_resumes(input_directory, output_directory)
