import json

def aggregate_resume_data(resume):
    # Aggregate Skills
    skills = ", ".join(resume.get("Skills", []))

    # Aggregate Education
    education = " ".join([
        f"{edu.get('Degree', '')} from {edu.get('Institution', '')} ({edu.get('Year', '')})"
        for edu in resume.get("Education", [])
    ])

    # Aggregate Certifications
    certifications = ", ".join(resume.get("Certifications", []))

    # Aggregate Experience
    experience = " ".join([
        f"{exp.get('Title', '')} at {exp.get('Company', '')} ({exp.get('Dates', '')}): " + ", ".join(exp.get("Responsibilities", []))
        for exp in resume.get("Experience", [])
    ])

    # Aggregate Projects
    projects = " ".join([
        f"{proj.get('Title', '')}: {proj.get('Description', '')}"
        for proj in resume.get("Projects", [])
    ])

    # Combine all fields into one string
    aggregated_data = " | ".join(filter(None, [skills, education, certifications, experience, projects]))


    return aggregated_data

def aggregate_job_description_data(job_description):
    """
    Aggregates data from a job description based on specific fields.

    Args:
        job_description (dict): The job description data.

    Returns:
        str: Aggregated data as a single string.
    """
    def extract_qualifications(qualifications):
        """
        Extracts and formats qualifications, handling both list of strings and list of dicts.
        """
        if isinstance(qualifications, list):
            if all(isinstance(q, dict) for q in qualifications):
                return ". ".join(q.get("value", "") for q in qualifications)
            return ". ".join(qualifications)
        return ""

    # Extract relevant fields
    job_title = job_description.get("job_title", "")
    skills = ", ".join(job_description.get("skills", []))
    required_qualifications = extract_qualifications(job_description.get("required_qualifications", []))
    preferred_qualifications = extract_qualifications(job_description.get("preferred_qualifications", []))
    responsibilities = ". ".join(job_description.get("responsibilities", []))

    # Aggregate data
    aggregated_data = (
        f"Job Title: {job_title}. "
        f"Skills: {skills}. "
        f"Required Qualifications: {required_qualifications}. "
        f"Preferred Qualifications: {preferred_qualifications}. "
        f"Responsibilities: {responsibilities}."
    )

    return aggregated_data
