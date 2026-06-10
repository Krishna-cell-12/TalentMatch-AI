import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class StructuredJobDescription(BaseModel):
    role_title: str
    seniority_level: str
    must_have_technical_skills: list[str]
    nice_to_have_technical_skills: list[str]
    implicit_behavioral_signals: list[str]
    minimum_years_experience: int | None
    core_responsibilities: list[str]

SYSTEM_PROMPT = """
You are an expert technical recruiter. Analyze the job description and return a structured JSON response matching the requested schema. 
Ignore corporate boilerplate. Respond ONLY with valid raw JSON. Do not include markdown code block formatting.
"""

def parse_job_description(jd_text: str):
    """Sends the raw JD to Groq and parses it into structured JSON."""
    
    # We pass the schema details directly inside the prompt to ensure alignment
    prompt = f"""
    Return a JSON object matching this schema:
    - role_title (string)
    - seniority_level (string)
    - must_have_technical_skills (list of strings)
    - nice_to_have_technical_skills (list of strings)
    - implicit_behavioral_signals (list of strings)
    - minimum_years_experience (integer or null)
    - core_responsibilities (list of strings)

    Analyze this job description:
    {jd_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    jd_path = "src/phase1/sample_jd.txt"
    with open(jd_path, "r") as f:
        raw_jd = f.read()

    print("Analyzing and parsing job description via Free Groq API...")
    parsed_profile = parse_job_description(raw_jd)
    
    output_path = "src/phase1/parsed_jd.json"
    with open(output_path, "w") as f:
        json.dump(parsed_profile, f, indent=4)
        
    print(f"Success! Output saved to {output_path}")