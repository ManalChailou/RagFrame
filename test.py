import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = """
Role: You are a software measurement expert specializing in the COSMIC method.
Task: Identify the functional users in the requirement.
Requirement: "The student enters the student ID to view grades."
Return only JSON.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)

print(response.choices[0].message.content)