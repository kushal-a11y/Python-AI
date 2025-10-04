import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

client = genai.Client(api_key=api_key)

while len(sys.argv) < 1:
    print("Please write a prompt...")

prompt = "".join(sys.argv[1:])

# prompt = (
#     "Why are Boot.dev and FreeCodeCamp such great places to learn backend "
#     "development? Use one paragraph maximum."
# )

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=prompt
)

print("Model response\n")
print(response.text)

usage = response.usage_metadata
print("Usage Metadata\n")
print("No of token(s) consumed\n")

print(f"Prompt: {usage.prompt_token_count}")
print(f"Prompt: {usage.candidates_token_count}")