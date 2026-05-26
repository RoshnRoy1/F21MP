import sys
sys.path.insert(0, "src")

from google import genai
from config import GEMINI_MODEL, GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)
response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="Say hello and tell me which model you are.",
)
print(response.text)
