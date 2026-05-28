import sys
sys.path.insert(0, "src")

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": "say hello and tell me which model you are"}],
)
print(response.choices[0].message.content)
