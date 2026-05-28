from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "claude-sonnet-4-6"
API_KEY = os.getenv("ANTHROPIC_API_KEY")

GEMINI_MODEL = "gemini-2.0-flash-lite"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
