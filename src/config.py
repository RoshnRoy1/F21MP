from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "claude-sonnet-4-6"
API_KEY = os.getenv("ANTHROPIC_API_KEY")

GEMINI_MODEL = "gemini-2.0-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
