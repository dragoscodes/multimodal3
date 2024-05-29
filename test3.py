import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print(os.environ.get('AWS_ACCESS_KEY_ID'))