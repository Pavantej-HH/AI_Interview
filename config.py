import os
from dotenv import load_dotenv


def load_config(app):
    load_dotenv()

    google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if google_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')


def get_env_vars():
    return {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'GEMINI_API_URL': os.getenv('GEMINI_API_URL'),
        'GOOGLE_CLOUD_PROJECT': os.getenv('GOOGLE_CLOUD_PROJECT'),
    }


