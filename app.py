import os
import io
import time
import base64 # On a besoin de ça pour décoder
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Imports pour Google Drive & Auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Imports pour le traitement des documents et la DB vectorielle
import pypdf
import chromadb

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DRIVE_FOLDER_NAME = "Jarvis_Knowledge"

# ChromaDB
CHROMA_PATH = "/var/data/chroma" # Point de montage du disque persistant de Render
CHROMA_COLLECTION_NAME = "jarvis_documents"

# --- DECODAGE DES SECRETS DEPUIS LES VARIABLES D'ENVIRONNEMENT ---
# C'est la magie qui fait fonctionner l'authentification sur Render
try:
    if os.environ.get("GOOGLE_CREDENTIALS_B64"):
        creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_B64")
        creds_json = base64.b64decode(creds_b64).decode('utf-8')
        with open("credentials.json", "w") as f:
            f.write(creds_json)
    
    if os.environ.get("GOOGLE_TOKEN_B64"):
        token_b64 = os.environ.get("GOOGLE_TOKEN_B64")
        token_json = base64.b64decode(token_b64).decode('utf-8')
        with open("token.json", "w") as f:
            f.write(token_json)
except Exception as e:
    print(f"AVERTISSEMENT: Impossible de décoder les secrets depuis les variables d'environnement. Erreur: {e}")


# --- INITIALISATION DES SERVICES ---
# Gemini
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h (ton pere)... Tu dois TOUJOURS répondre en français..."
    chat_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
    embedding_model = genai.GenerativeModel('models/embedding-001')
except Exception as e:
    print(f"ERREUR CRITIQUE: Démarrage impossible. Erreur Gemini: {e}")
    exit()

# ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

drive_service = None

# --- AUTHENTIFICATION GOOGLE DRIVE (lit maintenant les fichiers créés) ---
def authenticate_drive():
    global drive_service
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # Sur Render, on ne peut pas lancer le serveur local pour l'auth, donc si ça échoue, c'est une erreur de config
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Sauvegarde le token rafraîchi (important pour la longue durée)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        else:
            print("ERREUR CRITIQUE: Impossible de s'authentifier à Google Drive. 'token.json' est manquant, invalide ou expiré et ne peut pas être rafraîchi.")
            print("Assurez-vous que la variable d'environnement GOOGLE_TOKEN_B64 est correcte.")
            return

    drive_service = build('drive', 'v3', credentials=creds)
    print("Authentification Google Drive réussie.")

# --- Le reste du code (process_and_store_documents, routes Flask) est INCHANGÉ ---
# ... (colle ici le reste de ton app.py de la réponse précédente, à partir de la fonction `process_and_store_documents`) ...

def process_and_store_documents():
    # ... (code inchangé)
    print("Début de la synchronisation avec Google Drive...")
    # ... etc
def home():
    # ... (code inchangé)
def ask_jarvis():
    # ... (code inchangé)


# --- Lancement et synchronisation initiale ---
# Cette logique gère à la fois le lancement local et sur Render
print("Lancement du service Jarvis...")
time.sleep(5) # Laisse le temps au disque de Render de se monter
authenticate_drive()
process_and_store_documents() # Synchronise au démarrage

if __name__ == '__main__':
    app.run(debug=True, port=5000)
