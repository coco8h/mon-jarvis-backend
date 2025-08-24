import os
import io
import time
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import pypdf
import pinecone # On importe pinecone

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DRIVE_FOLDER_NAME = "Jarvis_Knowledge"

# Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "jarvis-knowledge" # Le nom de ton index sur Pinecone

# --- DECODAGE DES SECRETS (inchangé) ---
# ... (colle ici le bloc de décodage de la réponse précédente)
try:
    if os.environ.get("GOOGLE_CREDENTIALS_B64"):
        creds_json = base64.b64decode(os.environ.get("GOOGLE_CREDENTIALS_B64")).decode('utf-8')
        with open("credentials.json", "w") as f: f.write(creds_json)
    if os.environ.get("GOOGLE_TOKEN_B64"):
        token_json = base64.b64decode(os.environ.get("GOOGLE_TOKEN_B64")).decode('utf-8')
        with open("token.json", "w") as f: f.write(token_json)
except Exception as e:
    print(f"AVERTISSEMENT: Erreur décodage secrets: {e}")


# --- INITIALISATION DES SERVICES ---
# Gemini
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h (ton pere)..."
    chat_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
    embedding_model = genai.GenerativeModel('models/embedding-001')
except Exception as e:
    print(f"ERREUR CRITIQUE: Erreur Gemini: {e}"); exit()

# Pinecone
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(PINECONE_INDEX_NAME)
    print("Connexion à Pinecone réussie.")
except Exception as e:
    print(f"ERREUR CRITIQUE: Connexion à Pinecone impossible: {e}"); exit()

drive_service = None

# --- AUTHENTIFICATION GOOGLE DRIVE (inchangé) ---
def authenticate_drive():
    # ... (colle ici la fonction authenticate_drive de la réponse précédente)
    global drive_service
    creds = None
    if os.path.exists('token.json'): creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('token.json', 'w') as token: token.write(creds.to_json())
        else:
            print("ERREUR CRITIQUE: Auth Google Drive impossible.")
            return
    drive_service = build('drive', 'v3', credentials=creds)
    print("Authentification Google Drive réussie.")


# --- LOGIQUE RAG : LECTURE DE DRIVE ET INGESTION DANS PINECONE ---
def process_and_store_documents():
    print("Début de la synchronisation avec Google Drive...")
    # ... (La logique pour trouver et lire les fichiers sur Drive reste la même)
    # Le changement est dans la partie stockage :
    # ...
    # 5. Découper, vectoriser et stocker dans PINECONE
    # ...
    # (Je remets la fonction complète pour la clarté)
    if not drive_service: print("Service Drive non authentifié."); return
    try:
        folder_id_res = drive_service.files().list(q=f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'", spaces='drive').execute()
        if not folder_id_res.get('files', []): print(f"Dossier '{DRIVE_FOLDER_NAME}' non trouvé."); return
        folder_id = folder_id_res.get('files')[0].get('id')
        results = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items: print("Aucun document trouvé."); return
        
        for item in items:
            file_id, file_name = item['id'], item['name']
            
            # Vérifier si le document est déjà traité (en interrogeant Pinecone avec un metadata filter)
            fetch_res = index.fetch(ids=[f"{file_name}_0"])
            if fetch_res['vectors']:
                print(f"Document '{file_name}' déjà mémorisé. Ignoré.")
                continue

            print(f"Traitement du nouveau document : {file_name}")
            # ... (logique de téléchargement et de lecture de PDF/TXT inchangée)
            request = drive_service.files().get_media(fileId=file_id); fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            text_content = ""
            if file_name.lower().endswith('.pdf'):
                pdf_reader = pypdf.PdfReader(fh)
                for page in pdf_reader.pages: text_content += page.extract_text() or ""
            elif file_name.lower().endswith('.txt'):
                text_content = fh.read().decode('utf-8', errors='ignore')
            
            chunks = [text_content[i:i + 1000] for i in range(0, len(text_content), 1000)]
            
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                embedding = genai.embed_content(model="models/embedding-001", content=chunk, task_type="RETRIEVAL_DOCUMENT")['embedding']
                vectors_to_upsert.append({
                    "id": f"{file_name}_{i}",
                    "values": embedding,
                    "metadata": {"source": file_name, "text": chunk}
                })
            
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert)
                print(f"  -> {len(vectors_to_upsert)} paragraphes de '{file_name}' mémorisés dans Pinecone.")
        
        print("Synchronisation terminée.")
    except Exception as e:
        print(f"Erreur de synchronisation Drive: {e}")


# --- ROUTES FLASK ---
@app.route('/')
def home(): return "Jarvis Backend v4.0 (Pinecone Brain) is running!"

@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt', '')
    history = data.get('history', [])

    if not user_input: return jsonify({"error": "Aucun prompt fourni"}), 400

    try:
        # 1. Vectoriser la question de l'utilisateur
        query_embedding = genai.embed_content(model="models/embedding-001", content=user_input, task_type="RETRIEVAL_QUERY")['embedding']
        
        # 2. Interroger Pinecone pour trouver les textes les plus pertinents
        query_res = index.query(
            vector=query_embedding,
            top_k=3, # On prend les 3 résultats les plus pertinents
            include_metadata=True
        )
        
        context = "\n---\n".join([match['metadata']['text'] for match in query_res['matches']]) if query_res['matches'] else "Aucun contexte trouvé."

        # 3. Construire le prompt augmenté
        augmented_prompt = f"Contexte: {context}\n\nQuestion: {user_input}"
        
        chat_session = chat_model.start_chat(history=history)
        response = chat_session.send_message(augmented_prompt)
        
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Erreur /ask_jarvis: {e}"); return jsonify({"error": str(e)}), 500

# --- Lancement et synchronisation initiale ---
authenticate_drive()
process_and_store_documents()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
