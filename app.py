import os
import io
import time
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
SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] # On a juste besoin de lire
DRIVE_FOLDER_NAME = "Jarvis_Knowledge"

# ChromaDB
# On utilise un stockage sur le disque persistant de Render
# Assure-toi d'avoir un "Disk" attaché à ton service sur Render.
CHROMA_PATH = "/var/data/chroma" 
CHROMA_COLLECTION_NAME = "jarvis_documents"

# --- INITIALISATION DES SERVICES ---
# Gemini
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h (ton pere)... Tu dois TOUJOURS répondre en français..."
    chat_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
    embedding_model = genai.GenerativeModel('models/embedding-001')
except Exception as e:
    print(f"ERREUR CRITIQUE: Démarrage impossible. Erreur Gemini: {e}")
    # Gérer l'erreur de manière appropriée, par exemple en arrêtant l'application
    # exit()

# ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

drive_service = None

# --- AUTHENTIFICATION GOOGLE DRIVE ---
def authenticate_drive():
    global drive_service
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    drive_service = build('drive', 'v3', credentials=creds)
    print("Authentification Google Drive réussie.")

# --- LOGIQUE RAG : LECTURE DE DRIVE ET INGESTION DANS CHROMADB ---
def process_and_store_documents():
    print("Début de la synchronisation avec Google Drive...")
    if not drive_service:
        print("Service Drive non authentifié. Synchronisation annulée.")
        return

    try:
        # 1. Trouver l'ID du dossier "Jarvis_Knowledge"
        folder_id = None
        response = drive_service.files().list(q=f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'", spaces='drive').execute()
        if not response.get('files', []):
            print(f"Dossier '{DRIVE_FOLDER_NAME}' non trouvé sur Google Drive.")
            return
        folder_id = response.get('files')[0].get('id')

        # 2. Lister les fichiers dans le dossier
        results = drive_service.files().list(q=f"'{folder_id}' in parents", fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        
        if not items:
            print("Aucun document trouvé dans le dossier Jarvis_Knowledge.")
            return

        print(f"Trouvé {len(items)} document(s) à traiter.")

        for item in items:
            file_id = item['id']
            file_name = item['name']

            # 3. Vérifier si le document est déjà dans notre base de données
            if collection.get(where={"source": file_name})['ids']:
                print(f"Document '{file_name}' déjà mémorisé. Ignoré.")
                continue

            print(f"Traitement du nouveau document : {file_name}")

            # 4. Télécharger et lire le fichier
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            
            text_content = ""
            if file_name.lower().endswith('.pdf'):
                pdf_reader = pypdf.PdfReader(fh)
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            elif file_name.lower().endswith('.txt'):
                text_content = fh.read().decode('utf-8')
            
            # 5. Découper, vectoriser et stocker dans ChromaDB
            chunks = [text_content[i:i + 1000] for i in range(0, len(text_content), 1000)]
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                embedding_result = genai.embed_content(model="models/embedding-001", content=chunk, task_type="RETRIEVAL_DOCUMENT")
                
                collection.add(
                    ids=[f"{file_name}_{i}"],
                    embeddings=[embedding_result['embedding']],
                    documents=[chunk],
                    metadatas=[{"source": file_name}]
                )
                print(f"  -> Paragraphe {i+1}/{len(chunks)} de '{file_name}' mémorisé.")
        
        print("Synchronisation terminée.")

    except HttpError as error:
        print(f"Une erreur est survenue avec l'API Drive: {error}")

# --- ROUTES FLASK ---
@app.route('/')
def home():
    return "Jarvis Backend v3.0 (Cloud Brain) is running!"

@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt', '')
    history = data.get('history', [])

    if not user_input:
        return jsonify({"error": "Aucun prompt fourni"}), 400

    try:
        # 1. Chercher des informations pertinentes dans la base de connaissance
        embedding_result = genai.embed_content(model="models/embedding-001", content=user_input, task_type="RETRIEVAL_QUERY")
        results = collection.query(
            query_embeddings=[embedding_result['embedding']],
            n_results=3 # On prend les 3 résultats les plus pertinents
        )
        
        context = "\n---\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else "Aucun contexte trouvé."

        # 2. Construire le prompt augmenté
        augmented_prompt = f"En te basant sur le contexte de mes documents personnels ci-dessous, réponds à la question. Si l'information n'est pas dans le contexte, dis 'Je ne trouve pas cette information dans mes documents personnels' et réponds ensuite avec tes connaissances générales.\n\n[Contexte de mes documents]\n{context}\n\n[Question de l'utilisateur]\n{user_input}"
        
        # 3. Envoyer à Gemini pour la génération
        chat_session = chat_model.start_chat(history=history)
        response = chat_session.send_message(augmented_prompt)
        
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Erreur /ask_jarvis: {e}")
        return jsonify({"error": str(e)}), 500

# --- Lancement et synchronisation initiale ---
if __name__ == '__main__':
    # Pour le test local
    authenticate_drive()
    process_and_store_documents()
    app.run(debug=True, port=5000)
else:
    # Pour Gunicorn sur Render
    # On attend un peu que le disque soit monté
    time.sleep(5) 
    authenticate_drive()
    process_and_store_documents()
