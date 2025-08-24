import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
# Augmente la taille maximale des requêtes pour accepter des fichiers
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app)

# Configuration de l'API Key
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"ERREUR CRITIQUE: Impossible de configurer l'API Gemini. Erreur: {e}")
    GOOGLE_API_KEY = None

# Instruction système pour le modèle de chat
system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h (ton pere). Tu es serviable, concis et poli. Tu dois impérativement et TOUJOURS répondre en français, quel que soit le langage de la question de l'utilisateur."
chat_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)

# Modèle spécifiquement pour l'embedding
embedding_model = genai.GenerativeModel('models/embedding-001')

@app.route('/')
def home():
    return "Jarvis Backend v2.0 (RAG Enabled) is running!"

@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    if not GOOGLE_API_KEY:
        return jsonify({"error": "Serveur non configuré avec une clé API."}), 500
    
    data = request.json
    user_input = data.get('prompt', '')
    history = data.get('history', [])
    file_data = data.get('file_data')
    mime_type = data.get('mime_type')

    if not user_input and not file_data:
        return jsonify({"error": "Aucun prompt ou fichier fourni"}), 400
    
    try:
        content_parts = []
        if file_data and mime_type:
            content_parts.append({"mime_type": mime_type, "data": base64.b64decode(file_data)})
        content_parts.append(user_input)
        
        chat_session = chat_model.start_chat(history=history)
        response = chat_session.send_message(content_parts)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Erreur /ask_jarvis: {e}")
        return jsonify({"error": str(e)}), 500

# Nouvelle route pour vectoriser du texte
@app.route('/embed', methods=['POST'])
def embed_text():
    if not GOOGLE_API_KEY:
        return jsonify({"error": "Serveur non configuré avec une clé API."}), 500
    
    data = request.json
    text_to_embed = data.get('text')
    task_type = data.get('task_type', "RETRIEVAL_DOCUMENT") # Par défaut, pour stocker des documents

    if not text_to_embed:
        return jsonify({"error": "Aucun texte fourni pour l'embedding"}), 400
    
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text_to_embed,
            task_type=task_type
        )
        return jsonify({"embedding": result['embedding']})
    except Exception as e:
        print(f"Erreur /embed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
