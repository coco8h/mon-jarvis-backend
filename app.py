import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
# Augmente la taille maximale des requêtes pour accepter des fichiers
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

CORS(app)

# Configuration de l'API Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")
genai.configure(api_key=GOOGLE_API_KEY)

# Instruction système
system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h (ton pere). Tu es serviable, concis et poli. Tu dois impérativement et TOUJOURS répondre en français, quel que soit le langage de la question de l'utilisateur."
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=system_instruction
)

@app.route('/')
def home():
    return "Jarvis Backend is running!"

@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt', '')  # Utilise une chaîne vide par défaut
    history = data.get('history', [])
    
    # Récupération des données du fichier (Base64)
    file_data = data.get('file_data')
    mime_type = data.get('mime_type')

    if not user_input and not file_data:
        return jsonify({"error": "Aucun prompt ou fichier fourni"}), 400

    try:
        # On construit la liste des "parties" pour la requête Gemini
        content_parts = []
        
        # S'il y a un fichier, on le prépare
        if file_data and mime_type:
            file_bytes = base64.b64decode(file_data)
            content_parts.append({
                "mime_type": mime_type,
                "data": file_bytes
            })
        
        # On ajoute la question de l'utilisateur (même si elle est vide)
        content_parts.append(user_input)

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(content_parts)
        
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
