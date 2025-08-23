import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- Configuration de l'API Key (INCHANGÉ) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")
genai.configure(api_key=GOOGLE_API_KEY)

# ### MODIFICATION 1 : AJOUT DE L'INSTRUCTION SYSTÈME ###
# On définit une règle de base pour Jarvis.
system_instruction = "Tu es Jarvis, un assistant IA personnel creer par el coco alias coco8h. Tu es serviable, concis et poli. Tu dois impérativement et TOUJOURS répondre en français, quel que soit le langage de la question de l'utilisateur."

# ### MODIFICATION 2 : INITIALISATION DU MODÈLE AVEC L'INSTRUCTION ###
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=system_instruction  # On applique la règle ici.
)

# --- Route d'accueil (INCHANGÉ) ---
@app.route('/')
def home():
    return "Jarvis Backend is running!"

# --- Route pour les requêtes (INCHANGÉ DANS SA LOGIQUE) ---
@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt')
    history = data.get('history', [])

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini : {e}")
        return jsonify({"error": str(e)}), 500

# --- Lancement du serveur (INCHANGÉ) ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
