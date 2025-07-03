import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Permet à ton site web de communiquer avec ce serveur
import google.generativeai as genai

app = Flask(__name__)
CORS(app) # Active CORS pour toutes les routes

# Récupère la clé API de l'environnement (plus sécurisé que de la coder en dur)
GOOGLE_API_KEY = os.environ.get("AIzaSyA31a_E7kJKd6Ug2YRZ7QC_yibixZ0yzdk") 
if not GOOGLE_API_KEY:
    raise ValueError("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Route d'accueil (pour vérifier que le serveur tourne)
@app.route('/')
def home():
    return "Jarvis Backend is running!"

# Route pour les requêtes vocales de Jarvis
@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt')

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Appelle Gemini Pro
        response = model.generate_content(user_input)
        # Renvoie la réponse textuelle
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ne pas utiliser ça pour la production, juste pour les tests locaux
    # Pour Render, le serveur web gère le démarrage
    app.run(debug=True, port=5000) 