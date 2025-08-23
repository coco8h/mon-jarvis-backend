import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Récupère la clé API de l'environnement (plus sécurisé que de la coder en dur)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

genai.configure(api_key=GOOGLE_API_KEY)
# ### MODIFIÉ ### : J'utilise gemini-1.5-flash. 'gemini-2.5-flash' n'existe pas encore.
# C'est un excellent modèle pour le chat.
model = genai.GenerativeModel('gemini-2.5-flash')

# Route d'accueil (pour vérifier que le serveur tourne)
@app.route('/')
def home():
    return "Jarvis Backend is running!"

# Route pour les requêtes de Jarvis
@app.route('/ask_jarvis', methods=['POST'])
def ask_jarvis():
    data = request.json
    user_input = data.get('prompt')
    
    # ### MODIFIÉ ### : On récupère l'historique de la conversation envoyé par le frontend.
    # Si le frontend n'envoie pas d'historique, on prend une liste vide par défaut.
    history = data.get('history', [])

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # ### MODIFIÉ ### : C'est le changement le plus important !
        # Au lieu d'un simple appel à generate_content, nous démarrons un chat.
        
        # 1. On initialise une session de chat en lui donnant tout l'historique précédent.
        chat_session = model.start_chat(history=history)
        
        # 2. On envoie le nouveau message de l'utilisateur dans cette session de chat.
        response = chat_session.send_message(user_input)
        
        # 3. On renvoie la réponse textuelle. Le modèle a utilisé le contexte pour la générer.
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ne pas utiliser ça pour la production, juste pour les tests locaux
    # Pour Render, le serveur web gère le démarrage
    app.run(debug=True, port=5000)
