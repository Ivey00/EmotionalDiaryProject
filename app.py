# app.py
from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from db_connexion import save_to_mongodb, fetch_history_from_mongodb
import logging
from typing import List, Dict, Tuple
import pandas as pd


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Charger le modèle et le tokenizer
try:
    tokenizer = DistilBertTokenizer.from_pretrained('model/')
    model = DistilBertForSequenceClassification.from_pretrained('model/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise

emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
try:
    conseils_df = pd.read_csv('datasets/conseils.csv')
    conseils_dict = dict(zip(conseils_df.Emotion, zip(conseils_df.Emoji, conseils_df.Conseil)))
except Exception as e:
    logger.error(f"Erreur lors du chargement des conseils: {str(e)}")
    conseils_dict = {}

def get_conseil(emotion: str) -> Tuple[str, str]:
    """
    Récupère l'emoji et le conseil associés à une émotion.
    """
    return conseils_dict.get(emotion, ("❓", "Pas de conseil disponible pour cette émotion."))

def predict_emotions_top_k(text: str, k: int = 3) -> List[Dict[str, float]]:
    """
    Prédit les k émotions les plus probables pour un texte donné.
    Retourne une liste de dictionnaires contenant le nom et le score de chaque émotion.
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Le texte ne peut pas être vide et doit être une chaîne de caractères")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
        
        scores = predictions[0].cpu().numpy()
        top_indices = scores.argsort()[-k:][::-1]
        
        return [
            {"name": emotions[i], "score": float(scores[i])}
            for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/nouvelle-entree')
def nouvelle_entree():
    return render_template('nouvelle_entree.html')

@app.route('/historique')
def historique():
    entries = fetch_history_from_mongodb()
    return render_template('historique.html', entries=entries)

@app.route('/api/dashboard-data')
def dashboard_data():
    """
    Fournit les données nécessaires pour alimenter le tableau de bord.
    """
    try:
        # Récupération des données depuis MongoDB
        data = fetch_history_from_mongodb()
        logger.info(f"Données récupérées : {data}")

        if not data:
            logger.warning("Aucune donnée disponible dans MongoDB.")
            return jsonify({
                "main_emotion": "Aucune",
                "total_analyses": 0,
                "positive_emotion": "Aucune",
                "trend": {"labels": [], "datasets": []},
                "distribution": {"labels": [], "data": []}
            })

        emotion_counts = {}
        positive_emotions = ['joy', 'love', 'admiration', 'gratitude', 'pride', 'excitement', 'optimism', 'relief']

        for entry in data:
            if 'predicted_emotions' not in entry or not isinstance(entry['predicted_emotions'], list):
                logger.warning(f"Données mal formatées ignorées : {entry}")
                continue

            for emotion in entry['predicted_emotions']:
                name = emotion['name']
                score = emotion['score']
                emotion_counts[name] = emotion_counts.get(name, 0) + score

        # Trier les émotions par score total
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        main_emotion = sorted_emotions[0][0] if sorted_emotions else 'Aucune'

        # Calculer l'émotion positive la plus fréquente
        positive_emotion_scores = {k: v for k, v in emotion_counts.items() if k in positive_emotions}
        positive_emotion = max(positive_emotion_scores, key=positive_emotion_scores.get, default='Aucune')

        # Préparer les données pour les graphiques
        trend_data = {
            "labels": [entry['timestamp'][:10] for entry in data if 'timestamp' in entry],  # Extraire la date
            "datasets": [{
                "label": "Score total des émotions",
                "data": [sum(e['score'] for e in entry['predicted_emotions']) for entry in data if 'predicted_emotions' in entry],
                "fill": False,
                "borderColor": "#4bc0c0"
            }]
        }

        distribution_data = {
            "labels": [name for name, _ in sorted_emotions],
            "data": [score for _, score in sorted_emotions]
        }

        return jsonify({
            "main_emotion": main_emotion,
            "total_analyses": len(data),
            "positive_emotion": positive_emotion,
            "trend": trend_data,
            "distribution": distribution_data
        })

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données pour le tableau de bord : {e}")
        return jsonify({"error": "Erreur lors du chargement des données"}), 500

@app.route('/tableau-de-bord')
def tableau_de_bord():
    """
    Render the dashboard page.
    """
    return render_template('tableau_de_bord.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Route pour analyser le texte et retourner les émotions prédites avec leurs conseils.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Le contenu doit être en JSON"}), 400

        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "Le champ 'text' est requis"}), 400
            
        text = data['text']
        
        if not text.strip():
            return jsonify({"error": "Le texte ne peut pas être vide"}), 400

        # Prédire les émotions
        emotions_pred = predict_emotions_top_k(text)
        
        # Ajouter les conseils pour chaque émotion
        for emotion in emotions_pred:
            emoji, conseil = get_conseil(emotion['name'])
            emotion['emoji'] = emoji
            emotion['conseil'] = conseil
        
        # Sauvegarder dans MongoDB
        save_success = save_to_mongodb(text, emotions_pred)
        
        if not save_success:
            logger.error("Échec de la sauvegarde dans MongoDB")
            return jsonify({"error": "Erreur lors de la sauvegarde en base de données"}), 500

        return jsonify({
            "emotions": emotions_pred
        })

    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur interne: {str(e)}")
        return jsonify({"error": "Une erreur interne est survenue"}), 500
if __name__ == '__main__':
    app.run(debug=True)