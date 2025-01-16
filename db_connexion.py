from pymongo import MongoClient
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connexion à la base de données MongoDB
try:
    client = MongoClient("mongodb://localhost:27017/")  
    db = client["journal_emotions"]  
    collection = db["entries"]  
    logger.info("Connexion à MongoDB réussie.")
except Exception as e:
    logger.error(f"Erreur de connexion à MongoDB: {str(e)}")

def get_mongo_connection():
    """Établit une nouvelle connexion à MongoDB"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        return client
    except Exception as e:
        logger.error(f"Erreur de connexion à MongoDB: {str(e)}")
        return None

def save_to_mongodb(text, emotions):
    """
    Sauvegarde une entrée de journal et ses émotions dans MongoDB.
    
    Args:
        text (str): Le texte du journal
        emotions (list): Liste des émotions analysées
        
    Returns:
        bool: True si la sauvegarde est réussie, False sinon
    """
    try:
        entry = {
            "text": text,
            "predicted_emotions": emotions,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(entry)
        if result.inserted_id:
            logger.info(f"Journal inséré avec l'ID : {result.inserted_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde dans MongoDB: {str(e)}")
        return False

def fetch_history_from_mongodb():
    """
    Récupère l'historique des entrées depuis MongoDB.
    
    Returns:
        list: Liste des entrées ou liste vide en cas d'erreur
    """
    try:
        client = get_mongo_connection()
        if not client:
            raise Exception("Impossible d'établir la connexion à MongoDB")
        
        db = client['journal_emotions']
        collection = db['entries']
        
        # Récupérer toutes les entrées triées par date
        history = list(collection.find(
            {}, 
            {'text': 1, 'predicted_emotions': 1, 'timestamp': 1}
        ).sort('timestamp', -1))
        
        # Convertir les ObjectId en str pour la sérialisation JSON
        for entry in history:
            entry['_id'] = str(entry['_id'])
        
        client.close()
        return history
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        return []
    finally:
        if 'client' in locals():
            client.close()