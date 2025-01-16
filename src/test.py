import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from db_connexion import save_to_mongodb

# Charger le modèle et le tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('model/')
model = DistilBertForSequenceClassification.from_pretrained('model/')

# Configuration du device (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Liste des émotions dans l'ordre
emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def predict_emotions_top_k(text, k=3):
    # Tokeniser le texte
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)

    # Trier les émotions par score décroissant
    scores = predictions[0].cpu().numpy()
    top_indices = scores.argsort()[-k:][::-1]  # Indices des k scores les plus élevés
    top_emotions = [f"{emotions[i]} ({scores[i]:.2f})" for i in top_indices]
    # Stocker dans MongoDB
    save_to_mongodb(text, top_emotions)

    return top_emotions


# Tester le modèle avec une phrase
# Exemple de texte à tester
test_text = "I am very happy and grateful to God because he helped me a lot and made what I wanted come true."
    
# Obtenir les émotions prédites
result = predict_emotions_top_k(test_text, k=3)

    
# Afficher les résultats
print("Texte :", test_text)
print("Émotions prédites :", result)
