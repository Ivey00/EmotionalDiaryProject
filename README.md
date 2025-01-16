# EmotionalDiaryProject

## Introduction

Ce projet vise à développer une application intégrant le traitement automatique du langage naturel (NLP) pour analyser et interpréter les émotions exprimées dans des données textuelles. En utilisant GoEmotions, une version fine-tunée de DistilBERT, l'application prédit les émotions basées sur les entrées des utilisateurs.

L'application utilise Flask comme framework principal pour l'interface web et MongoDB pour la gestion de la base de données, assurant ainsi flexibilité, évolutivité et facilité de traitement et de stockage des données utilisateur.

## Vidéo de Démonstration
![image](https://github.com/user-attachments/assets/44c0dd12-fa72-46d8-975d-7ce0f8e44243)
![image](https://github.com/user-attachments/assets/ad179d9c-585e-43b2-a8fc-5b3979bd2bd3)

Vous pouvez visionner une démonstration de l'application en ouvrant le fichier vidéo disponible dans le projet.

Chemin : `static/videos/demo.mp4`

Pour la lire, utilisez un lecteur multimédia ou ouvrez le fichier dans votre navigateur.

### Fonctionnalités

- Prédiction des émotions : Analysez du texte pour prédire des émotions comme la joie, la tristesse, la colère, etc.

- Sauvegarde des entrées : Enregistrez les entrées utilisateur et leurs émotions prédites dans MongoDB.

- Historique des entrées : Consultez un historique complet des textes analysés et des émotions associées.

## Structure du Répertoire

La structure du projet est organisée comme suit :

```
EmotionalDiaryProject/
├── datasets/
├── installation/
├── model/
├── notebooks/
├── src/
├── static/
│   └── css/
└── templates/
```

### Détails des Répertoires et Fichiers

- **datasets/** : Contient les ensembles de données utilisés pour entraîner et tester le modèle NLP.
  - **conseils.csv** : Une petitbase de données contient les conseils associés à chaque émotions.
  - **go_emotions_train.csv** : Contient l'ensembles de données utilisés pour entraîner le modèle.
  - **go_emotions_validation.csv** : Contient l'ensembles de données utilisés pour valider le modèle.
  - **go_emotions_test.csv** : Contient l'ensembles de données utilisés pour tester le modèle.

- **installation/** : Fournit des scripts et des instructions pour installer les dépendances et configurer l'environnement de développement.
  - **requirement.txt** : Les dependences nécessaire pour l'environment.
  - **installation.ipynb** : installation des dépendances dans kernel d'environment.
  
- **model/** : Inclut les fichiers relatifs au modèle NLP après le fine-tuning. Ces fichiers sont essentiels pour le déploiement et l'inférence :
  - **config.json** : Contient les paramètres de configuration du modèle, comme la taille du vocabulaire, les dimensions des embeddings et d'autres détails structurels.
  - **model.safetensors** : Stocke les poids du modèle fine-tuné dans un format optimisé pour la sécurité et l'efficacité.
  - **special_tokens_map.json** : Définit les tokens spéciaux comme `[CLS]`, `[SEP]`, ou `[PAD]` utilisés par le modèle.
  - **tokenizer_config.json** : Fournit les paramètres de configuration du tokenizer, comme le type de tokenizer et les règles de pré-traitement.
  - **vocab.txt** : Liste les mots et leurs indices dans le vocabulaire du modèle.

- **notebooks/** : Contient des notebooks Jupyter utilisés pour l'exploration des données, le prétraitement et l'expérimentation avec le modèle.

- **src/** : Renferme le code source principal de l'application, y compris :
  - **train.py** : Script pour entraîner le modèle NLP sur les données pré-traitées. Ce fichier inclut les fonctions pour charger les données, définir le modèle, configurer les hyperparamètres, et lancer l'entraînement.
  - **test.py** : Script pour évaluer le modèle entraîné.

- **static/** : Regroupe les fichiers statiques tels que les images, les fichiers JavaScript et les feuilles de style CSS.
  - **css/** : Contient les fichiers CSS pour le style de l'application.

- **templates/** : Inclut les fichiers HTML utilisés par Flask pour rendre les pages web.

## Installation

Pour installer et exécuter l'application localement, suivez les étapes ci-dessous :

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/Ivey00/EmotionalDiaryProject.git
   cd EmotionalDiaryProject
   ```

2. **Créer un environnement virtuel** :

   ```bash
   python -m venv env
   source env/bin/activate  # Sur Windows : env\Scripts\activate
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r installation/requirements.txt
   ```

4. **Configurer la base de données** :

   Assurez-vous que MongoDB est installé et en cours d'exécution sur votre machine. Créez une base de données nommée `emotional_diary` et mettez à jour les informations de connexion dans le fichier de configuration approprié.

5. **Exécuter l'application** :

   ```bash
   flask run
   ```

   L'application sera accessible à l'adresse `http://127.0.0.1:5000/`.

## Utilisation

Une fois l'application en cours d'exécution, vous pouvez accéder à l'interface web pour saisir du texte. L'application analysera le texte et prédira les émotions correspondantes en utilisant le modèle GoEmotions.

## Contribution

Les contributions sont les bienvenues ! Veuillez soumettre une *issue* pour signaler des bogues ou proposer des améliorations. Les *pull requests* seront examinées attentivement.

## Licence

Ce projet est sous licence MIT. Veuillez consulter le fichier `LICENSE` pour plus de détails.

---

