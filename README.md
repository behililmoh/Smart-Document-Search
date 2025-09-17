
# 🔎 **Smart Document Search**

Ce projet est une solution complète pour la gestion et la recherche sémantique de documents, construite autour d'un moteur de recherche vectorielle rapide et d'une interface utilisateur intuitive.

Il combine la robustesse du traitement de fichiers avec un moteur de recherche **HNSW** (Hierarchical Navigable Small World) pour une performance optimale. Le système est conçu pour être facilement extensible et maintenable.

-----

## 🚀 **Fonctionnalités Clés**

  * **Interface Utilisateur Intuitive** : Une interface web construite avec **Streamlit** permet aux utilisateurs d'interagir facilement avec le système, de télécharger de nouveaux documents et d'effectuer des recherches.
  * **Recherche Sémantique Avancée** : Utilise des **embeddings** (`Sentence-Transformers`) pour comprendre le sens de la requête et trouver les documents les plus pertinents, même si les mots-clés ne correspondent pas exactement.
  * **Ajout Incrémental de Documents** : L'index de recherche peut être mis à jour avec de nouveaux documents sans avoir à être reconstruit entièrement, ce qui le rend efficace pour les grandes collections de données.
  * **Extraction de Texte Robuste** : Gère l'extraction de texte de divers formats de fichiers (PDF, DOCX, etc.) et inclut un nettoyage automatique pour enlever le bruit comme les numéros de page et les en-têtes.
  * **Affichage des Résultats Pertinents** : Au lieu d'afficher le document entier, le système présente des **extraits** (snippets) pertinents qui contiennent la réponse à la requête, améliorant ainsi l'expérience de l'utilisateur.

-----

## 📦 **Structure du Projet**

```
mon_projet_combine/
├── src/
│   ├── app.py                   # L'application web Streamlit.
│   ├── document_processor.py      # Gère l'extraction, le nettoyage et la création des embeddings.
│   └── vector_search_engine.py    # Gère l'indexation HNSW et la recherche.
├── data/
│   ├── raw_documents/             # Dossier où placer vos documents.
│   └── index_data/                # Sauvegarde les index HNSW et les données du système.
├── requirements.txt               # Liste des dépendances.
└── README.md
```

-----

## ⚙️ **Installation et Utilisation**

### 1\. Installation

Assurez-vous d'avoir Python installé, puis naviguez jusqu'à la racine de votre projet dans un terminal. Installez toutes les dépendances requises :

```bash
pip install -r requirements.txt
```

### 2\. Ajout de Documents

Placez les fichiers que vous souhaitez indexer (PDFs, DOCX, TXT, etc.) dans le dossier `data/raw_documents/`.

### 3\. Lancement de l'Application

Depuis le dossier `src`, lancez l'application web avec la commande Streamlit :

```bash
cd src
streamlit run app.py
```

Votre navigateur web s'ouvrira automatiquement et vous affichera l'interface de recherche. Vous pouvez alors ajouter d'autres documents via l'interface et commencer à rechercher.