import streamlit as st
import os
import glob
import numpy as np
from document_processor import DocumentProcessor
from vector_search_engine import VectorSearchEngine

# Initialisation des composants du système
@st.cache_resource
def get_system():
    processor = DocumentProcessor()
    dimension = processor.embedding_model.get_sentence_embedding_dimension()
    search_engine = VectorSearchEngine(dimension=dimension)
    return processor, search_engine

processor, search_engine = get_system()

# Chargement ou création de l'index au démarrage
search_engine.load_or_create_index()

# Titre de l'application
st.title("🔎 Moteur de Recherche de Documents Avancé")

# --- Barre latérale pour l'ajout de documents ---
st.sidebar.header("Ajouter de nouveaux documents")
uploaded_files = st.sidebar.file_uploader("Chargez un ou plusieurs fichiers (PDF, DOCX, etc.)", type=["pdf", "docx", "txt", "html", "csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    new_docs = []
    new_embeddings = []
    
    with st.spinner("Traitement des nouveaux documents..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", "raw_documents", uploaded_file.name)
            # Sauvegarde du fichier uploadé sur le disque
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Traitement et ajout des nouveaux documents à l'index
            processor.upload_document(file_path)
            new_doc = processor.documents[-1]
            new_docs.append(new_doc)
            new_embeddings.append(processor.create_query_embedding(new_doc)[0])
    
    if new_docs:
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
        search_engine.add_documents(new_docs, new_embeddings)
        st.sidebar.success(f"{len(new_docs)} document(s) ajouté(s) et indexé(s) !")
    else:
        st.sidebar.warning("Aucun nouveau document à indexer.")
st.sidebar.info(f"Nombre de documents indexés : {len(search_engine.documents)}")


# --- Formulaire de recherche ---
st.header("Effectuer une recherche")
query = st.text_input("Entrez votre requête ici")

if query:
    query_vector = processor.create_query_embedding(query)
    results = search_engine.search(query_vector, k=5)
    
    st.subheader(f"Résultats pour : '{query}'")
    if results:
        for res in results:
            document_text = res['document']
            # Amélioration de l'affichage: extraire le snippet pertinent
            snippet = processor.extract_snippet(document_text, query, max_chars=300)
            
            st.write(f"**Pertinence :** {1 - res['distance']:.2f}")
            st.write(f"**Extrait :** {snippet}...")
            st.markdown("---")
    else:
        st.info("Aucun résultat trouvé.")