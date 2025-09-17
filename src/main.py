import os
import glob
from document_processor import DocumentProcessor
from vector_search_engine import VectorSearchEngine
import numpy as np

def main():
    """
    Fonction principale pour exécuter le système de recherche de documents.
    """
    processor = DocumentProcessor()
    search_engine = VectorSearchEngine(dimension=processor.embedding_model.get_sentence_embedding_dimension())
    
    # 1. Chargement de l'état existant ou création d'un nouvel index vide
    search_engine.load_or_create_index()

    # 2. Ajout de nouveaux documents s'il y en a
    print("--- 📂 Vérification des nouveaux documents ---")
    current_dir = os.path.dirname(__file__)
    documents_path = os.path.join(current_dir, '..', 'data', 'raw_documents', '*')
    documents_to_add_path = glob.glob(documents_path)

    new_docs = []
    new_embeddings = []

    # Vérifier quels documents sont nouveaux
    existing_docs = [meta['full_path'] for meta in processor.document_metadata]
    for doc_path in documents_to_add_path:
        if doc_path not in existing_docs:
            if processor.upload_document(doc_path):
                new_docs.append(processor.documents[-1])
                new_embeddings.append(processor.create_query_embedding(new_docs[-1])[0])
            
    if new_docs:
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
        search_engine.add_documents(new_docs, new_embeddings)
    else:
        print("Aucun nouveau document à ajouter.")

    # 3. Exécution de requêtes de recherche
    print("\n--- 🔍 Recherche ---")
    while True:
        user_query = input("Entrez votre requête (ou 'quitter' pour terminer) : ")
        if user_query.lower() == 'quitter':
            break

        if not user_query.strip():
            print("Veuillez entrer une requête valide.")
            continue
        
        try:
            query_vector = processor.create_query_embedding(user_query)
            k_results = 5
            results = search_engine.search(query_vector, k=k_results)
            
            print(f"\nRésultats pour '{user_query}':")
            if results:
                for res in results:
                    print(f"- Distance: {res['distance']:.3f} | Document: {res['document']}")
            else:
                print("Aucun résultat trouvé.")
        
        except Exception as e:
            print(f"Une erreur est survenue lors de la recherche: {e}")

if __name__ == "__main__":
    main()