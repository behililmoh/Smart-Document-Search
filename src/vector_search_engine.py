import os
import numpy as np
import hnswlib
import pickle

class VectorSearchEngine:
    """
    Classe pour gérer un moteur de recherche vectorielle en utilisant hnswlib.
    Crée et interroge un index HNSW pour une recherche rapide des plus proches voisins.
    """

    def __init__(self, dimension=128):
        self.dimension = dimension
        self.documents = []
        self.hnsw_index = None
        self.embeddings = None
        self.index_path = "data/index_data/hnsw_index.bin"
        self.embeddings_path = "data/index_data/embeddings.npy"
        self.documents_path = "data/index_data/documents.pkl"
        os.makedirs("data/index_data", exist_ok=True)

    def load_or_create_index(self):
        """
        Charge un index existant ou en crée un nouveau vide.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            print("Chargement des index et des vecteurs existants...")
            self.load_state()
            print(f"Index chargé avec {len(self.documents)} documents.")
        else:
            print("Création d'un nouvel index...")
            # Création de l'index HNSW vide
            self.hnsw_index = hnswlib.Index(space="cosine", dim=self.dimension)
            self.hnsw_index.init_index(max_elements=100000) # Capacité initiale
            self.embeddings = np.array([], dtype=np.float32).reshape(0, self.dimension)
            print("Nouvel index créé.")

    def add_documents(self, documents_to_add, embeddings_to_add):
        """
        Ajoute de nouveaux documents à l'index existant de manière incrémentale.
        """
        if not self.hnsw_index:
            self.load_or_create_index()

        num_new_docs = len(documents_to_add)
        total_docs = len(self.documents) + num_new_docs

        if total_docs > self.hnsw_index.get_max_elements():
            print(f"Redimensionnement de l'index de {self.hnsw_index.get_max_elements()} à {total_docs + 10000}...")
            self.hnsw_index.resize_index(total_docs + 10000)

        # Ajouter les nouveaux documents et embeddings
        new_ids = list(range(len(self.documents), total_docs))
        self.hnsw_index.add_items(embeddings_to_add, new_ids)

        # Mettre à jour les listes de documents et embeddings
        self.documents.extend(documents_to_add)
        self.embeddings = np.vstack([self.embeddings, embeddings_to_add])
        
        # Sauvegarde
        self.save_state()
        print(f"Ajout de {num_new_docs} documents. Total: {len(self.documents)}.")

    def save_state(self):
        """
        Sauvegarde l'index HNSW et les listes de documents et d'embeddings.
        """
        self.hnsw_index.save_index(self.index_path)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def load_state(self):
        """
        Charge l'index et les listes de documents et d'embeddings.
        """
        self.hnsw_index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.hnsw_index.load_index(self.index_path)
        self.embeddings = np.load(self.embeddings_path)
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)

    def search(self, query_vector, k=5):
        """
        Recherche les k documents les plus proches d'un vecteur de requête.
        """
        if not self.hnsw_index:
            raise ValueError("Index non chargé ou créé.")
        
        if len(self.documents) < k:
            k = len(self.documents)

        labels, distances = self.hnsw_index.knn_query(query_vector, k=k)
        
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            results.append({
                'id': int(idx),
                'distance': dist,
                'document': self.documents[idx]
            })
        
        return results