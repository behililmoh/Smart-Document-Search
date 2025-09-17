import os
import pandas as pd
import numpy as np
import PyPDF2
import docx
from bs4 import BeautifulSoup
import csv
import pickle
import datetime
import fitz # Importez PyMuPDF sous son nom usuel
import re
import hashlib
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    """
    Système de traitement de documents avec génération d'embeddings sémantiques.
    Gère l'upload de fichiers, l'extraction de texte et la création de vecteurs pour la recherche.
    """
    
    def __init__(self):
        # Utilise un modèle de "Sentence Transformer" pour la génération d'embeddings sémantiques
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.labels = []
        self.document_metadata = []

    def extract_snippet(self, text, query, max_chars=300):
       """   Extrait un court extrait de texte (snippet) contenant les mots de la requête.   """
       words = re.findall(r'\b\w+\b', query.lower())
       for word in words:
        match = re.search(r'\b\w*' + re.escape(word) + r'\w*\b', text, re.IGNORECASE)
        if match:
            start_index = max(0, match.start() - max_chars // 2)
            end_index = min(len(text), match.end() + max_chars // 2)
            
            # Ajuster les indices pour ne pas couper un mot
            snippet_start = text.rfind(' ', 0, start_index) + 1 if start_index > 0 else 0
            snippet_end = text.find(' ', end_index) if end_index < len(text) else len(text)
            
            return text[snippet_start:snippet_end]
    
         # Si aucun mot de la requête n'est trouvé, retourner les premiers caractères
        return text[:max_chars]


        
    def upload_document(self, file_path, doc_type="general"):
        """Upload et traite différents types de documents"""
        try:
            text_content = self._extract_text_from_file(file_path)
            if text_content:
              # Nettoyage du texte après l'extraction
              cleaned_text = self._clean_text(text_content)
            
              file_stats = os.stat(file_path)
              doc_metadata = {
                # ... (le reste du dictionnaire reste inchangé)
              }
            
                
            self.documents.append(text_content)
            self.labels.append(doc_type)
            self.document_metadata.append(doc_metadata)
                
            print(f"Document {file_path} traité avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path}: {e}")
            return False
    
    def add_text_directly(self, text_content, doc_type="general", title="Document direct"):
        """Ajoute du texte directement sans fichier"""
        if text_content.strip():
            doc_metadata = {
                'filename': title,
                'full_path': 'direct_input',
                'size': len(text_content.encode()),
                'created': datetime.datetime.now().isoformat(),
                'modified': datetime.datetime.now().isoformat(),
                'added_to_system': datetime.datetime.now().isoformat(),
                'doc_type': doc_type,
                'text_length': len(text_content),
                'hash': hashlib.md5(text_content.encode()).hexdigest()
            }
            
            self.documents.append(text_content.strip())
            self.labels.append(doc_type)
            self.document_metadata.append(doc_metadata)
            
            print(f"Texte ajouté avec succès: {title}")
            return True
        else:
            print("Le texte ne peut pas être vide")
            return False
    
    def create_embeddings(self):
        """Génère des embeddings sémantiques pour tous les documents"""
        if not self.documents:
            raise ValueError("Aucun document à vectoriser")
        
        # Le modèle d'embedding encode tous les documents d'un coup
        embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False)
        return embeddings.astype(np.float32)
    
    def create_query_embedding(self, query):
        """Génère un embedding sémantique pour une requête utilisateur"""
        query_vector = self.embedding_model.encode([query], convert_to_tensor=False)
        return query_vector.astype(np.float32)

    def _extract_text_from_file(self, file_path):
        """Extrait le texte selon le type de fichier"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension in ['.doc', '.docx']:
            return self._extract_from_word(file_path)
        elif file_extension == '.html':
            return self._extract_from_html(file_path)
        elif file_extension == '.csv':
            return self._extract_from_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return self._extract_from_excel(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Type de fichier non supporté: {file_extension}")
        
    def _extract_from_pdf(self, file_path):
        """Extrait le texte d'un fichier PDF en utilisant PyMuPDF. """
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"Erreur lors de l'extraction du PDF avec PyMuPDF: {e}")
            return ""
        return text.strip()
    
    #def _extract_from_pdf(self, file_path):
    #    """Extrait le texte d'un fichier PDF"""
    #    text = ""
    #    with open(file_path, 'rb') as file:
    #        pdf_reader = PyPDF2.PdfReader(file)
    #        for page in pdf_reader.pages:
    #            text += page.extract_text() + "\n"
    #    return text.strip()
    def _clean_text(self, text):
        """Nettoie le texte en supprimant les en-têtes, pieds de page, numéros de page et les multiples espaces. """
    # Expression régulière pour les numéros de page (ex: 1, 2, 3...)
    # ou les en-têtes/pieds de page simples.
    # Note: Ceci est un exemple simple, à adapter selon vos documents.
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Supprime les numéros de page isolés
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE) # Supprime "Page X of Y"
    
    # Supprime les sauts de ligne multiples et les espaces multiples
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
    
        return text.strip()
        
    def _extract_from_word(self, file_path):
        """Extrait le texte d'un fichier Word"""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    
    def _extract_from_html(self, file_path):
        """Extrait le texte d'un fichier HTML"""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            text = soup.get_text()
        return text.strip()
    
    def _extract_from_csv(self, file_path):
        """Extrait le contenu d'un fichier CSV"""
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return text
    
    def _extract_from_excel(self, file_path):
        """Extrait le contenu d'un fichier Excel"""
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return text
    
    def _extract_from_txt(self, file_path):
        """Extrait le contenu d'un fichier texte"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def save_processor_state(self, file_path):
        """Sauvegarde l'état du processeur"""
        state = {
            'documents': self.documents,
            'labels': self.labels,
            'document_metadata': self.document_metadata,
            'save_date': datetime.datetime.now().isoformat()
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"État du processeur sauvegardé dans: {file_path}")

    def load_processor_state(self, file_path):
        """Charge l'état du processeur depuis un fichier"""
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            self.documents = state['documents']
            self.labels = state['labels']
            self.document_metadata = state.get('document_metadata', [])
        print(f"État du processeur chargé depuis: {file_path}")
        print(f"Documents chargés: {len(self.documents)}")
    
    def get_storage_info(self):
        """Retourne des informations sur le stockage"""
        if not self.documents:
            return {"total_documents": 0, "total_size": 0}
        
        total_chars = sum(len(doc) for doc in self.documents)
        total_size_bytes = sum(len(doc.encode()) for doc in self.documents)
        
        storage_info = {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'total_size_bytes': total_size_bytes,
            'total_size_mb': round(total_size_bytes / (1024*1024), 2),
            'document_types': dict(pd.Series(self.labels).value_counts()),
            'documents_with_metadata': len(self.document_metadata)
        }
        
        return storage_info

    def export_documents_to_csv(self, filename="documents_export.csv"):
        """Exporte tous les documents vers un fichier CSV"""
        if not self.documents:
            print("Aucun document à exporter")
            return False
        
        data = []
        for i, (doc, label, metadata) in enumerate(zip(self.documents, self.labels, self.document_metadata)):
            data.append({
                'id': i,
                'filename': metadata.get('filename', f'Document_{i}'),
                'doc_type': label,
                'text_length': len(doc),
                'added_date': metadata.get('added_to_system', 'Unknown'),
                'text_content': doc,
                'hash': metadata.get('hash', 'N/A')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Documents exportés vers: {filename}")
        return True