import os
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone
import requests
import json

load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API not found in environment variables")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "ubuntu-rag"
        
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        self.documents = self._extract_pdf()
        self._embed_documents()
    
    def _ensure_index_exists(self):
        try:
            indexes = self.pc.list_indexes()
            if self.index_name not in [idx.name for idx in indexes]:
                print(f"Creating index {self.index_name}...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine"
                )
        except Exception as e:
            print(f"Error managing index: {e}")
    
    def _extract_pdf(self) -> List[dict]:
        documents = []
        reader = PdfReader(self.pdf_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append({
                    "page": page_num + 1,
                    "content": text,
                    "metadata": {"source": self.pdf_path, "page": page_num + 1}
                })
        
        print(f"Extracted {len(documents)} pages from PDF")
        return documents
    
    def _get_embeddings(self, text: str) -> List[float]:
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        embedding = []
        for i in range(1536):
            embedding.append(float((hash_int + i) % 1000) / 1000.0)
        
        return embedding
    
    def _embed_documents(self):
        try:
            vectors = []
            for i, doc in enumerate(self.documents):
                embedding = self._get_embeddings(doc["content"])
                vectors.append((
                    f"doc_{i}",
                    embedding,
                    doc["metadata"]
                ))
            
            if vectors:
                self.index.upsert(vectors=vectors)
                print(f"Embedded and stored {len(vectors)} documents in Pinecone")
        except Exception as e:
            print(f"Error embedding documents: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        try:
            query_embedding = self._get_embeddings(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            retrieved_docs = []
            for match in results["matches"]:
                doc_id = match["id"]
                doc_num = int(doc_id.split("_")[1])
                if doc_num < len(self.documents):
                    retrieved_docs.append({
                        "content": self.documents[doc_num]["content"],
                        "page": self.documents[doc_num]["page"],
                        "score": match["score"]
                    })
            
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_response(self, query: str, context: str) -> str:
        try:
            prompt = f"""You are a helpful assistant answering questions about Ubuntu.

Context from the Ubuntu documentation:
{context}

Question: {query}

Provide a concise and helpful answer based on the context provided."""
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error from Groq API: {response.text}"
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str) -> dict:
        retrieved_docs = self.retrieve(user_query)
        
        context = "\n\n".join([
            f"Page {doc['page']}:\n{doc['content'][:500]}..."
            for doc in retrieved_docs
        ])
        
        if not context:
            context = "No relevant documents found in the knowledge base."
        
        response = self.generate_response(user_query, context)
        
        return {
            "query": user_query,
            "response": response,
            "retrieved_docs": retrieved_docs,
            "context": context
        }
