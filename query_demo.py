import os
import chromadb
import openai
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        try:
            self.collection = self.chroma_client.get_collection(name="documents")
            existing_count = self.collection.count()
            print(f"📂 Found existing collection with {existing_count} chunks")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("📋 Created new persistent collection")

    def get_collection_info(self) -> dict:
        try:
            count = self.collection.count()
            if count > 0:
                all_metadata = self.collection.get().get('metadatas') or []
                source_counts = {}
                for meta in all_metadata:
                    if meta is not None:
                        source = meta.get('source', 'unknown')
                        source_counts[source] = source_counts.get(source, 0) + 1
                sources = list(source_counts.keys())
            else:
                sources = []
                source_counts = {}
            return {
                "total_chunks": count,
                "sources": sources,
                "source_counts": source_counts,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}

    def search_similar_chunks(self, query: str, n_results: int = 3) -> List[dict]:
        try:
            print(f"🔍 Searching for: '{query}'")
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            chunks = []
            docs = results.get('documents') or [[]]
            metas = results.get('metadatas') or [[]]
            dists = results.get('distances') or [[]]
            if docs and docs[0]:
                print(f"📋 Found {len(docs[0])} relevant chunks")
                for i in range(len(docs[0])):
                    meta = metas[0][i] if metas and metas[0] and len(metas[0]) > i and metas[0][i] is not None else {}
                    dist = dists[0][i] if dists and dists[0] and len(dists[0]) > i and dists[0][i] is not None else 1.0
                    chunks.append({
                        'text': docs[0][i],
                        'metadata': meta,
                        'distance': dist
                    })
            return chunks
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        try:
            context = "\n\n".join([f"Source: {chunk['metadata']['source']}\n{chunk['text']}" for chunk in context_chunks])
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            content = response.choices[0].message.content
            return content.strip() if content else "No response generated."
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    def query(self, question: str, n_results: int = 5) -> dict:
        print(f"\n🔍 Query: {question}")
        similar_chunks = self.search_similar_chunks(question, n_results)
        if not similar_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "chunks_found": 0
            }
        source_counts = {}
        for chunk in similar_chunks:
            source = chunk['metadata']['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        print(f"📊 Found {len(similar_chunks)} relevant chunks with similarity scores:")
        for i, chunk in enumerate(similar_chunks, 1):
            distance = chunk['distance']
            relevance = "very relevant" if distance < 0.3 else "relevant" if distance < 0.6 else "somewhat relevant"
            source = chunk['metadata']['source']
            clean_text = ' '.join(chunk['text'].split())
            preview = clean_text[:60] + "..." if len(clean_text) > 60 else clean_text
            print(f"  {i}. 📄 {source}")
            print(f"     📊 Distance: {distance:.3f} ({relevance})")
            print(f"     📝 Preview: {preview}")
            print()
        if len(source_counts) > 1:
            print(f"📈 Source distribution:")
            for source, count in source_counts.items():
                print(f"  📄 {source}: {count} chunks")
        answer = self.generate_answer(question, similar_chunks)
        sources = list(set([chunk['metadata']['source'] for chunk in similar_chunks]))
        return {
            "answer": answer,
            "sources": sources,
            "chunks_found": len(similar_chunks),
            "chunks": similar_chunks
        }

    def load_demo_queries(self, filename: str = "questions.txt") -> List[str]:
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f.readlines() if line.strip()]
                print(f"📋 Loaded {len(queries)} questions from {filename}")
                return queries
            else:
                print(f"⚠️ Questions file {filename} not found, using default queries")
                return [
                    "What are the main research topics covered in these documents?",
                    "What is red teaming in AI and why is it important?",
                    "What are the key findings or contributions from these papers?"
                ]
        except Exception as e:
            print(f"❌ Error loading questions file: {str(e)}")
            return ["What are the main topics covered in these documents?"]

def main():
    print("🔎 Simple RAG Query Demo")
    print("=" * 40)
    rag = SimpleRAG()
    info = rag.get_collection_info()
    print(f"\n📊 Collection status:")
    print(f"  📦 Storage location: {info['persist_directory']}")
    print(f"  📊 Total chunks: {info['total_chunks']}")
    print(f"  📄 Documents: {len(info['sources'])}")
    if info['sources']:
        print("  📋 Documents in collection:")
        for source in info['sources']:
            chunk_count = info.get('source_counts', {}).get(source, 0)
            print(f"    ✅ {source} ({chunk_count} chunks)")
    else:
        print("  ⚠️ No documents found in collection. Please run the ingestion workflow first.")
        return
    demo_queries = rag.load_demo_queries("questions.txt")
    print(f"\n💬 Demo Queries:")
    print("=" * 40)
    for query in demo_queries:
        result = rag.query(query)
        print(f"\n❓ Question: {query}")
        print(f"💡 Answer: {result['answer']}")
        print(f"📂 Sources: {', '.join(result['sources'])}")
        print("-" * 60)

if __name__ == "__main__":
    main()
