import os
import boto3
import chromadb
import openai
from dotenv import load_dotenv
import tempfile
import PyPDF2
from docx import Document
from typing import List
import uuid

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize S3 client (will use default credentials if env vars not set)
        s3_kwargs = {}
        if os.getenv('AWS_ACCESS_KEY_ID'):
            s3_kwargs['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID')
        if os.getenv('AWS_SECRET_ACCESS_KEY'):
            s3_kwargs['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY')
        if os.getenv('AWS_DEFAULT_REGION'):
            s3_kwargs['region_name'] = os.getenv('AWS_DEFAULT_REGION')
        
        self.s3_client = boto3.client('s3', **s3_kwargs)
        
        # Initialize ChromaDB with persistence
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name="documents")
            existing_count = self.collection.count()
            print(f"📂 Found existing collection with {existing_count} chunks")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print("📋 Created new persistent collection")
        
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def list_s3_files(self) -> List[str]:
        """List all files in the S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                print(f"📁 Found {len(files)} files in bucket '{self.bucket_name}':")
                for file in files:
                    print(f"  📄 {file}")
                return files
            else:
                print(f"📁 No files found in bucket '{self.bucket_name}'")
                return []
        except Exception as e:
            print(f"❌ Error listing S3 files: {str(e)}")
            return []
    
    def check_document_exists(self, s3_key: str) -> bool:
        """Check if document is already processed in the collection"""
        try:
            results = self.collection.get(
                where={"source": s3_key}
            )
            return len(results['ids']) > 0
        except Exception:
            return False
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            
            # Get unique sources with chunk counts
            if count > 0:
                all_metadata = self.collection.get()['metadatas']
                source_counts = {}
                for meta in all_metadata:
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
    
    def reset_collection(self):
        """Reset the ChromaDB collection (useful for demo purposes)"""
        try:
            self.chroma_client.delete_collection(name="documents")
            print("🗑️ Deleted existing collection")
        except Exception:
            pass  # Collection didn't exist
        
        self.collection = self.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("📋 Created fresh collection")

    def download_file_from_s3(self, s3_key: str) -> str:
        """Download file from S3 and return local temporary file path"""
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1])
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Download file from S3
            self.s3_client.download_file(self.bucket_name, s3_key, temp_file_path)
            print(f"✅ Downloaded {s3_key} from S3")
            
            return temp_file_path
        except Exception as e:
            print(f"❌ Error downloading {s3_key}: {str(e)}")
            raise Exception(f"Failed to download {s3_key}")

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from different file types"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_extension == '.pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif file_extension in ['.docx', '.doc']:
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            else:
                print(f"⚠️ Unsupported file type: {file_extension}")
                return ""
                
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks (optimized for performance)"""
        if not text.strip():
            return []
        
        # Clean the text first
        text = text.replace('\n\n', '\n').replace('\r', '\n')
        
        chunks = []
        text_length = len(text)
        start = 0
        
        print(f"    📊 Processing {text_length} characters...")
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Only look for boundaries if we're not at the end
            if end < text_length:
                # Look for the best breaking point within a small window
                search_start = max(start, end - 100)  # Look back max 100 chars
                
                # Find sentence boundary (period + space)
                period_pos = text.rfind('. ', search_start, end)
                if period_pos > search_start:
                    end = period_pos + 1
                else:
                    # Fall back to word boundary
                    space_pos = text.rfind(' ', search_start, end)
                    if space_pos > search_start:
                        end = space_pos
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Filter out very small chunks
                chunks.append(chunk)
            
            # Progress indicator for large texts
            if len(chunks) % 50 == 0 and len(chunks) > 0:
                progress = (start / text_length) * 100
                print(f"      📈 Progress: {progress:.0f}% ({len(chunks)} chunks created)")
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)  # Ensure we always advance
            
            # Safety check to prevent infinite loops
            if start >= end:
                start = end
        
        print(f"    ✅ Created {len(chunks)} chunks total")
        return chunks

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch (faster than individual calls)"""
        try:
            # Use newer, faster embedding model
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"  # Newer, faster, cheaper model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"❌ Error getting batch embeddings: {str(e)}")
            # Fallback to individual calls if batch fails
            return [self.get_embedding(text) for text in texts]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (fallback method)"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Updated to newer model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {str(e)}")
            return []

    def add_document_to_vectorstore(self, s3_key: str):
        """Download document from S3, process it, and add to vector store"""
        print(f"🔄 Processing document: {s3_key}")
        
        # Check if already processed
        if self.check_document_exists(s3_key):
            print(f"✅ Document {s3_key} already exists in collection (skipping)")
            return True
        
        # Download file
        try:
            temp_file_path = self.download_file_from_s3(s3_key)
        except Exception:
            return False
        
        try:
            # Extract text
            text = self.extract_text_from_file(temp_file_path)
            if not text.strip():
                print(f"⚠️ No text extracted from {s3_key}")
                return False

            # Chunk text
            chunks = self.chunk_text(text)
            print(f"📄 Created {len(chunks)} chunks from {s3_key}")
            
            if not chunks:
                print(f"⚠️ No valid chunks created from {s3_key}")
                return False
            
            # Process chunks in batches for faster embedding
            batch_size = 20  # Process 20 chunks at a time
            documents = []
            metadatas = []
            embeddings = []
            ids = []
            
            print(f"🔄 Processing {len(chunks)} chunks in batches of {batch_size}...")
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_start_idx = i
                
                print(f"  📊 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
                
                # Get embeddings for the batch
                batch_embeddings = self.get_embeddings_batch(batch_chunks)
                
                # Process each chunk in the batch
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    if not embedding:
                        continue
                    
                    # Prepare data for ChromaDB
                    chunk_idx = batch_start_idx + j
                    chunk_id = f"{s3_key}_chunk_{chunk_idx}_{uuid.uuid4().hex[:8]}"
                    documents.append(chunk)
                    metadatas.append({
                        "source": s3_key,
                        "chunk_index": chunk_idx,
                        "chunk_id": chunk_id
                    })
                    embeddings.append(embedding)
                    ids.append(chunk_id)

            # Add to ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                print(f"✅ Added {len(documents)} chunks to persistent vector store")
                return True
            else:
                print(f"⚠️ No valid chunks to add for {s3_key}")
                return False
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def search_similar_chunks(self, query: str, n_results: int = 3) -> List[dict]:
        """Search for similar chunks using the query"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            chunks = []
            if results['documents'] and results['documents'][0]:
                print(f"📋 Found {len(results['documents'][0])} relevant chunks")
                for i in range(len(results['documents'][0])):
                    chunks.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        """Generate answer using OpenAI with retrieved context"""
        try:
            # Prepare context
            context = "\n\n".join([f"Source: {chunk['metadata']['source']}\n{chunk['text']}" 
                                 for chunk in context_chunks])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Get response from OpenAI (using faster model)
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Fast and cost-effective
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Reduced for faster responses
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else "No response generated."
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    def query(self, question: str, n_results: int = 5) -> dict:
        """Main query method - search and generate answer (increased default results)"""
        print(f"\n🔍 Query: {question}")
        
        # Search for similar chunks
        similar_chunks = self.search_similar_chunks(question, n_results)
        
        if not similar_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "chunks_found": 0
            }
        
        print(f"📋 Found {len(similar_chunks)} relevant chunks")
        
        # Show source distribution for debugging
        source_counts = {}
        for chunk in similar_chunks:
            source = chunk['metadata']['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"📊 Source distribution:")
        for source, count in source_counts.items():
            print(f"  📄 {source}: {count} chunks")
        
        # Generate answer
        answer = self.generate_answer(question, similar_chunks)
        
        # Prepare response
        sources = list(set([chunk['metadata']['source'] for chunk in similar_chunks]))
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks_found": len(similar_chunks),
            "chunks": similar_chunks  # Include for debugging
        }
    
    def load_demo_queries(self, filename: str = "questions.txt") -> List[str]:
        """Load demo queries from a text file"""
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


# Example usage and demo
def main():
    """Demo the RAG system with persistence"""
    print("🚀 Starting Simple RAG Demo with Persistence")
    print("=" * 60)
    
    # Initialize RAG system (with persistence)
    rag = SimpleRAG()
    
    # Check current collection status
    info = rag.get_collection_info()
    print(f"\n📊 Current collection status:")
    print(f"  📦 Storage location: {info['persist_directory']}")
    print(f"  📊 Total chunks: {info['total_chunks']}")
    print(f"  📄 Processed documents: {len(info['sources'])}")
    
    if info['sources']:
        print(f"  📋 Documents already in collection:")
        for source in info['sources']:
            chunk_count = info.get('source_counts', {}).get(source, 0)
            print(f"    ✅ {source} ({chunk_count} chunks)")
    
    # Get ALL files from S3 bucket (automatic detection)
    print(f"\n📁 Scanning S3 bucket for all files...")
    available_files = rag.list_s3_files()
    
    # Filter to only PDF files (you can modify this filter as needed)
    pdf_files = [f for f in available_files if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in S3 bucket")
        return
    
    print(f"\n📄 Found {len(pdf_files)} PDF files in S3:")
    for pdf_file in pdf_files:
        print(f"  📄 {pdf_file}")
    
    # Check which files need processing vs already processed
    files_to_process = []
    files_already_processed = []
    
    for pdf_file in pdf_files:
        if rag.check_document_exists(pdf_file):
            files_already_processed.append(pdf_file)
            print(f"✅ {pdf_file} already processed (will skip)")
        else:
            files_to_process.append(pdf_file)
            print(f"🔄 {pdf_file} needs processing")
    
    # Process only new documents
    if files_to_process:
        print(f"\n📚 Processing {len(files_to_process)} new documents...")
        print("=" * 50)
        
        for pdf_file in files_to_process:
            print(f"\n🔄 Processing: {pdf_file}")
            success = rag.add_document_to_vectorstore(pdf_file)
            if success:
                print(f"✅ Successfully processed and stored {pdf_file}")
            else:
                print(f"❌ Failed to process {pdf_file}")
        
        # Show updated collection status
        final_info = rag.get_collection_info()
        print(f"\n📊 Updated collection status:")
        print(f"  📊 Total chunks: {final_info['total_chunks']}")
        print(f"  📄 Total documents: {len(final_info['sources'])}")
    else:
        print(f"\n✅ All {len(files_already_processed)} documents already processed!")
        print("🚀 Using existing embeddings from persistent storage")
    
    # Load demo queries from external file
    demo_queries = rag.load_demo_queries("questions.txt")
    
    print(f"\n💬 Demo Queries:")
    print("=" * 50)
    for query in demo_queries:
        result = rag.query(query)
        
        print(f"\n❓ Question: {query}")
        print(f"💡 Answer: {result['answer']}")
        print(f"📂 Sources: {', '.join(result['sources'])}")
        print("-" * 80)

if __name__ == "__main__":
    main()
