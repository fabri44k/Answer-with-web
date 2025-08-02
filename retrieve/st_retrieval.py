from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerRetriever:
   
    # "Qwen/Qwen3-Embedding-0.6B"
   
    __text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=50,
        length_function=len
        # separators=[
            
        # ]
    )

    __MIN_SCORE = 0.4  # Minimum similarity score to consider a chunk relevant
    
   
    def __init__(self, model_name):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__embedder = SentenceTransformer(model_name, device)


    def __split_into_chunk(self, document):
            chunks = self.__text_splitter.split_text(document)
            return chunks
    
    #** MAIN METHOD
    def get_relevant_chunks(self, data, query, max_chunk):
        
        relevant_chunks = []
        chunks = self.__split_into_chunk(data)

        chunks_embeddings = self.__embedder.encode_document(chunks, convert_to_tensor=True)

        top_k = min(max_chunk, len(chunks))
        query_embedding = self.__embedder.encode_query(query, convert_to_tensor=True)

        
        similarity_scores = self.__embedder.similarity(query_embedding, chunks_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)

    
        for score, idx in zip(scores, indices):
            if score >= self.__MIN_SCORE:
                relevant_chunks.append(chunks[idx])
       
        if not relevant_chunks:
            print("[ERROR] No relevant chunks found.")
            return []
        return relevant_chunks