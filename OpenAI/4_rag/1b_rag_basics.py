# Importações necessárias
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ollama import embeddings

# Definindo o caminho do diretório de persistência
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Define o modelo de Embedding
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Carrega o vector store existente com a função de embedding
db = Chroma(
    persist_directory=persistent_directory, embedding_function=embeddings
)

# Define a pergunta do usuário
query = 'Who is Odysseys wife?'

# Cria um retriever para recuperar os documentos mais relevantes
# e define o limiar de similaridade
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 3, 'score_threshold': 0.4},
)

# Recupera os documentos mais relevantes com base na pergunta
relevant_docs = retriever.invoke(query)

# Imprime os documentos relevantes
print(f'\n--- Relevant Documents ---')
for i, doc in enumerate(relevant_docs, 1):
    print(f'Document {i}:\n{doc.page_content}\n')
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
