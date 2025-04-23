# Importações necessárias
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

# Define o diretório contendo os arquivos de texto e o diretório de persistência
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, 'books')
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, 'chroma_db_with_metadata')

print(f'Books directory: {books_dir}')
print(f'Persistent directory: {persistent_directory}')

# Checa se o diretório de persistência existe
if not os.path.exists(persistent_directory):
    print('Persistent directory does not exist. Initializing vector store...')

    # Garante que o diretório de livros existe
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f'The directory {books_dir} does not exist. Please check the path.'
        )

    # Lista todos os arquivos de texto no diretório
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    # Lê todos os arquivos e armazena-os em uma lista de documentos com metadados
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding='utf-8')
        book_docs = loader.load()
        for doc in book_docs:
            # Adiciona matadados ao documento, indicando sua fonte de origem
            doc.metadata = {'source': book_file}
            documents.append(doc)

    # Separa os documentos em chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Imprime informações sobre os documentos divididos
    print(f'\n--- Document Chunks Information ---')
    print(f'Number of documents: {len(docs)}')

    # Cria embeddings para os documentos
    print(f'\n--- Creating Embeddings ---')
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    print(f'\n--- Finishing creating Embeddings ---')

    # Cria o vector store a partir dos documentos e embeddings
    print(f'\n--- Creating and persisting Vector Store ---')
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print(f'\n--- Finishing creating and persisting Vector Store ---')
else:
    print('Vector store already exists. No need to initialize.')
