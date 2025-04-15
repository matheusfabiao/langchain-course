# Importações básicas
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

# Definindo o caminho do arquivo e do diretório de persistência
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books', 'odyssey.txt')
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Checando se o vector store já existe
# Se não existir, cria um novo
if not os.path.exists(persistent_directory):
    print(
        'Persistent directory does not exist. Initializing new vector store...'
    )

    # Garante que o arquivo de texto existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'The file {file_path} does not exist. Please check the file path.'
        )

    # Lê o conteúdo do arquivo de texto
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Divide o texto em chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Imprime informações sobre os documentos divididos
    print(f'\n--- Document Chunks Informations ---')
    print(f'Number of documents: {len(docs)}')
    print(f'Sample chunk: {docs[0].page_content}\n')

    # Cria Embeddings
    print('\n--- Creating Embeddings ---')
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text'
    )   # Atualize com o modelo de embeddings desejado
    print('\n--- Finished Creating Embeddings ---')

    # Cria o vector store e persiste automaticamente
    print('\n--- Creating Vector Store ---')
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print('\n--- Finished Creating Vector Store ---')

else:
    print('Persistent directory already exists. No need to initialize.')
