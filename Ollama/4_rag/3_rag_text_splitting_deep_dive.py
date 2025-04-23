import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

# Define o diretório do arquivo de texto e do diretório de persistência
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books', 'romeo_and_juliet.txt')
db_dir = os.path.join(current_dir, 'db')

# Checa se o arquivo de texto existe
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f'The file {file_path} does not exist. Please check the path.'
    )

# Lê o conteúdo do arquivo de texto
loader = TextLoader(file_path)
documents = loader.load()

# Define o modelo de embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Função para criar e salvar o vector store do Chroma
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f'\n--- Creating vector store for {store_name} ---')
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f'--- Finished creating vector store for {store_name} ---')
    else:
        print(
            f'\nVector store for {store_name} already exists. No need to initialize.'
        )


# 1. Character-based Splitting
# Divide o texto em chunks baseados em um número específico de caracteres.
# Útil para tamanhos de blocos consistentes, independentemente da estrutura do conteúdo.
print('\n--- Using Character-based Splitting ---')
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, 'chroma_db_char')

# 2. Sentence-based Splitting
# Divide o texto em chunks baseados em sentenças, garantindo que cada chunk seja uma sentença completa.
# Ideal para manter a estrutura natural do texto.
print('\n--- Using Sentence-based Splitting ---')
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, 'chroma_db_sent')

# 3. Token-based Splitting
# Divide o texto em chunks baseados em um número específico de tokens.
# Útil para modelos transformadores com um limite restrito de tokens.
print('\n--- Using Token-based Splitting ---')
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, 'chroma_db_token')

# 4. Recursive Character-based Splitting
# Tenta dividir o texto em limites naturais (frases, parágrafos) dentro do limite de caracteres.
# Equilibra a manutenção da coerência e o cumprimento dos limites de caracteres.
print('\n--- Using Recursive Character-based Splitting ---')
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, 'chroma_db_rec_char')

# 5. Custom Splitting
# Permite a criação de uma função de divisão personalizada para atender às necessidades específicas do projeto.
# Útil para documentos com uma estrutura complexa ou quando é necessário um controle mais refinado sobre a divisão.
print('\n--- Using Custom Splitting ---')


class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str) -> list[str]:
        # Lógica de divisão personalizada (por exemplo, baseada em parágrafos)
        return text.split('\n\n')


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, 'chroma_db_custom')

# Função para realizar uma query no vector store e imprimir os resultados
def query_vector_store(query, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f'\n--- Querying vector store for {store_name} ---')
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings,
        )
        retriever = db.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k': 3, 'score_threshold': 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Imprime os documentos recuperados
        print(f'\n--- Relevant Documents for {store_name} ---')
        for i, doc in enumerate(relevant_docs, 1):
            print(f'Document {i}:\n{doc.page_content}\n')
            if doc.metadata:
                print(f'Source: {doc.metadata.get("source", "Unknown")}\n')
    else:
        print(
            f'\nVector store for {store_name} does not exist. Please create it first.'
        )


# Define a pergunta que será feita
query = 'How did Juliet die?'

# Realiza queries nos vector stores criados
query_vector_store(query, 'chroma_db_char')
query_vector_store(query, 'chroma_db_sent')
query_vector_store(query, 'chroma_db_token')
query_vector_store(query, 'chroma_db_rec_char')
query_vector_store(query, 'chroma_db_custom')
