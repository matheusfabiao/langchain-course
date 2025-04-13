# Importação dos pacotes Python necessários para o projeto
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_ollama import ChatOllama

"""
Passos para replicar este exemplo:
1. Criar uma conta no Firebase
2. Criar um novo projeto Firebase
    - Copiar o ID do projeto
3. Criar um banco de dados Firestore no projeto Firebase
4. Instalar o Google Cloud CLI no seu sistema
    - https://cloud.google.com/sdk/docs/install?hl=pt-br
    - Autenticar-se no Google Cloud CLI com sua conta Google
        - https://cloud.google.com/docs/authentication/provide-credentials-adc?hl=pt-br#local-dev
    - Defina seu projeto padrão para o novo projeto Firebase que você criou
5. Habilite a API do Firebase no Google Cloud Console
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

# Configuração de variáveis do Firestore
PROJECT_ID = 'langchain-demo-49fef'
SESSION_ID = 'user_session'   # Pode ser um nome de usuário ou um ID único
COLLECTION_NAME = 'chat_history'

# Inicializar o Client do Firestore:
#   - O Client do Firestore é um cliente que permite interagir com o Firestore
#   - O Firestore é um banco de dados NoSQL do Firebase
#   - O Client do Firestore é usado para criar, ler, atualizar e deletar documentos no Firestore
#   - O Client do Firestore é inicializado com o ID do projeto
print('Inicializando o Client do Firestore...')
client = firestore.Client(project=PROJECT_ID)

# Inicializar o histórico do chat no Firestore
print('Inicializando o histórico do chat no Firestore...')
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print('Histórico do chat inicializado com sucesso!')
print('Histórico do chat atual:', chat_history.messages)


# Inicializar o modelo de chat do Ollama
print('Inicializando o modelo de chat do Ollama...')
model = ChatOllama(model='qwen2.5:3b', temperature=0)

print(
    "Comece a interação com o modelo de chat do Ollama! Digite 'exit' para sair."
)

while True:
    human_input = input('You: ')
    if human_input.lower() == 'exit':
        break

    chat_history.add_user_message(human_input)

    ai_output = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_output.content)

    print('-' * 100)
    print('AI:', ai_output.content)
    print('-' * 100)
