# Importação dos pacotes Python necessários para o projeto
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Cria uma instância do modelo de chat do Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
)

# Define uma lista vazia para armazenar o histórico de conversas
chat_history = []

# Adiciona uma mensagem de sistema ao histórico de conversas
system_message = SystemMessage(content='You are a helpful AI assistant.')
chat_history.append(system_message)

# Loop principal do programa
while True:
    # Solicita ao usuário que digite uma mensagem
    query = input('You: ')
    # Verifica se a mensagem é 'exit', se for, encerra o programa
    if query.lower() == 'exit':
        break
    # Adiciona a mensagem do usuário ao histórico de conversas
    chat_history.append(HumanMessage(content=query))
    # Envia o histórico de conversas para o modelo e obtém a resposta
    result = model.invoke(chat_history)
    response = result.content
    # Adiciona a resposta do modelo ao histórico de conversas
    chat_history.append(AIMessage(content=response))
    # Exibe a resposta do modelo
    print('-' * 100)
    print('AI:', response)
    print('-' * 100)

print('----- Chat History -----')
print(chat_history)
