# Importação dos pacotes Python necessários para o projeto
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Cria uma instância do modelo de chat da Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
)

# System message:
#     - Mensagem do sistema que define o comportamento do assistente
#     - Pode ser usado para instruir o modelo a seguir um comportamento específico
# Human message:
#     - Mensagem do usuário que representa a entrada do usuário
#     - Pode ser usada para fornecer informações de contexto ou instruções
messages = [
    SystemMessage(content='Solve the following math problems.'),
    HumanMessage(content='What is 81 divided by 9?'),
]

# Invoca o modelo com as mensagens e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer from AI: {result.content}')

# AI message:
#     - Mensagem do modelo que representa a saída do modelo
#     - Pode ser usada para fornecer respostas ou instruções
messages = [
    SystemMessage(content='Solve the following math problems.'),
    HumanMessage(content='What is 81 divided by 9?'),
    AIMessage(content='The answer is 9.'),
    HumanMessage(content='What is 10 times 5?'),
]

# Invoca o modelo com as mensagens e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer from AI: {result.content}')
