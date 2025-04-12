# Importação dos pacotes Python necessários para o projeto
from langchain_ollama import ChatOllama

# Cria uma instância do modelo de chat do Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke('What is the capital of France?')
print('Full Result:', end='\n\n')
print(result)
print('-' * 100)
print('Content Only:', end='\n\n')
print(result.content)
