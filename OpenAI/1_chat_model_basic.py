# Importação dos pacotes Python necessários para o projeto
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Cria uma instância do modelo de chat da OpenAI com configurações específicas
model = ChatOpenAI(
    model_name='gpt-4o',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke('What is the capital of France?')
print('Full Result:', end='\n\n')
print(result)
print('-' * 100)
print('Content Only:', end='\n\n')
print(result.content)
