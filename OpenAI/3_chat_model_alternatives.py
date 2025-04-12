# Importação dos pacotes Python necessários para o projeto
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Carrega as variáveis de ambiente do arquivo.env
load_dotenv()

messages = [
    SystemMessage(content='Solve the following math problems.'),
    HumanMessage(content='What is 81 divided by 9?'),
]

# Exemplo com ChatOpenAI
from langchain_openai import ChatOpenAI

# Cria uma instância do modelo de chat da OpenAI com configurações específicas
model = ChatOpenAI(
    model_name='gpt-4o',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer with ChatOpenAI: {result.content}')

# Exemplo com Anthropic
from langchain_anthropic import ChatAnthropic

# Cria uma instância do modelo de chat da Anthropic com configurações específicas
model = ChatAnthropic(
    model='claude-3-opus-20240229',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer with ChatAnthropic: {result.content}')

# Exemplo com Ollama
from langchain_ollama import ChatOllama

# Cria uma instância do modelo de chat da Ollama com configurações específicas
model = ChatOllama(
    model='llama3.2:8b',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer with ChatOllama: {result.content}')

# Exemplo com ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Cria uma instância do modelo de chat da Google Generative AI com configurações específicas
model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.7,
)

# Envia uma mensagem para o modelo e exibe os resultados formatados
result = model.invoke(messages)
print(f'Answer with ChatGoogleGenerativeAI: {result.content}')
