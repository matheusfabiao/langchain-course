# Importação dos pacotes Python necessários para o projeto
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Carregando as variáveis de ambiente do arquivo .env
load_dotenv()

# Criação do modelo de chat da OpenAI com configurações específicas
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0,
)

# Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
        ('human', 'Me conte {quantidade_piadas} piadas engraçadas.'),
    ]
)

# Criação da cadeia de execução utilizando LangChain Expression Language (LCEL)
# O StrOutputParser é utilizado para converter a saída do modelo em uma string
chain = prompt_template | model | StrOutputParser()

# Rodando a cadeia de execução
result = chain.invoke({'assunto': 'advogados', 'quantidade_piadas': 3})

# Imprimindo o resultado
print(result)
