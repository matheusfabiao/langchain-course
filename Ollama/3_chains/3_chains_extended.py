# Importações necessárias
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_ollama import ChatOllama

# Criação do modelo de chat do Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
)

# Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
        ('human', 'Me conte {quantidade_piadas} piadas engraçadas.'),
    ]
)

# Define passos de processamento adicionais com RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(
    lambda x: f'Quantidade de palavras: {len(x.split())}\n{x}'
)

# Cria a cadeia de execução utilizando Langchain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | uppercase_output
    | count_words
)

# Executando a cadeia com um dicionário de entrada
result = chain.invoke({'assunto': 'pessoas', 'quantidade_piadas': 2})

# Imprimindo o resultado
print(result)
