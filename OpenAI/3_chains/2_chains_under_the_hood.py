from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Carregamento das variáveis de ambiente do arquivo .env
load_dotenv()

# Criação do modelo de chat do Ollama com configurações específicas
model = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
)

# Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
        ('human', 'Me conte {quantidade_piadas} piadas engraçadas.'),
    ]
)

# Criação de executáveis para cada parte do pipeline, individualmente
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Criação da cadeia de Sequência utilizando o RunnableSequence
# É equivalente à implementação anterior, com LCEL
chain = RunnableSequence(
    first=format_prompt, middle=[invoke_model], last=parse_output
)

# Execução da cadeia
response = chain.invoke({'assunto': 'cachorros', 'quantidade_piadas': '3'})

# Exibição da resposta
print(response)
