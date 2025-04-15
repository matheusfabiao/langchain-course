# Importações necessárias
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Criação do modelo de chat da OpenAI com configurações específicas
model = ChatOpenAI(
    model='gpt-4',
    temperature=0,
)

# Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an expert product reviewer.'),
        ('human', 'List the main features of the product {product_name}.'),
    ]
)

# Define o passo de análise dos prós com RunnableLambd
def analyse_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are an expert product reviewer.'),
            (
                'human',
                'Given these features: {features}, list the pros of these features.',
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define o passo de análise dos contras com RunnableLambda
def analyse_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are an expert product reviewer.'),
            (
                'human',
                'Given these features: {features}, list the cons of these features.',
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combinar prós e contras em uma review única
def combine_pros_cons(pros, cons):
    return f'Pros:\n{pros}\n\nCons:\n{cons}'


# Simplicando as branches com LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyse_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyse_cons(x)) | model | StrOutputParser()
)

# Criando uma cadeia combinada usando Langchain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(
        branches={'pros': pros_branch_chain, 'cons': cons_branch_chain}
    )
    | RunnableLambda(
        lambda x: combine_pros_cons(
            x['branches']['pros'], x['branches']['cons']
        )
    )
)

# Executando a cadeia com um dicionário de entrada
result = chain.invoke({'product_name': 'MacBook Pro'})

# Imprimindo o resultado
print(result)
