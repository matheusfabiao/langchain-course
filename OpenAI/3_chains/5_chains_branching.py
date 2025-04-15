# Importações necessárias
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_ollama import ChatOllama

# Criação do modelo de chat do Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
)

# Define o prompt template para os diferentes tipos de feedback
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        (
            'human',
            'Generate a thank you note for this positive feedback: {feedback}.',
        ),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        (
            'human',
            'Generate a response addressing this negative feedback: {feedback}.',
        ),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        (
            'human',
            'Generate a request for more details for this neutral feedback: {feedback}.',
        ),
    ]
)

# Escalonamento é um feedback que não é positivo, negativo ou neutro
# Ele deve ser analisado por uma agente humano
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        (
            'human',
            'Generate a message to escalate this feedback to a human agent: {feedback}.',
        ),
    ]
)

# Define o template de classificação de feedback
classification_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        (
            'human',
            'Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}',
        ),
    ]
)

# Define as RunnableBranches para lidar com os diferentes tipos de feedback
branches = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: 'negative' in x,
        negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: 'neutral' in x,
        neutral_feedback_template | model | StrOutputParser(),
    ),
    escalate_feedback_template | model | StrOutputParser(),
)

# Cria a cadeia de classificação de feedback
classification_chain = classification_template | model | StrOutputParser()

# Combina a cadeia de classificação de feedback com as branches
chain = classification_chain | branches

# Exemplos de feedback
# Good Review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad Review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral Review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I am not sure about the product yet. Can you tell me more about its features and benefits?"

# Executa a cadeia com um exemplo de feedback
review = 'I am not sure about the product yet. Can you tell me more about its features and benefits?'
result = chain.invoke({'feedback': review})

# Imprime o resultado
print(result)
