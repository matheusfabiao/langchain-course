# Importação dos pacotes Python necessários para o projeto
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Criação do modelo de chat do Ollama com configurações específicas
model = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
)

# Parte 1: Criação do Prompt utilizando template string
print('\n------ Prompt originado do Template ------\n')
template = 'Conte-me uma piada sobre {assunto}.'
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({'assunto': 'gatos'})
result = model.invoke(prompt)
print(result.content)


# Parte 2: Criação do Prompt utilizando template string com múltiplas variáveis
print('\n------ Prompt originado do Template com múltiplas variáveis ------\n')
template_multiple = """Você é um assistente de IA muito útil.
Human: Me conte uma história {adjetivo} sobre um {animal}.
Assistente:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke(
    {
        'adjetivo': 'engraçada',
        'animal': 'panda',
    }
)
result = model.invoke(prompt)
print(result.content)

# Parte 3: Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
print(
    '\n------ Prompt originado do Template com SystemMessage e HumanMessage (usando tuplas) ------\n'
)
messages = [
    ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
    ('human', 'Me conte {quantidade_piadas} piadas engraçadas.'),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(
    {
        'assunto': 'advogados',
        'quantidade_piadas': 3,
    }
)
result = model.invoke(prompt)
print(result.content)

# Parte 4: Criação do Prompt utilizando SystemMessage e HumanMessage (usando objetos)
print(
    '\n------ Prompt originado do Template com SystemMessage e HumanMessage (usando objetos) ------\n'
)
messages = [
    ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
    HumanMessage(content='Me conte 3 piadas engraçadas.'),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({'assunto': 'advogados'})
result = model.invoke(prompt)
print(result.content)
