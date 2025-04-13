# Importação dos pacotes Python necessários para o projeto
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Parte 1: Criação do Prompt utilizando template string
template = 'Conte-me uma piada sobre {assunto}.'
prompt_template = ChatPromptTemplate.from_template(template)

print('\n------ Prompt originado do Template ------\n')
prompt = prompt_template.invoke({'assunto': 'gatos'})
print(prompt)


# Parte 2: Criação do Prompt utilizando template string com múltiplas variáveis
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
print('\n------ Prompt originado do Template com múltiplas variáveis ------\n')
print(prompt)

# Parte 3: Criação do Prompt utilizando SystemMessage e HumanMessage (usando tuplas)
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
print(
    '\n------ Prompt originado do Template com SystemMessage e HumanMessage (usando tuplas) ------\n'
)
print(prompt)

# Parte 4: Criação do Prompt utilizando SystemMessage e HumanMessage (usando objetos)
#   - Este método funciona desde que não haja variáveis no objeto HumanMessage
#   - O objeto HumanMessage aceita apenas strings
messages = [
    ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
    HumanMessage(content='Me conte 3 piadas engraçadas.'),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({'assunto': 'advogados'})
print(
    '\n------ Prompt originado do Template com SystemMessage e HumanMessage (usando objetos) ------\n'
)
print(prompt)

# Parte 5: Criação do Prompt utilizando SystemMessage e HumanMessage (usando objetos)
#   - Este método não funciona, pois o objeto HumanMessage não aceita variáveis
#   - O objeto HumanMessage aceita apenas strings
messages = [
    ('system', 'Você é um comediante que faz piadas sobre {assunto}.'),
    HumanMessage(content='Me conte {quantidade_piadas} piadas engraçadas.'),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(
    {
        'assunto': 'advogados',
        'quantidade_piadas': 3,
    }
)
print(
    '\n------ Prompt originado do Template com SystemMessage e HumanMessage (usando objetos) ------\n'
)
print(prompt)
