# 🤖 Curso de LangChain para IA Generativa

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![UV Package Manager](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![LangChain Version](https://img.shields.io/badge/langchain-0.3.23%2B-green)](https://python.langchain.com/)

Curso completo de LangChain para Inteligência Artificial Generativa, oferecendo uma abordagem prática para criar aplicações poderosas. Este curso está disponível em duas versões: uma utilizando OpenAI e outra com Ollama, permitindo que você escolha a opção mais adequada às suas necessidades financeiras.

> 🎓 Adaptado do curso original do professor Brandon Hancock ([@aiwithbrandon](https://www.youtube.com/@aiwithbrandon)) para a nova versão do LangChain, incluindo uma implementação alternativa com Ollama.

## 📋 Pré-requisitos

- Python 3.12 ou superior
- UV Package Manager
- Conta na OpenAI (para a versão OpenAI) ou Ollama instalado localmente (para a versão Ollama)

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/matheusfabiao/langchain-course.git
cd langchain-course
```

2. Configure o ambiente Python:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

3. Instale as dependências com UV:
```bash
uv pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas credenciais
```

## 🗂️ Estrutura do Projeto

```
├── OpenAI/                 # Exemplos usando OpenAI
│   ├── 1_chat_model_basic.py
│   ├── 2_chat_model_basic_conversation.py
│   └── 3_chat_model_alternatives.py
├── Ollama/                 # Exemplos usando Ollama
│   └── 1_chat_models/
│       ├── 1_chat_model_basic.py
│       ├── 2_chat_model_basic_conversation.py
│       └── 3_chat_model_alternatives.py
└── pyproject.toml         # Configurações do projeto
```

## 💻 Exemplos de Uso

### Versão OpenAI
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model_name='gpt-4o')
result = model.invoke([
    SystemMessage(content='Solve the following math problems.'),
    HumanMessage(content='What is 81 divided by 9?')
])
```

### Versão Ollama
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOllama(model='qwen2.5:3b')
result = model.invoke([
    SystemMessage(content='Solve the following math problems.'),
    HumanMessage(content='What is 81 divided by 9?')
])
```

## 🛠️ Ferramentas e Dependências

- LangChain (≥0.3.23)
- LangChain OpenAI (≥0.3.12)
- LangChain Ollama (≥0.3.1)
- LangChain Chroma (≥0.2.2)
- LangChain HuggingFace (≥0.1.2)
- Python-dotenv (≥1.1.0)

### Ferramentas de Desenvolvimento

- Blue (≥0.9.1) - Formatador de código
- ISort (≥6.0.1) - Organizador de imports
- Taskipy (≥1.14.1) - Executor de tarefas

## 📝 Tarefas Disponíveis

Utilize o Taskipy para executar tarefas comuns:

```bash
# Formatar o código (Blue + ISort)
task format

# Executar um arquivo específico
task run 1_chat_models/1_chat_model_basic.py
```

## 📚 Recursos Adicionais

- [Documentação do LangChain](https://python.langchain.com/)
- [Canal do Brandon Hancock](https://www.youtube.com/@aiwithbrandon)
- [Documentação do Ollama](https://ollama.ai/)

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

⭐ Se este curso foi útil para você, considere dar uma estrela no repositório!