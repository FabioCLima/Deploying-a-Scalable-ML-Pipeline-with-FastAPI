# Diário de Estudo - Deploying a Scalable ML Pipeline with FastAPI

Documento para registrar o progresso e aprendizados durante o desenvolvimento do projeto.

---

## Progresso do Projeto

### 1. Clonei o projeto starter
- Repositório clonado do Udacity
- Projeto base com estrutura de ML pipeline usando FastAPI

### 2. Instalei o ambiente virtual usando uv
- Criado `.venv` com Python 3.12
- Configurado `.python-version` para garantir versão consistente
- Migrado dependências do `requirements.txt` para `pyproject.toml`
- Dependências de produção: fastapi, gunicorn, pandas, scikit-learn, uvicorn
- Dependências de desenvolvimento: pytest, ruff

### 3. Criei o GitHub Actions (CI)
- Arquivo: `.github/workflows/ci.yml`
- Substituído flake8 por **ruff** (mais moderno e rápido)
- Configurado para rodar pytest e ruff

---

## Explicações e Conceitos

### GitHub Actions

GitHub Actions é um sistema de **CI/CD** (Integração Contínua / Entrega Contínua) integrado ao GitHub. Ele executa automaticamente tarefas quando eventos acontecem no repositório.

#### Fluxo de execução

```
┌─────────────────────────────────────────────────────────────┐
│  Você faz push ou abre PR para main                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions detecta o evento (trigger)                  │
│  on: push / pull_request                                    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Inicia uma máquina virtual Ubuntu (runs-on: ubuntu-latest) │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Executa os steps em sequência:                             │
│                                                             │
│  1. Checkout code      → Baixa seu código                   │
│  2. Install uv         → Instala o gerenciador uv           │
│  3. Set up Python      → Instala Python (lê .python-version)│
│  4. Install deps       → uv sync (lê uv.lock)               │
│  5. Ruff check         → Verifica erros de código           │
│  6. Ruff format        → Verifica formatação                │
│  7. Pytest             → Roda os testes                     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Se TODOS passarem → ✅ CI verde                            │
│  Se QUALQUER falhar → ❌ CI vermelho (bloqueia merge)       │
└─────────────────────────────────────────────────────────────┘
```

#### Estrutura do arquivo ci.yml

```yaml
name: CI                    # Nome do workflow

on:                         # QUANDO executar
  push:
    branches: [main]        # Em push para main
  pull_request:
    branches: [main]        # Em PRs para main

jobs:                       # O QUE executar
  test-and-lint:
    runs-on: ubuntu-latest  # ONDE executar

    steps:                  # PASSOS sequenciais
      - name: Checkout      # Cada step tem um nome
        uses: actions/...   # "uses" = action pronta (do marketplace)

      - name: Run tests
        run: uv run pytest  # "run" = comando bash
```

#### Por que usar CI/CD?

1. **Qualidade** - Garante que código quebrado não entre no main
2. **Automação** - Não precisa rodar testes manualmente
3. **Consistência** - Todo mundo segue o mesmo padrão
4. **Feedback rápido** - Sabe imediatamente se algo quebrou

---

## Análise Completa do Projeto

### Estrutura do Projeto

```
Project Root/
├── data/
│   └── census.csv          # Dataset de classificação (30k+ linhas)
├── model/
│   └── .gitignore          # Diretório para salvar modelos treinados
├── ml/
│   ├── __init__.py
│   ├── data.py             # ✅ Processamento de dados (COMPLETO)
│   └── model.py            # ⚠️ Funções ML (PARCIAL - tem TODOs)
├── .github/workflows/
│   └── ci.yml              # ✅ CI/CD Pipeline (COMPLETO)
├── main.py                 # ❌ FastAPI app (TODO)
├── train_model.py          # ❌ Script de treino (TODO)
├── test_ml.py              # ❌ Testes unitários (TODO)
├── local_api.py            # ❌ Cliente API (TODO)
├── model_card_template.md  # ❌ Documentação do modelo (TODO)
├── pyproject.toml          # ✅ Configuração moderna
└── .python-version         # ✅ Python 3.12
```

### Fluxo do ML Pipeline

```
┌────────────────────┐
│ 1. CARREGAR DADOS  │  train_model.py → pd.read_csv("data/census.csv")
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 2. SPLIT DADOS     │  train_test_split() → train, test
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 3. PROCESSAR       │  ml/data.py → process_data()
│    - One-Hot Encode│  Categorical → números
│    - Label Binarize│  salary → 0/1
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 4. TREINAR MODELO  │  ml/model.py → train_model()
│    (RandomForest?) │  Escolher algoritmo ML
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 5. AVALIAR         │  compute_model_metrics()
│    - Precision     │  performance_on_categorical_slice()
│    - Recall        │  → slice_output.txt
│    - F1 Score      │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 6. SALVAR MODELO   │  save_model() → model/model.pkl
└────────┬───────────┘
         ▼
┌────────────────────┐
│ 7. DEPLOY API      │  main.py → FastAPI
│    GET /           │  Mensagem de boas-vindas
│    POST /data/     │  Inferência do modelo
└────────────────────┘
```

### Status de Implementação

| Componente | Arquivo | Status | O que falta |
|------------|---------|--------|-------------|
| Processamento dados | ml/data.py | ✅ Completo | - |
| Métricas | ml/model.py | ✅ Completo | - |
| train_model() | ml/model.py | ❌ TODO | Escolher algoritmo ML |
| inference() | ml/model.py | ❌ TODO | Implementar predição |
| save/load_model() | ml/model.py | ❌ TODO | Serialização pickle |
| slice_performance() | ml/model.py | ❌ TODO | Filtrar por categoria |
| Script treino | train_model.py | ❌ TODO | 6 partes para completar |
| FastAPI app | main.py | ❌ TODO | Endpoints GET/POST |
| Testes | test_ml.py | ❌ TODO | Mínimo 3 testes |
| Cliente API | local_api.py | ❌ TODO | Testar endpoints |
| Model Card | model_card_template.md | ❌ TODO | Documentação |

---

## Próximos Passos (Ordem Recomendada)

### Fase 1: Core ML

- [ ] Implementar `train_model()` em ml/model.py
- [ ] Implementar `inference()` em ml/model.py
- [ ] Implementar `save_model()` e `load_model()` em ml/model.py
- [ ] Implementar `performance_on_categorical_slice()` em ml/model.py

### Fase 2: Pipeline de Treino

- [ ] Completar train_model.py (6 TODOs)
- [ ] Rodar treino e gerar modelo
- [ ] Gerar slice_output.txt

### Fase 3: Testes

- [ ] Escrever 3+ testes em test_ml.py
- [ ] Garantir que CI passa

### Fase 4: API

- [ ] Implementar FastAPI em main.py
- [ ] Implementar GET / e POST /data/
- [ ] Testar com local_api.py

### Fase 5: Documentação

- [ ] Preencher model_card_template.md
- [ ] Screenshots para entrega

---

## Sugestões de Melhoria para Aprendizado

### 1. Pre-commit Hooks (Automatizar qualidade)

**Pre-commit hooks** são scripts que rodam automaticamente **antes** de cada commit. Se o script falhar, o commit é bloqueado.

#### Fluxo sem pre-commit

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ git add .   │ ──▶ │ git commit  │ ──▶ │ Push        │ ──▶ CI falha!
└─────────────┘     └─────────────┘     └─────────────┘
                                         (código com erro
                                          de formatação)
```

#### Fluxo com pre-commit

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ git add .   │ ──▶ │ git commit  │ ──▶ │ HOOK RODA   │ ──▶ │ Commit OK   │
└─────────────┘     └─────────────┘     │ ruff check  │     └─────────────┘
                                        │ ruff format │
                                        └──────┬──────┘
                                               │
                                        Erro? ──▶ Commit BLOQUEADO
                                               │  "Fix antes de commitar"
                                        OK?   ──▶ Commit permitido
```

#### Benefícios

| Sem pre-commit                       | Com pre-commit                     |
| ------------------------------------ | ---------------------------------- |
| Descobre erro no CI (depois do push) | Descobre erro antes do commit      |
| Perde tempo esperando CI             | Feedback instantâneo               |
| Commits com código feio              | Código sempre formatado            |

#### Como configurar

**Passo 1 - Instalar pre-commit:**

```bash
uv add --dev pre-commit
```

**Passo 2 - Criar `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff          # linting
        args: [--fix]
      - id: ruff-format   # formatação
```

**Passo 3 - Ativar os hooks:**

```bash
uv run pre-commit install
```

**Passo 4 - Pronto! Agora todo `git commit` roda ruff automaticamente.**

### 2. Cobertura de Testes

Adicionar pytest-cov para ver % de código testado:

```bash
uv add --dev pytest-cov
uv run pytest --cov=ml --cov-report=html
```

### 3. Type Hints

Adicionar tipagem ao código para melhor documentação:

```python
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    ...
```

### 4. Logging

Substituir prints por logging para produção:

```python
import logging
logging.info("Treinando modelo...")
```

### 5. DVC (Data Version Control)

**DVC** é o "Git para dados". O Git não foi feito para arquivos grandes (datasets, modelos ML), então o DVC resolve isso.

#### O Problema

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Git funciona bem para:          │ Git NÃO funciona para:           │
├─────────────────────────────────┼───────────────────────────────────┤
│ Código (.py, .js, .yaml)        │ Datasets grandes (100MB+)        │
│ Arquivos pequenos               │ Modelos treinados (.pkl, .h5)    │
│ Texto                           │ Imagens/Vídeos para treino       │
└─────────────────────────────────┴───────────────────────────────────┘
```

#### Como o DVC funciona

```text
┌─────────────────────────────────────────────────────────────────────┐
│   Arquivo grande              DVC                    Git            │
│   (census.csv)                                                      │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────┐            ┌─────────────┐       ┌─────────────┐     │
│   │ 3.5 MB  │ ──────────▶│ Storage     │       │ .git        │     │
│   │ dados   │            │ (DAGsHub)   │       │             │     │
│   └─────────┘            └─────────────┘       └─────────────┘     │
│        │                                              ▲             │
│        ▼                                              │             │
│   ┌─────────────┐                                     │             │
│   │census.csv.dvc│ ───────────────────────────────────              │
│   │ (ponteiro)  │    Arquivo pequeno (300 bytes)                   │
│   │ hash: abc123│    vai pro Git!                                  │
│   └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

#### Configuração com DAGsHub (nosso projeto)

**Passo 1 - Instalar DVC:**

```bash
uv add --dev dvc
```

**Passo 2 - Inicializar DVC:**

```bash
dvc init
```

**Passo 3 - Configurar remote do DAGsHub:**

```bash
# Adicionar remote
dvc remote add -d origin https://dagshub.com/FabioCLima/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.dvc

# Configurar autenticação (use seu token do DAGsHub)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user FabioCLima
dvc remote modify origin --local password SEU_TOKEN_DAGSHUB
```

**Passo 4 - Adicionar dataset ao DVC:**

```bash
# Remover do Git (se já estava rastreado)
git rm -r --cached data/census.csv

# Adicionar ao DVC
dvc add data/census.csv

# Commitar ponteiro no Git
git add data/census.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"
```

**Passo 5 - Enviar dados para DAGsHub:**

```bash
dvc push
```

#### Comandos DVC úteis

```bash
# Baixar dados (outro dev ou CI)
dvc pull

# Ver status dos arquivos
dvc status

# Ver arquivos rastreados
dvc list .
```

#### Onde obter o token DAGsHub

1. Acesse: [dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)
2. Clique em **"Generate New Token"**
3. Dê um nome (ex: "dvc-local")
4. Copie e guarde em local seguro

---

## Comandos Úteis

```bash
# Sincronizar dependências
uv sync

# Rodar testes
uv run pytest

# Verificar linting
uv run ruff check .

# Corrigir linting automaticamente
uv run ruff check --fix .

# Verificar formatação
uv run ruff format --check .

# Formatar código
uv run ruff format .
```
