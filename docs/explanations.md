
# Machine Learning Pipeline

O que você tem em mãos agora é o "coração" da engenharia de machine learning. O arquivo `train_model.py` é o roteiro que conecta a limpeza de dados, o treinamento e a avaliação em um único processo automatizado.

## O que é um Machine Learning Pipeline?

Um **Machine Learning Pipeline** é uma forma de codificar o fluxo de trabalho de ponta a ponta. Em vez de rodar células de um notebook manualmente, você cria uma sequência de passos automatizados:

1. **Ingestão:** Carregamento dos dados brutos.
2. **Pré-processamento:** Limpeza, tratamento de nulos e conversão de texto para números (usando o `data.py`).
3. **Treinamento:** Criação do modelo matemático (usando o `model.py`).
4. **Avaliação:** Teste do modelo com dados que ele nunca viu para garantir que ele não apenas "decorou" as respostas.
5. **Serialização:** Salvamento do modelo e dos encoders para uso posterior na API.

---

### Implementação do `train_model.py`

Abaixo está o código preenchido seguindo as suas instruções e utilizando as funções dos arquivos que você forneceu:

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# 1. Carregar os dados (Ajuste o project_path conforme necessário)
project_path = os.getcwd() # Pega o diretório atual do projeto
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

# 2. Split dos dados (80% treino, 20% teste)
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Colunas categóricas fornecidas
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# 3. Processar os dados de treino
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Processar os dados de teste (usando o encoder/lb do treino)
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# 4. Treinar o modelo
model = train_model(X_train, y_train)

# 5. Salvar o modelo, o encoder e o binarizador de labels
# Nota: No seu arquivo model.py, save_model aceita (model, path)
model_dir = os.path.join(project_path, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

save_model(model, os.path.join(model_dir, "model.pkl"))
save_model(encoder, os.path.join(model_dir, "encoder.pkl"))
save_model(lb, os.path.join(model_dir, "lb.pkl"))

# 6. Rodar inferência no dataset de teste
preds = inference(model, X_test)

# Calcular e imprimir métricas globais
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Métricas Globais - Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# 7. Performance em fatias (Slices)
# Limpa o arquivo de saída se ele já existir
output_file = "slice_output.txt"
if os.path.exists(output_file):
    os.remove(output_file)

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        
        # Chama a função implementada no model.py
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        
        with open(output_file, "a") as f:
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
            f.write("-" * 30 + "\n")

```

---

### Por que esse script é vital para o seu FastAPI?

* **Consistência de Artefatos:** Ele gera os arquivos `.pkl` (ou `.joblib`). No FastAPI, você não vai treinar nada; você apenas carregará esses arquivos.
* **Segurança (Slicing):** A análise por "fatias" (slices) permite que você veja se o modelo é tendencioso (ex: se ele performa pior para uma determinada raça ou gênero). Isso é fundamental para um deploy ético.
* **Automação:** Agora que você tem esse script, pode colocá-lo no seu **GitHub Action**. Sempre que você mudar o modelo, o Action roda o treino e gera os arquivos novos automaticamente.
