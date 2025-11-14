# Demonstração de DVC + CI/CD + MLFlow

## DVC
### Versionamento de dados e pipeline

#### Neste respotiório será abordada a utilização de uma pasta local como storage de arquivos do DVC, de modo a contornar o problema do google drive e facilitar o entendimento.

Clone este repositório em uma pasta fora do one drive para não ter erros durante a atualização de cache do dvc.
```
git clone git@github.com:mateusbcavalcanti/demonstrando-dvc.git
```

Crie a pasta onde o dvc armazenará os dados em qualquer local da sua máquina, por ex em documentos:
```
mkdir {seu-caminho-ate-documentos}/storage dvc teste
```

Defina a sua nova pasta de teste como o remote do dvc:
```
#entre na pasta raiz do projeto
dvc remote add -d local_remote {seu-caminho-ate-documentos}/storage dvc teste
```
Como é a primeira vez que estamos executando o código, os conjuntos de treinamento e de validação ainda não estão na pasta data.
Por isso, é necessário rodar o script 'preprocess.py', então execute-o.

```
python src/preprocess.py
```

Agora que os dados foram gerados, vamos adicioná-los ao controle do DVC:
```
dvc add data/train
dvc add data/val
git add data/val.dvc data/train.dvc
```

Isso cria um arquivo .dvc que referencia o dataset. Agora que temos tudo, podemos dar um push que irá armazenar os arquivos na nossa pasta de storage teste.:
```
dvc push
```
Para testar se está funcionando mesmo, após o push, apague os datasets de treino e validacao e dê um dvc pull na pasta raiz.
Se os arquivos voltarem, deu certo! :)
```
dvc pull
```
Então para salvar isso no git hub, faça um commit.
```
git add .
git commit -m "Fazendo configuração do dvc"
git push origin main
```
### Pipeline DVC
Arquivo `dvc.yaml`:
```
stages:
  # preprocess:
  #   cmd: pip install mlflow
  train:
    cmd: python src/train.py data/train/ data/val/ models/model.pkl
    deps:
      - src/train.py
      - data/train/
      - data/val/
    outs:
      - models/model.pkl
    metrics:
      - metrics/metrics.json
      # - metrics/plots/roc_curve.png

  # evaluate:
  #   cmd: python src/evaluate.py
  #   deps:
  #     - src/evaluate.py
  #     - metrics/metrics.json

```

Execute esse comando para que o dvc possa ser o responsável por rodar todo o código, como é definido na pipeline:
```
dvc repro
```

## Treino Automático no GitHub Actions
Arquivo `.github/workflows/github-ci.yml`:
```
name: CI/CD - MLOps Demo

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install dvc[gs] mlflow

      - name: Reproduce DVC pipeline
        run: dvc repro

      - name: Push updated metrics
        run: |
          dvc push
          git add metrics/ dvc.lock
          git commit -m "Update metrics after pipeline run" || echo "No changes"
          git push
```
Isso irá garantir que toda vez que você fizer `push`, o GitHub:
- Instala o ambiente
- Executa o pipeline DVC
- Atualiza métricas automaticamente


## MLFlow - Rastreamento de Experimentos
No script `train.py`, adicione o tracking, e.g.: 
```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("demo-mlops")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

```

Uma vez adicionado, execute: 
```
mlflow ui
```

