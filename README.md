# Classificação de Células Infectadas por Malária com Deep Learning

Projeto da disciplina **SIN 393 – Introdução à Visão Computacional (2026)**  
Universidade Federal de Viçosa – Campus Rio Paranaíba | Sistemas de Informação  
Prof. João Fernando Mari

**Autores:** Izabela Pereira Duarte e Manuelle Julia Gomes Santana

🎥 **Vídeo de apresentação (YouTube):** <link do vídeo>
📄 **Artigo (IEEE, PDF):** <link ou caminho em `docs/artigo.pdf`>

---

## 1. Descrição do projeto

A malária é diagnosticada tradicionalmente por microscopia de esfregaços sanguíneos, um processo manual, demorado e dependente da experiência do avaliador. Este projeto desenvolve e compara modelos de Deep Learning para classificar automaticamente imagens de células como **parasitadas** (infectadas) ou **não infectadas**.

### Objetivos

- Treinar e avaliar pelo menos dois modelos do `torchvision` (ex.: <Modelo A> e <Modelo B>).
- Aplicar e justificar estratégias de *data augmentation*.
- Investigar a questão experimental: **qual o impacto do transfer learning (pesos pré-treinados no ImageNet vs. treinamento do zero) na classificação de células infectadas por malária, e como isso varia entre arquiteturas e com a quantidade de dados de treino?**
- Avaliar com matriz de confusão, acurácia, precisão, recall e F1-score, além de análise qualitativa dos erros.

## 2. Dataset

- **Nome:** Cell Images for Detecting Malaria
- **Fonte:** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Classes:** `Parasitized` e `Uninfected` (classificação binária)
- **Tamanho:** <nº total de imagens> (<nº por classe>)
- **Observações:** <resolução variável, balanceamento entre classes, etc.>

### Divisão dos dados (hold-out)

| Conjunto   | Proporção | Nº de imagens |
|------------|-----------|---------------|
| Treino     | <70%>     | <...>         |
| Validação  | <15%>     | <...>         |
| Teste      | <15%>     | <...>         |

A divisão foi feita com `train_test_split` (scikit-learn), de forma **estratificada** e com **semente fixa** (`random_state=<42>`). O conjunto de teste é usado apenas na avaliação final.

## 3. Metodologia

### 3.1 Pré-processamento e data augmentation

| Transformação | Parâmetros | Justificativa |
|---------------|------------|---------------|
| Resize        | <224×224>  | <entrada esperada pelos modelos> |
| Normalização  | <média/desvio do ImageNet> | <uso de pesos pré-treinados> |
| <Flip horizontal/vertical> | <p=0.5> | <células não têm orientação fixa> |
| <Rotação>     | <±X°>      | <...> |

### 3.2 Modelos e desenho experimental

Foram treinados dois modelos do `torchvision` (<Modelo A, ex.: ResNet18> e <Modelo B, ex.: EfficientNet-B0>), cada um em duas condições de inicialização, totalizando **4 configurações**:

| | Pré-treinado (fine-tuning) | Do zero |
|---|---|---|
| <Modelo A> | A-PT | A-ZERO |
| <Modelo B> | B-PT | B-ZERO |

Para avaliar o efeito em cenários com poucos dados, cada configuração também foi treinada com frações do conjunto de treino (**<10%, 25% e 100%>**). A fração é amostrada de forma estratificada e com semente fixa; validação e teste permanecem os mesmos em todos os experimentos.

Para isolar o efeito investigado, **todas as configurações usam o mesmo augmentation, os mesmos hiperparâmetros e a mesma divisão dos dados**; muda apenas a inicialização dos pesos (e a arquitetura, na comparação entre modelos). Cada experimento é repetido com <3> sementes, reportando média ± desvio padrão.

| Modelo | Pesos iniciais | Ajustes na última camada | Nº de parâmetros |
|--------|----------------|--------------------------|------------------|
| <Modelo A> | ImageNet / aleatório | <Linear(…, 2)> | <...> |
| <Modelo B> | ImageNet / aleatório | <Linear(…, 2)> | <...> |

### 3.3 Configuração de treinamento

- **Framework:** PyTorch / torchvision
- **Otimizador:** <Adam / SGD> | **Learning rate:** <...>
- **Função de perda:** <CrossEntropyLoss>
- **Batch size:** <...> | **Épocas:** <...>
- **Early stopping / scheduler:** <...>
- **Estratégia de fine-tuning:** <todas as camadas / apenas a última camada primeiro>
- **Sementes:** <42, 43, 44>
- **Hardware:** <Google Colab, GPU T4, etc.>

### 3.4 Avaliação

- Matriz de confusão
- Acurácia, Precisão, Recall e F1-score (problema binário; classe positiva: `Parasitized`)
- Destaque para o **recall da classe `Parasitized`**, pois falsos negativos (célula infectada classificada como saudável) são o erro mais crítico em diagnóstico
- Comparação entre as 4 configurações (média ± desvio padrão) e curva de desempenho em função da fração de dados de treino
- Análise qualitativa: exemplos de acertos e erros, com discussão das possíveis causas

## 4. Resultados

> Preencher após os experimentos.

Resultados no conjunto de teste (100% do treino, média ± desvio):

| Configuração | Acurácia | Precisão | Recall | F1-score |
|--------------|----------|----------|--------|----------|
| A-PT | | | | |
| A-ZERO | | | | |
| B-PT | | | | |
| B-ZERO | | | | |

Desempenho por fração de dados de treino (<F1 ou recall>):

| Configuração | 10% | 25% | 100% |
|--------------|-----|-----|------|
| A-PT | | | |
| A-ZERO | | | |
| B-PT | | | |
| B-ZERO | | | |

Matrizes de confusão, curvas de treino/validação e exemplos de erros estão em `results/`.

## 5. Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── notebooks/        # notebooks de experimentação (Colab)
├── src/              # código-fonte (dataset, treino, avaliação)
├── results/          # figuras, métricas, matrizes de confusão
├── docs/             # artigo (PDF) e materiais complementares
└── data/             # NÃO versionado (ver instruções abaixo)
```

## 6. Como executar

### Requisitos

- Python 3.<10+>
- GPU recomendada (ou Google Colab)
- Dependências em `requirements.txt` (principais: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`)

### Passo a passo

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd <nome-do-repositorio>

# 2. Criar ambiente virtual e instalar dependências
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Baixar o dataset do Kaggle e extrair em data/
#    https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

# 4. Treinar e avaliar
python src/train.py --model <modelo> --epochs <N>
python src/evaluate.py --model <modelo>
```

Alternativamente, abra o notebook `notebooks/<nome>.ipynb` no Google Colab: <badge/link do Colab>.

> Reprodutibilidade: todas as execuções usam a semente `<42>`.

## 7. Referências

- Dataset: <citação do dataset / artigo original, se houver>
- <Referência 2>
- <Referência 3>

## 8. Licença

<MIT, ou a que o grupo escolher>
