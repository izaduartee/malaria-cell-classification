# Classificação de células infectadas por malária com deep learning

Projeto da disciplina SIN 393 – Introdução à Visão Computacional (2026)
Universidade Federal de Viçosa, Campus Rio Paranaíba | Sistemas de Informação
Professor: João Fernando Mari

Autoras: Izabela Pereira Duarte e Manuelle Julia Gomes Santana

Vídeo de apresentação (YouTube): link a ser adicionado

Artigo (formato IEEE): `docs/artigo.pdf`, a ser adicionado

---

## 1. Descrição do projeto

O diagnóstico da malária costuma ser feito por microscopia de esfregaços de sangue, um trabalho manual, demorado e que depende da experiência de quem avalia. Este projeto treina redes neurais convolucionais para classificar imagens de células sanguíneas como parasitadas ou não infectadas.

A questão investigada é qual o efeito do transfer learning, ou seja, de começar o treino com pesos pré-treinados na ImageNet em vez de pesos aleatórios. Para responder, comparamos duas arquiteturas do `torchvision`, ResNet-18 e MobileNetV2, nas duas condições de inicialização, e repetimos os treinos com 10%, 25% e 100% do conjunto de treino para ver se o efeito muda quando há menos dados.

### Objetivos

- Treinar e avaliar duas arquiteturas do `torchvision`: ResNet-18 e MobileNetV2.
- Aplicar data augmentation e justificar cada transformação.
- Comparar pesos pré-treinados com treino do zero, em duas arquiteturas e em três quantidades de dados de treino.
- Avaliar com matriz de confusão, acurácia, precisão, recall e F1 no conjunto de teste, e analisar exemplos de erros.

## 2. Dataset

- **Nome:** Cell Images for Detecting Malaria
- **Fonte:** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Classes:** `Parasitized` e `Uninfected` (classificação binária)
- **Tamanho:** 27.558 imagens, 13.779 por classe
- **Observações:** as classes são perfeitamente balanceadas e nenhuma imagem está corrompida. A resolução varia (largura de 46 a 394 px, altura de 40 a 385 px, média de cerca de 133 × 133 px), então as imagens são redimensionadas antes do treino. A pasta `cell_images` aparece duplicada dentro do arquivo baixado e só uma cópia é usada. A página do Kaggle não especifica a licença dos dados.

### Divisão dos dados (hold-out)

| Conjunto  | Proporção | Imagens | Por classe |
|-----------|-----------|---------|------------|
| Treino    | 70%       | 19.290  | 9.645      |
| Validação | 15%       | 4.134   | 2.067      |
| Teste     | 15%       | 4.134   | 2.067      |

A divisão usa `train_test_split` do scikit-learn, estratificada e com `random_state=42`, em duas etapas: 70% para treino e 30% restantes, divididos ao meio entre validação e teste. Os caminhos de cada conjunto estão em `splits/` (`train.csv`, `val.csv`, `test.csv`). O conjunto de teste só é usado na avaliação final, depois da escolha da melhor época pela loss de validação.

## 3. Metodologia

### 3.1 Pré-processamento e data augmentation

Só o conjunto de treino recebe aumento de dados. Validação e teste passam apenas por redimensionamento e normalização.

| Transformação | Parâmetros | Justificativa |
|---------------|------------|---------------|
| Resize | 224 × 224 | Tamanho esperado pelos modelos pré-treinados. |
| Flip horizontal e vertical | p = 0,5 cada | Células não têm orientação fixa na lâmina. |
| Rotação aleatória | até 30° | Mesmo motivo dos flips. Os cantos criados pela rotação ficam pretos, como o fundo das imagens. |
| ColorJitter | brilho 0,2, contraste 0,2, saturação 0,1, matiz 0,02 | Iluminação e coloração variam entre lâminas. O matiz quase não muda porque a cor do parasita é a pista principal. |
| Normalização | média (0,485, 0,456, 0,406) e desvio (0,229, 0,224, 0,225) | Estatísticas da ImageNet, usadas nos pesos pré-treinados. |

Não há recortes aleatórios, porque um recorte poderia retirar o parasita da imagem e deixar o rótulo errado.

### 3.2 Modelos e desenho experimental

| Código | Arquitetura | Pesos iniciais | Taxa de aprendizado |
|--------|-------------|----------------|---------------------|
| A-PT | ResNet-18 | ImageNet | 1e-4 |
| A-ZERO | ResNet-18 | aleatórios | 1e-3 |
| B-PT | MobileNetV2 | ImageNet | 1e-4 |
| B-ZERO | MobileNetV2 | aleatórios | 1e-3 |

A última camada de cada modelo foi trocada por uma camada linear de duas saídas: `Linear(512, 2)` na ResNet-18 (cerca de 11,2 milhões de parâmetros no total) e `Linear(1280, 2)` na MobileNetV2 (cerca de 2,2 milhões). Todas as camadas são treinadas, inclusive nos modelos pré-treinados.

Cada configuração foi treinada com 10%, 25% e 100% do conjunto de treino. As frações são amostradas de forma estratificada, com a mesma semente, e validação e teste são os mesmos em todos os casos.

A taxa de aprendizado muda junto com a inicialização: os modelos pré-treinados usam um valor menor, para ajustar os pesos sem apagar o que já aprenderam, e os treinados do zero usam um valor maior. Por isso a comparação mede duas receitas de treino e não isola o efeito dos pesos. Esse controle não foi executado (ver seção 4.5).

### 3.3 Configuração de treinamento

- Framework: PyTorch e torchvision
- Otimizador: Adam
- Função de perda: `CrossEntropyLoss`
- Tamanho do lote: 64
- Épocas: 10, iguais em todas as configurações
- Seleção do modelo: pesos da época com menor loss de validação
- Sem scheduler e sem early stopping
- Semente: 42 (uma única execução por configuração)
- Hardware: Google Colab com GPU Tesla T4, cerca de 25 minutos por experimento com 100% dos dados

### 3.4 Avaliação

- Matriz de confusão no conjunto de teste
- Acurácia, precisão, recall e F1, com `Parasitized` como classe positiva. O F1 agregado usa a média macro, que coincide com a ponderada porque o teste tem 2.067 imagens de cada classe.
- Atenção ao recall de `Parasitized`: um falso negativo é uma célula infectada classificada como saudável, o erro mais grave em um diagnóstico.
- Teste de McNemar exato entre pares de modelos, que compara as mesmas imagens de teste. Com quatro comparações, o limiar de significância ajustado por Bonferroni é 0,0125.
- Análise qualitativa com exemplos de acertos e erros sorteados com semente fixa.

## 4. Resultados

### 4.1 Desempenho no teste com 100% dos dados de treino

| Configuração | Acurácia | Precisão | Recall | F1 | Falsos negativos | Falsos positivos |
|--------------|----------|----------|--------|----|------------------|------------------|
| A-PT   | 97,27% | 98,27% | 96,23% | 97,24% | 78  | 35 |
| A-ZERO | 96,13% | 98,18% | 94,00% | 96,05% | 124 | 36 |
| B-PT   | 97,05% | 98,55% | 95,50% | 97,00% | 93  | 29 |
| B-ZERO | 96,32% | 97,33% | 95,26% | 96,28% | 98  | 54 |

Precisão, recall e F1 se referem à classe `Parasitized`. Os pesos pré-treinados melhoraram a acurácia nas duas arquiteturas: 1,14 ponto na ResNet-18 e 0,73 na MobileNetV2. Na ResNet-18 o ganho veio principalmente dos falsos negativos, que caíram de 124 para 78. Na MobileNetV2 os falsos negativos quase não mudaram (98 contra 93) e a melhora veio dos falsos positivos (54 contra 29). Nos quatro modelos o recall ficou abaixo da precisão, ou seja, o erro mais comum é deixar passar uma célula infectada.

### 4.2 Efeito da quantidade de dados de treino

Acurácia no teste:

| Configuração | 10% | 25% | 100% |
|--------------|-----|-----|------|
| A-PT   | 96,35% | 96,95% | 97,27% |
| A-ZERO | 94,75% | 95,98% | 96,13% |
| B-PT   | 96,06% | 96,71% | 97,05% |
| B-ZERO | 95,38% | 95,86% | 96,32% |

Os modelos pré-treinados ficam acima dos treinados do zero em todas as frações. A vantagem na ResNet-18 é maior com 10% dos dados (1,60 ponto) do que com 100% (1,14), mas na MobileNetV2 ela fica entre 0,68 e 0,85 ponto em qualquer fração, então o padrão não se repete nas duas arquiteturas. As curvas se achatam depois de 25%: com cerca de 4.800 imagens, o A-PT já chega a 96,95%. A figura está em `results/acuracia_por_fracao.png`, com o eixo vertical cortado.

### 4.3 Teste de McNemar

| Comparação | Só o primeiro acertou | Só o segundo acertou | p-valor |
|------------|-----------------------|----------------------|---------|
| A-PT vs A-ZERO   | 82 | 35 | menor que 0,0001 |
| B-PT vs B-ZERO   | 73 | 43 | 0,0068 |
| A-PT vs B-PT     | 41 | 32 | 0,35 |
| A-ZERO vs B-ZERO | 29 | 37 | 0,39 |

As duas diferenças entre pré-treinado e do zero ficam abaixo de 0,0125. Entre as arquiteturas, com o mesmo tipo de inicialização, não há diferença detectável. O teste mede apenas a variação causada pela escolha das imagens de teste, não a variação de treinar de novo com outra semente.

### 4.4 Análise dos erros

`results/exemplos_erros.png` mostra quatro falsos negativos e quatro falsos positivos do A-PT, sorteados com semente fixa, além de quatro acertos. Nos falsos negativos aparecem parasitas pequenos na borda da célula e células escuras ou azuladas em que não se distingue uma mancha. Nos falsos positivos aparecem regiões mais escuras de coloração e uma célula fragmentada. Essas são hipóteses a partir de poucos exemplos, e parte dos casos extremos pode ser erro de rótulo no dataset. A maioria dos erros tem probabilidade prevista entre 0,35 e 0,75, ou seja, são casos limítrofes.

### 4.5 Limitações

- Há uma única semente por configuração. Duas execuções sem semente fixa do mesmo experimento (guardadas no notebook como primeira execução) mostraram variações de cerca de 0,5 ponto na acurácia de validação.
- A taxa de aprendizado difere entre pré-treinados e treinados do zero, e o controle planejado (ResNet-18 com as duas taxas cruzadas) não foi executado. Não dá para atribuir a diferença só aos pesos.
- Todos os modelos treinam por 10 épocas, então os que usam 10% dos dados recebem bem menos atualizações de pesos. Parte da queda com poucos dados pode vir do treino curto.
- O dataset não traz identificador de paciente. Células da mesma lâmina podem cair em treino e teste, o que tende a inflar os resultados.

Os arquivos gerados estão em `results/`: `metricas_teste.csv`, `mcnemar.csv`, `matrizes_confusao.png`, `acuracia_por_fracao.png`, `exemplos_erros.png`, além dos históricos de treino (`history_*.csv`) e das previsões no teste (`test_preds_*.csv`) de cada experimento.

## 5. Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── notebooks/   # notebook com EDA, divisão dos dados, treino e análises (Colab)
├── splits/      # train.csv, val.csv e test.csv
├── results/     # métricas, figuras, históricos e previsões
├── docs/        # artigo em PDF
└── data/        # não versionada, criada ao baixar o dataset
```

Os pesos treinados (`checkpoints/`) não são versionados.

## 6. Como executar

O código está em um notebook, `notebooks/malaria-cell-classification.ipynb`, pensado para o Google Colab com GPU. O notebook com todos os experimentos leva algumas horas em uma T4.

Abrir no Colab: https://colab.research.google.com/github/izaduartee/malaria-cell-classification/blob/main/notebooks/malaria-cell-classification.ipynb

### No Colab

1. Ative a GPU em Ambiente de execução, Alterar tipo de ambiente de execução, T4 GPU.
2. Crie um token de API no Kaggle (Settings, API) e salve em Secrets do Colab com o nome `KAGGLE_API_TOKEN`, com acesso ao notebook ativado.
3. Rode as células em ordem. O notebook baixa o dataset, refaz a divisão com a semente 42 (que reproduz os arquivos de `splits/`), treina os modelos e gera as figuras e tabelas em `results/`.

### Localmente

```bash
git clone https://github.com/izaduartee/malaria-cell-classification.git
cd malaria-cell-classification
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# dataset (requer token do Kaggle em KAGGLE_API_TOKEN)
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
unzip -q cell-images-for-detecting-malaria.zip -d data

jupyter notebook notebooks/malaria-cell-classification.ipynb
```

O notebook espera as pastas `data/cell_images/Parasitized` e `data/cell_images/Uninfected`. A instalação do PyTorch com GPU depende da versão do CUDA; instruções em https://pytorch.org/get-started/locally/.

### Reprodutibilidade

A divisão dos dados, as frações de treino e os experimentos principais usam a semente 42. O notebook mantém uma primeira execução dos quatro experimentos, sem semente fixa e sem avaliação no teste, apenas como registro. Os resultados deste README vêm da execução com semente fixa.

## 7. Referências

- Rajaraman, S. et al. Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. PeerJ, 2018.
- He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. CVPR, 2016.
- Sandler, M. et al. MobileNetV2: inverted residuals and linear bottlenecks. CVPR, 2018.
- Deng, J. et al. ImageNet: a large-scale hierarchical image database. CVPR, 2009.
- Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. ICLR, 2015.
- Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS, 2019.
- Pedregosa, F. et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 2011.
- McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 1947.
