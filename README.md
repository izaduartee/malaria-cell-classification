# Classificação de células infectadas por malária com deep learning

Projeto da disciplina SIN 393 – Introdução à Visão Computacional (2026)
Universidade Federal de Viçosa, Campus Rio Paranaíba | Sistemas de Informação
Professor: João Fernando Mari

Autoras: Izabela Pereira Duarte e Manuelle Julia Gomes Santana

Vídeo de apresentação (YouTube): https://youtu.be/O_TA-EIz-wQ

Artigo (formato IEEE): [docs/artigo.pdf](docs/artigo.pdf)

---

## 1. Descrição do projeto

O diagnóstico da malária costuma ser feito por microscopia de esfregaços de sangue, um trabalho manual, demorado e que depende da experiência de quem avalia. Este projeto treina redes neurais convolucionais para classificar imagens de células sanguíneas como parasitadas ou não infectadas.

A questão investigada é qual o efeito do transfer learning, ou seja, de começar o treino com pesos pré-treinados na ImageNet em vez de pesos aleatórios. Para responder, comparamos duas arquiteturas do `torchvision`, ResNet-18 e MobileNetV2, nas duas condições de inicialização, e repetimos os treinos com 10%, 25% e 100% do conjunto de treino para ver se o efeito muda quando há menos dados. A divisão principal dos dados é feita por paciente, para que células de um mesmo paciente não apareçam em treino e teste ao mesmo tempo.

### Objetivos

- Treinar e avaliar duas arquiteturas do `torchvision`: ResNet-18 e MobileNetV2.
- Aplicar data augmentation e justificar cada transformação.
- Comparar pesos pré-treinados com treino do zero, em duas arquiteturas e em três quantidades de dados de treino.
- Avaliar com matriz de confusão, acurácia, precisão, recall e F1 no conjunto de teste, testar as diferenças com o teste de McNemar e analisar exemplos de erros.
- Comparar a divisão por paciente com uma divisão aleatória por imagem.

## 2. Dataset

- **Nome:** Cell Images for Detecting Malaria
- **Fonte:** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Classes:** `Parasitized` e `Uninfected` (classificação binária)
- **Tamanho:** 27.558 imagens, 13.779 por classe
- **Pacientes:** 200, sendo 150 infectados por *Plasmodium falciparum* e 50 saudáveis, segundo o artigo que originou o dataset. O código do paciente está no início do nome de cada arquivo (por exemplo `C100P61...`) e é usado para agrupar as imagens.
- **Observações:** as classes são perfeitamente balanceadas no dataset completo e nenhuma imagem está corrompida. A resolução varia (largura de 46 a 394 px, altura de 40 a 385 px, média de cerca de 133 × 133 px), então as imagens são redimensionadas antes do treino. A pasta `cell_images` aparece duplicada dentro do arquivo baixado e só uma cópia é usada. A página do Kaggle não especifica a licença dos dados.

### Divisão dos dados (principal): por paciente

A divisão usa `GroupShuffleSplit` do scikit-learn com `random_state=42`, em duas etapas: 70% dos pacientes para treino e 30% restantes, divididos ao meio entre validação e teste. Nenhum paciente aparece em mais de um conjunto.

| Conjunto  | Pacientes | Imagens | Parasitized | Uninfected |
|-----------|-----------|---------|-------------|------------|
| Treino    | 140       | 20.633  | 10.965      | 9.668      |
| Validação | 30        | 3.605   | 1.547       | 2.058      |
| Teste     | 30        | 3.320   | 1.267       | 2.053      |

Como o número de imagens varia entre pacientes e a divisão não é estratificada por classe, a proporção de células parasitadas é de cerca de 53% no treino, 43% na validação e 38% no teste. Os arquivos estão em `splits_patient/` (`train.csv`, `val.csv`, `test.csv`). O conjunto de teste só é usado na avaliação final, depois da escolha da melhor época pela loss de validação.

### Divisão para comparação: por imagem

Os mesmos experimentos foram executados antes com uma divisão aleatória por imagem (`train_test_split` estratificado, `random_state=42`, 70%, 15% e 15%: 19.290, 4.134 e 4.134 imagens). Nessa divisão os 200 pacientes aparecem nos três conjuntos. Os arquivos estão em `splits/`.

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

A última camada de cada modelo foi trocada por uma camada linear de duas saídas: `Linear(512, 2)` na ResNet-18 e `Linear(1280, 2)` na MobileNetV2. Todas as camadas são treinadas, inclusive nos modelos pré-treinados.

Cada configuração foi treinada com 10%, 25% e 100% do conjunto de treino. As frações (cerca de 2.063, 5.158 e 20.633 imagens) são amostradas de forma estratificada, com a mesma semente, e validação e teste são os mesmos em todos os casos.

A taxa de aprendizado muda junto com a inicialização: os modelos pré-treinados usam um valor menor, para ajustar os pesos sem apagar o que já aprenderam, e os treinados do zero usam um valor maior. Por isso a comparação mede duas receitas de treino e não isola o efeito dos pesos. Esse controle não foi executado.

### 3.3 Configuração de treinamento

- Framework: PyTorch e torchvision
- Otimizador: Adam
- Função de perda: `CrossEntropyLoss`
- Tamanho do lote: 64
- Épocas: 10, iguais em todas as configurações
- Seleção do modelo: pesos da época com menor loss de validação
- Sem scheduler e sem early stopping
- Semente: 42 (uma única execução por configuração)
- Hardware: Google Colab com GPU Tesla T4, cerca de 28 minutos por experimento com 100% dos dados

### 3.4 Avaliação

- Matriz de confusão no conjunto de teste.
- Acurácia, precisão, recall e F1, com `Parasitized` como classe positiva. As três últimas se referem a essa classe, sem agregação entre classes.
- Atenção ao recall: um falso negativo é uma célula infectada classificada como saudável, o erro mais grave em um diagnóstico.
- Teste de McNemar exato entre pares de modelos, que compara as mesmas imagens de teste. Com quatro comparações, o limiar de significância ajustado por Bonferroni é 0,0125.
- Análise qualitativa com exemplos de acertos e erros sorteados com semente fixa.

## 4. Resultados

Todos os valores desta seção vêm da divisão por paciente, exceto a tabela 4.5.

### 4.1 Desempenho no teste com 100% dos dados de treino

| Configuração | Acurácia | Precisão | Recall | F1 | Falsos negativos | Falsos positivos |
|--------------|----------|----------|--------|----|------------------|------------------|
| A-PT   | 97,20% | 96,00% | 96,69% | 96,34% | 42 | 51 |
| A-ZERO | 96,23% | 96,57% | 93,45% | 94,99% | 83 | 42 |
| B-PT   | 96,81% | 97,08% | 94,48% | 95,76% | 70 | 36 |
| B-ZERO | 96,20% | 96,72% | 93,21% | 94,94% | 86 | 40 |

O pré-treino aumentou a acurácia nas duas arquiteturas (0,97 ponto na ResNet-18 e 0,61 na MobileNetV2) e reduziu os falsos negativos: de 83 para 42 na ResNet-18 e de 86 para 70 na MobileNetV2. Na ResNet-18, isso veio com um pequeno aumento de falsos positivos (de 42 para 51). Em três dos quatro modelos o recall ficou abaixo da precisão, ou seja, o erro mais comum é deixar passar uma célula infectada. A exceção foi o A-PT.

### 4.2 Teste de McNemar

| Comparação | Só o primeiro acertou | Só o segundo acertou | p-valor |
|------------|-----------------------|----------------------|---------|
| A-PT vs A-ZERO   | 71 | 39 | 0,0029 |
| B-PT vs B-ZERO   | 52 | 32 | 0,0375 |
| A-PT vs B-PT     | 43 | 30 | 0,1597 |
| A-ZERO vs B-ZERO | 26 | 25 | 1,0000 |

Na ResNet-18, a diferença entre pré-treinado e do zero é significativa com o limiar corrigido de 0,0125. Na MobileNetV2, o valor 0,0375 só seria significativo ao nível de 0,05 sem a correção. Entre as arquiteturas, com a mesma inicialização, não há diferença detectável, o que não equivale a mostrar que são iguais. O teste mede apenas a variação causada pela escolha das imagens de teste, não a variação de treinar de novo com outra semente.

### 4.3 Efeito da quantidade de dados de treino

Acurácia no teste:

| Configuração | 10% | 25% | 100% |
|--------------|-----|-----|------|
| A-PT   | 96,45% | 96,36% | 97,20% |
| A-ZERO | 95,54% | 95,96% | 96,23% |
| B-PT   | 96,57% | 97,17% | 96,81% |
| B-ZERO | 95,51% | 95,87% | 96,20% |

Com 10% dos dados, os modelos pré-treinados ficam no nível dos treinados do zero com 100%, ou um pouco acima. Essa diferença, de 0,2 a 0,4 ponto, é menor do que a incerteza de um teste de 3.320 imagens de apenas 30 pacientes. A vantagem do pré-treino varia entre 0,4 e 1,3 ponto, sem tendência clara com a quantidade de dados, e as curvas não são monótonas: o B-PT passa de 97,17% com 25% dos dados para 96,81% com 100%. Como as 10 épocas são fixas, os modelos com menos dados recebem menos atualizações de pesos.

### 4.4 Análise dos erros

`results_patient/exemplos_erros.png` mostra quatro falsos negativos, quatro falsos positivos e quatro acertos do A-PT, sorteados com semente fixa. Nos falsos negativos aparecem células com coloração tênue ou com o parasita perto da borda, e nos falsos positivos aparecem artefatos de coloração e detritos. São hipóteses a partir de poucos exemplos, e parte dos casos pode ser erro de rótulo no dataset.

### 4.5 Divisão por imagem e por paciente

Acurácia no teste nas duas divisões:

| Configuração | Por imagem | Por paciente | Diferença |
|--------------|------------|--------------|-----------|
| A-PT   | 97,27% | 97,20% | -0,07 |
| A-ZERO | 96,13% | 96,23% | +0,10 |
| B-PT   | 97,05% | 96,81% | -0,24 |
| B-ZERO | 96,32% | 96,20% | -0,12 |

A acurácia mudou no máximo 0,24 ponto, e em um dos quatro modelos subiu. Os conjuntos de teste são diferentes (4.134 e 3.320 imagens, com outros pacientes e outra proporção de classes), então a comparação é aproximada, e as contagens de falsos negativos não são comparadas entre as duas divisões. Nestes dados, separar por paciente não reduziu a acurácia de forma detectável, mas isso não permite concluir que o vazamento entre pacientes seja irrelevante em geral.

### 4.6 Limitações

- Há uma única semente por configuração. Execuções sem semente fixa, feitas durante o desenvolvimento, variaram cerca de 0,5 ponto na acurácia de validação.
- A taxa de aprendizado difere entre pré-treinados e treinados do zero, e o controle (ResNet-18 com as duas taxas cruzadas) não foi executado. Não dá para atribuir a diferença só aos pesos.
- Todos os modelos treinam por 10 épocas, então os que usam 10% dos dados recebem bem menos atualizações de pesos.
- O teste reúne 30 pacientes, com 38% de células parasitadas contra 53% no treino, e foi usada uma única divisão por paciente. O resultado depende de quais pacientes caíram no teste.

Os arquivos gerados estão em `results_patient/` (divisão por paciente) e em `results/` (divisão por imagem): `metricas_teste.csv`, `mcnemar.csv`, `matrizes_confusao.png`, `acuracia_por_fracao.png`, `exemplos_erros.png`, `curvas_aprendizado.png`, `ranking_acuracia_final.png`, além dos históricos de treino (`history_*.csv`) e das previsões no teste (`test_preds_*.csv`) de cada experimento.

## 5. Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── notebooks/        # notebook com EDA, divisão por paciente, treino e análises (Colab)
├── splits_patient/   # divisão principal, por paciente
├── results_patient/  # resultados da divisão por paciente
├── splits/           # divisão aleatória por imagem, para comparação
├── results/          # resultados da divisão por imagem
├── docs/             # artigo em PDF
└── data/             # não versionada, criada ao baixar o dataset
```

Os pesos treinados (`checkpoints/`) não são versionados.

## 6. Como executar

O código está em um notebook, `notebooks/malaria-cell-classification.ipynb`, escrito para o Google Colab com GPU. Rodar todos os experimentos (quatro modelos com 100% dos dados e as frações de 10% e 25%) leva cerca de 3 horas em uma T4.

Abrir no Colab: https://colab.research.google.com/github/izaduartee/malaria-cell-classification/blob/main/notebooks/malaria-cell-classification.ipynb

### No Colab

1. Ative a GPU em Ambiente de execução, Alterar tipo de ambiente de execução, T4 GPU.
2. Crie um token de API no Kaggle (Settings, API) e salve em Secrets do Colab com o nome `KAGGLE_API_TOKEN`, com acesso ao notebook ativado.
3. Rode as células em ordem. O notebook monta o Google Drive, baixa o dataset, faz a divisão por paciente com a semente 42, treina os modelos e grava resultados e pesos na pasta `Experimentos_Divisao_Paciente` do Drive. Se a sessão cair, rodar a célula de treino de novo retoma do ponto em que parou, porque cada experimento concluído e cada época ficam salvos.

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

O notebook espera as pastas `data/cell_images/Parasitized` e `data/cell_images/Uninfected`. As células que montam o Google Drive e leem o token pelos Secrets do Colab precisam ser trocadas por caminhos e variáveis locais. A instalação do PyTorch com GPU depende da versão do CUDA; instruções em https://pytorch.org/get-started/locally/.

### Reprodutibilidade

A divisão por paciente, as frações de treino e os experimentos usam a semente 42. A divisão por imagem, usada só na comparação, foi feita com `train_test_split` e a mesma semente em uma versão anterior do notebook, e seus arquivos estão em `splits/` e `results/`.

## 7. Referências

1. World Health Organization. World malaria report 2023. Geneva: WHO, 2023.
2. Rajaraman, S. et al. Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. PeerJ, v. 6, e4568, 2018.
3. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. CVPR, 2016.
4. Sandler, M. et al. MobileNetV2: inverted residuals and linear bottlenecks. CVPR, 2018.
5. Muhandisin, A.; Azhar, Y. Malaria blood cell image classification using transfer learning with fine-tune ResNet50 and data augmentation. Jurnal RESTI, v. 6, n. 5, p. 891-897, 2022.
6. Muhammad, V. et al. Malaria parasite cell classification using transfer learning with state-of-the-art CNN architectures. Biology, v. 14, n. 12, p. 1792, 2025.
7. Pamungkas, Y. et al. A systematic review of transfer learning approaches for malaria diagnosis using red blood cell imaging. International Journal of Robotics and Control Systems, v. 5, n. 4, p. 2197-2217, 2025.
8. Pedregosa, F. et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.
9. Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS, 2019.
10. Deng, J. et al. ImageNet: a large-scale hierarchical image database. CVPR, 2009.
11. Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. ICLR, 2015.
12. McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, v. 12, n. 2, p. 153-157, 1947.
