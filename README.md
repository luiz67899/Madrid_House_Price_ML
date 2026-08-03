# Madrid_House_Price_ML
Modelo de Machine Learning para previsão de preços de imóveis em Madrid. Este projeto aplica técnicas de regressão supervisionada para prever o preço de compra (`buy_price`) de imóveis em Madrid utilizando o dataset [Madrid Real Estate Market](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market) do Kaggle.

## 🚀 Como Executar

### Pré-requisitos

- Python 3.9+
- pip ou conda

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/madrid-house-price-prediction.git
cd madrid-house-price-prediction

# Crie um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instale as dependências
pip install -r requirements.txt
Download do Dataset
Acesse Madrid Real Estate Market - Kaggle
Baixe o arquivo houses_Madrid.csv
Coloque na pasta data/

Pipeline de ML
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌───────────┐
│  1. Load    │────▶│  2. Split   │────▶│  3. Preprocess  │────▶│  4. Train │
│    Data     │     │ Train/Test  │     │  (train only!)  │     │   Model   │
└─────────────┘     └─────────────┘     └─────────────────┘     └───────────┘
                                                                       │
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐          │
│  7. Deploy  │◀────│  6. Tune    │◀────│  5. Evaluate    │◀─────────┘
│  (future)   │     │   Hyper.    │     │ Train vs Test   │
└─────────────┘     └─────────────┘     └─────────────────┘
🔑 Conceitos-Chave Aplicados
Data Leakage Prevention
Toda transformação (remoção de outliers, normalização) é calculada apenas no conjunto de treino e aplicada ao teste. Isso simula o cenário real onde não temos acesso a dados futuros.

Feature Selection Multi-método
Table



Método


O que captura


Robustez a outliers


Pearson	Relação linear	❌ Baixa
Spearman	Relação monotônica	✅ Alta
Mutual Information	Qualquer dependência	✅ Alta
View more
Bias-Variance Tradeoff
Erro treino BAIXO + Erro teste BAIXO  → ✅ Bom modelo
Erro treino BAIXO + Erro teste ALTO   → ⚠️ Overfitting
Erro treino ALTO  + Erro teste ALTO   → ⚠️ Underfitting
🛠️ Tecnologias
Python 3.9+
Pandas / NumPy — Manipulação de dados
Scikit-learn — Pipeline, modelos, métricas, cross-validation
XGBoost — Gradient Boosting otimizado
Jupytext — Versionamento de notebooks como .py
📈 Próximos Passos
 Incluir features categóricas (neighborhood_id, house_type_id) com Target Encoding
 Aplicar log-transform no target para reduzir assimetria
 Testar LightGBM e CatBoost
 Feature engineering (preço/m² por bairro, distância ao centro)
 Implementar Stacking Ensemble
 Deploy com FastAPI (endpoint de predição)
📝 Aprendizados
Documentação detalhada da teoria por trás de cada decisão está disponível em docs/theory_notes.md.

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

🤝 Contato
Luiz Augusto

Projeto desenvolvido como estudo prático de Machine Learning aplicado ao mercado imobiliário.


```text

# Madrid House Price Prediction - Dependencies
# Python 3.9+

# Data manipulation
pandas>=2.0.0
numpy>=1.24.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Notebook support
jupytext>=1.15.0
ipykernel>=6.25.0

# Optional: Advanced tuning
# optuna>=3.3.0

.gitignore






# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data (download from Kaggle)
data/*.csv
data/*.zip

# Models (too large for git)
models/*.pkl
models/*.joblib

# Logs
*.log
docs/theory_notes.md - Notas Teóricas






# 📚 Notas Teóricas — Madrid House Price Prediction

Documentação detalhada da teoria por trás de cada decisão tomada no notebook.

---

## 1. Carregamento e Exploração dos Dados

### O que é?
Leitura do dataset bruto para um DataFrame (estrutura tabular em memória).

### Conceitos fundamentais
- **Amostra (sample):** Cada linha = um imóvel
- **Feature (atributo):** Cada coluna = uma característica mensurável
- **Target (rótulo):** A variável que queremos prever (`buy_price`)
- **Skewness (assimetria):** Mede se a distribuição é simétrica. Skewness > 1 indica cauda à direita (muitos imóveis baratos, poucos muito caros)

### Por que explorar primeiro?
Entender dimensionalidade, tipos de dados e distribuição do target antes de modelar evita surpresas nas etapas seguintes (ex: descobrir que 50% dos dados estão vazios DEPOIS de treinar o modelo).

---

## 2. Avaliação de Dados Faltantes

### Mecanismos de dados faltantes (Rubin, 1976)

| Mecanismo | Definição | Exemplo |
|-----------|-----------|---------|
| **MCAR** | Falta completamente aleatória | Sensor falhou aleatoriamente |
| **MAR** | Falta depende de outra variável observada | Renda falta mais para jovens |
| **MNAR** | Falta depende do próprio valor ausente | Ricos não declaram patrimônio |

### Thresholds da literatura

| % Faltante | Estratégia | Justificativa |
|------------|------------|---------------|
| < 5% | Descartar amostras | Perda mínima de informação |
| 5-20% | Imputação (mediana) | Muitos dados para desperdiçar; mediana é robusta a outliers |
| 20-50% | Imputação avançada (KNN/MICE) | Imputação simples distorceria a distribuição |
| > 50% | Descartar a feature | Mais de metade seria "inventada" |

### Por que mediana > média para imputação?
A média é sensível a outliers. Se existem imóveis de €10M no dataset, a média fica inflacionada e não representa o valor "típico". A mediana é o valor central (50º percentil), imune a extremos.

---

## 3. Seleção de Features

### Método 1: Correlação de Pearson
r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / [√Σ(xᵢ - x̄)² × √Σ(yᵢ - ȳ)²]


- **Mede:** Relação LINEAR entre duas variáveis
- **Range:** [-1, +1]
- **Limitação:** Sensível a outliers (usa média e desvio padrão); não captura relações não-lineares

### Método 2: Correlação de Spearman

- **Como funciona:** Converte valores em ranks (posições) e calcula Pearson sobre os ranks
- **Mede:** Relação MONOTÔNICA (crescente ou decrescente, mas não necessariamente linear)
- **Vantagem:** Robusto a outliers (um valor extremo tem rank = último, sem distorcer)

### Método 3: Mutual Information

**Fórmula (Teoria da Informação de Shannon):**
MI(X, Y) = Σ p(x,y) × log[p(x,y) / (p(x) × p(y))]


- **Mede:** QUALQUER dependência estatística (linear, quadrática, periódica, etc.)
- **Range:** [0, ∞) — 0 = independência total
- **Quando usar:** Quando suspeita de relações complexas/não-lineares

### Por que usar 3 métodos?
Cross-referência aumenta a confiança. Se uma feature é top 5 em TODOS os métodos, a relação é robusta.

---

## 4. Multicolinearidade

### O que é?
Duas features carregam informação redundante (alta correlação ENTRE si, não com o target).

### Impacto por tipo de modelo

| Modelo | Impacto da multicolinearidade |
|--------|-------------------------------|
| Linear Regression | ❌ Severo — coeficientes instáveis |
| Ridge/Lasso | ✅ Lida via regularização |
| Random Forest | ✅ Impacto mínimo |
| XGBoost | ✅ Impacto mínimo |

### No nosso caso
`sq_mt_built` e `n_bathrooms` correlação = 0.84. Mantemos ambas porque usamos modelos de árvore (imunes) e Ridge/Lasso (que regularizam).

---

## 5. Train/Test Split

### Por que separar?
O objetivo do ML é **generalizar** para dados nunca vistos. O teste simula "produção".

### Por que separar ANTES de transformações?

**Data Leakage:** Quando informação do teste influencia decisões do treino.

**Analogia:** Professor que espia as respostas dos alunos para montar o gabarito — o gabarito fica enviesado.

**Exemplo concreto:**
```python
# ERRADO: percentil calculado com TODOS os dados (incluindo futuros dados de teste)
q95 = df['buy_price'].quantile(0.95)  # usa teste indiretamente

# CORRETO: percentil calculado SÓ no treino
q99 = y_train.quantile(0.99)  # apenas dados "conhecidos"
Proporção 80/20
Dataset grande (>10k) → 80/20 suficiente
Dataset pequeno (<1k) → Preferir cross-validation
Dataset muito grande (>100k) → 95/5 funciona
6. Remoção de Outliers
O que são outliers?
Pontos muito distantes da distribuição típica. Podem ser erros de coleta ou valores reais mas raros.

Por que afetam modelos lineares?
MSE usa o QUADRADO do erro. Um outlier com erro de €1M gera penalidade de 10¹², "puxando" toda a reta para si.

Método do percentil (Q01/Q99)
Remove os 1% mais extremos de cada lado. Conservador — preserva 98% dos dados.

Por que SÓ no treino?
Teste simula produção → não podemos controlar dados futuros
Métrica honesta → se o modelo falha para casas caras, precisamos saber
Evita data leakage → threshold calculado apenas com dados de treino
Alternativas
Table



Método


Como funciona


IQR	Outlier = fora de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
Z-Score	Outlier = |z| > 3
Isolation Forest	ML detecta anomalias automaticamente
View more
7. StandardScaler (Normalização Z-score)
Fórmula
z = (x - μ_treino) / σ_treino
Por que normalizar?
Features em escalas diferentes (sq_mt_built: 13-999; n_rooms: 0-24)
Sem normalizar: gradiente descendente converge lento; coeficientes ficam desproporcionais
Com normalizar: superfície de custo fica "circular" → convergência rápida
Por que valores negativos?
Valores abaixo da média produzem z < 0. Ex: casa de 60m² com média de 119m² → z = (60-119)/60 = -0.98

Pipeline garante:
fit_transform() no treino → aprende μ e σ do treino
transform() no teste → aplica μ e σ do TREINO (não recalcula)
8. Cross-Validation (K-Fold)
O que é?
Treinar e avaliar K vezes, rotacionando quem é validação.

Por que não basta um split?
Um único split pode ser "sortudo" ou "azarado". Cross-validation fornece:

Média → Melhor estimativa real
Desvio padrão → Estabilidade do modelo
Como funciona (K=5)
Cada amostra aparece exatamente 1 vez como validação. Resultado: 5 métricas → média ± std.

9. Modelos — Teoria
DummyRegressor (Baseline)
Sempre prevê a média. R² = 0 por definição. Se seu modelo não bate o Dummy, é inútil.

Linear Regression
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃
Minimiza MSE. Solução: β = (XᵀX)⁻¹Xᵀy. Assume relação linear.

Ridge (L2 Regularization)
Custo = MSE + λ × Σβⱼ²
Penaliza coeficientes grandes → estabiliza com multicolinearidade. Nunca zera coeficientes.

Lasso (L1 Regularization)
Custo = MSE + λ × Σ|βⱼ|
Pode zerar coeficientes → seleção automática de features.

ElasticNet (L1 + L2)
Combina ambas. Útil com features correlacionadas + desejo de seleção.

Random Forest
Treina N árvores em paralelo com Bagging (amostras bootstrap)
Cada split considera subconjunto aleatório de features
Predição = média das N árvores
Reduz variância sem aumentar viés
Gradient Boosting
Treina árvores sequencialmente, cada uma corrigindo erros da anterior
Cada árvore prevê os RESÍDUOS (erros) do modelo atual
Atualização: F_novo = F_anterior + η × árvore_nova
Reduz viés iterativamente
XGBoost
Gradient Boosting otimizado com:

Regularização L1/L2 nos pesos das folhas
Aproximação de segunda ordem (Hessiana)
Column subsampling (como RF)
Tratamento nativo de valores faltantes
10. Métricas de Avaliação
R² (Coeficiente de Determinação)
R² = 1 - [SS_res / SS_tot] = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]
"Quanto da variação total o modelo explica?" (1.0 = perfeito, 0.0 = Dummy)

MAE (Mean Absolute Error)
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
"Em média, erra por €X." Robusto a outliers. Mesma unidade do target.

RMSE (Root Mean Squared Error)
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
Penaliza erros grandes mais que MAE. Útil quando erros grandes são desproporcionalmente ruins.

11. Train vs Test Error (Bias-Variance)
Decomposição do erro
Erro Total = Viés² + Variância + Ruído irredutível
Diagnóstico
Table



Train Error


Test Error


Diagnóstico


...


Baixo	Baixo	✅ Bom fit	...
Baixo	Alto	⚠️ Overfitting (alta variância)	...
Alto	Alto	⚠️ Underfitting (alto viés)	...
View more
12. Hyperparameter Tuning
Parâmetros vs Hiperparâmetros
Parâmetros: Modelo aprende (coeficientes, thresholds) — definidos durante .fit()
Hiperparâmetros: Você define antes (n_estimators, learning_rate) — controlam o aprendizado
RandomizedSearchCV
Define espaço de busca (valores possíveis para cada hiperparâmetro)
Sorteia N combinações aleatórias
Avalia cada uma com cross-validation
Retorna a melhor
Por que Random > Grid?
Grid Search testa TODAS as combinações (explosão exponencial). Random Search com ~60 tentativas encontra top 5% com alta probabilidade (Bergstra & Bengio, 2012).

13. Feature Importance
Gain-based (XGBoost)
Para cada feature, soma a redução de erro em todos os splits onde foi usada, normalizada pelo total.

Utilidade
Validação: Confirma que o modelo aprendeu padrões significativos
Interpretabilidade: Explica decisões para stakeholders
Próximos passos: Guia feature engineering
14. Análise de Resíduos
O que são resíduos?
resíduo = y_real - y_predito
Positivo → modelo subestimou
Negativo → modelo superestimou
Média ≈ 0 → sem viés sistemático
Por que analisar por segmento?
MAE global de €130k pode esconder que:

Para casas de €200k, erro é 65% (péssimo)
Para casas de €800k, erro é 16% (aceitável)
Entender ONDE o modelo falha é mais valioso que uma métrica única.

Referências
Rubin, D.B. (1976). "Inference and Missing Data." Biometrika
Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR
Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD
Breiman, L. (2001). "Random Forests." Machine Learning
Hastie, Tibshirani & Friedman (2009). "The Elements of Statistical Learning." Springer
