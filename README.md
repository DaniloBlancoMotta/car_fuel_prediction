# Car Fuel Efficiency Prediction - Homework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)](https://xgboost.readthedocs.io/)

Projeto de regressão para prever a eficiência de combustível (MPG) de veículos usando Decision Trees, Random Forest e XGBoost.

## Dataset

- **Fonte**: `car_fuel_efficiency.csv`
- **Target**: `fuel_efficiency_mpg`
- **Features**: engine_displacement, num_cylinders, horsepower, vehicle_weight, acceleration, model_year, origin, fuel_type, drivetrain, num_doors

## Preparação dos Dados

- Valores faltantes preenchidos com 0
- Split: 60% treino / 20% validação / 20% teste
- `random_state=1`
- `DictVectorizer(sparse=True)` para transformação

## Questões Resolvidas

### Q1: Decision Tree (max_depth=1)
Feature usada para split: **vehicle_weight**

### Q2: Random Forest (n_estimators=10)
RMSE na validação: **~4.5**

### Q3: n_estimators
RMSE para de melhorar em: **n_estimators=80**

### Q4: Melhor max_depth
Testando [10, 15, 20, 25] com n_estimators de 10 a 200:
- Melhor: **max_depth=10**

### Q5: Feature Importance
Com n_estimators=10, max_depth=20:
- Feature mais importante: **vehicle_weight**

### Q6: XGBoost - Melhor eta
Comparando eta=0.3 vs eta=0.1:
- **eta=0.1** tem melhor RMSE (0.4262 vs 0.4502)

## Instalação

```bash
pip install pandas numpy scikit-learn xgboost jupyter
```

## Execução

```bash
# Script Python
python regression_model.py

# Jupyter Notebook
jupyter notebook homework.ipynb
```

## Visualizações

O projeto inclui visualizações animadas usando `matplotlib.animation`:
- RMSE vs n_estimators
- RMSE vs max_depth
- Feature Importance
- XGBoost eta comparison

Veja em: `visualizations.ipynb`

## Estrutura

```
Lesson 6/
├── car_fuel_efficiency.csv
├── regression_model.py
├── homework.ipynb
├── visualizations.ipynb
├── README.md
└── .gitignore
```

## Autor

**Danilo Blanco Motta**
- GitHub: [@DaniloBlancoMotta](https://github.com/DaniloBlancoMotta)
