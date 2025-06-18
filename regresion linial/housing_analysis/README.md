# Análisis y Modelo de Regresión Lineal para Datos de Viviendas

## Descripción
Este proyecto realiza un análisis exploratorio y desarrolla un modelo de regresión lineal para predecir el valor medio de las viviendas usando el dataset `housing.csv`.

Se incluyen pasos completos desde la limpieza y preprocesamiento de los datos, visualización, ingeniería de características, hasta el entrenamiento y evaluación del modelo.

## Contenido
- Limpieza de datos y manejo de valores faltantes
- Codificación one-hot para variables categóricas
- Visualizaciones: histogramas, mapas de dispersión, mapas de calor
- Creación de nuevas características (e.g., proporción de dormitorios a habitaciones)
- División en conjuntos de entrenamiento y prueba
- Modelo de regresión lineal con evaluación por R² y error cuadrático medio
- Escalado de variables con StandardScaler

## Requisitos
- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy

Puedes instalar las dependencias usando:

```bash
pip install pandas seaborn matplotlib scikit-learn numpy
```

## Uso
1. Coloca el archivo `housing.csv` en el mismo directorio que el script.
2. Ejecuta el script:

```bash
python housing_analysis.py
```

El script realizará el análisis, mostrará gráficos y evaluará el modelo.

## Datos
El dataset `housing.csv` debe contener las siguientes columnas relevantes:

- `median_house_value` (valor objetivo)
- `ocean_proximity` (variable categórica)
- `median_income`
- `total_bedrooms`
- `total_rooms`
- `latitude`
- `longitude`
- `population`
- Entre otras variables numéricas relacionadas con viviendas

## Autor
Jose Vigo


# #####################################################################

# Analysis and Linear Regression Model for Housing Data

## Description
This project performs exploratory analysis and develops a linear regression model to predict median home values ​​using the `housing.csv` dataset.

Complete steps are included, from data cleaning and preprocessing, visualization, feature engineering, to model training and evaluation.

## Contents
- Data cleaning and missing values ​​handling
- One-hot encoding for categorical variables
- Visualizations: histograms, scatter plots, heat maps
- Creating new features (e.g., bedroom-to-room ratio)
- Splitting into training and test sets
- Linear regression model with R² and mean square error evaluation
- Scaling variables with StandardScaler

## Requirements
- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy

You can install the dependencies using:

```bash
pip install pandas seaborn matplotlib scikit-learn numpy
```

## Usage
1. Place the `housing.csv` file in the same directory as the script.

``` 2. Run the script:

```bash
python housing_analysis.py
```

The script will perform the analysis, display graphs, and evaluate the model.

## Data
The `housing.csv` dataset should contain the following relevant columns:

- `median_house_value` (target value)
- `ocean_proximity` (categorical variable)
- `median_income`
- `total_bedrooms`
- `total_rooms`
- `latitude`
- `longitude`
- `population`
- Among other housing-related numerical variables

## Author
Jose Vigo