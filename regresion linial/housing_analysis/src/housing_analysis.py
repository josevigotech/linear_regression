# Importar librerías
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar datos
datos = pd.read_csv("housing.csv")

# Exploración inicial
print(datos["ocean_proximity"].value_counts())
datos.info()
print(datos.describe())

# Histograma de variables numéricas
datos.hist(figsize=(15,8), bins=30, edgecolor="black")
plt.show()

# Scatterplot de ubicación y valor de vivienda, tamaño según población
sb.scatterplot(
    x="latitude", y="longitude", data=datos, 
    hue="median_house_value", palette="coolwarm", 
    s=datos["population"] / 100
)
plt.title("Mapa de precios de viviendas según ubicación y población")
plt.show()

# Scatterplot para viviendas con ingreso alto (>14)
sb.scatterplot(
    x="latitude", y="longitude", 
    data=datos[datos.median_income > 14], 
    hue="median_house_value", palette="coolwarm"
)
plt.title("Viviendas con ingresos altos (>14)")
plt.show()

# Quitar filas con valores nulos
datos_na = datos.dropna()
datos_na.info()

# One-hot encoding para la columna categórica 'ocean_proximity'
dummies = pd.get_dummies(datos_na["ocean_proximity"], dtype=int)
datos_na = datos_na.join(dummies)
datos_na = datos_na.drop(["ocean_proximity"], axis=1)
print(datos_na.head())

# Correlaciones
print(datos.corr())

# Mapa de calor de correlaciones
sb.set(rc={'figure.figsize': (15,8)})
sb.heatmap(datos_na.corr(), annot=True, cmap="YlGnBu")
plt.title("Mapa de calor de correlaciones")
plt.show()

# Correlación ordenada respecto a 'median_house_value'
datos_numeric = datos.select_dtypes(include=["number"])
correlaciones = datos_numeric.corr()["median_house_value"].sort_values(ascending=False)
print("Correlaciones con median_house_value:")
print(correlaciones)

# Scatterplot de ingreso vs valor vivienda
sb.scatterplot(x=datos_na["median_house_value"], y=datos_na["median_income"])
plt.title("Ingreso vs Valor de Vivienda")
plt.show()

# Crear nueva característica: proporción de habitaciones por dormitorio
datos_na["bedroom_ratio"] = datos_na["total_bedrooms"] / datos_na["total_rooms"]

# Mapa de calor actualizado con la nueva característica
sb.set(rc={'figure.figsize': (15,8)})
sb.heatmap(datos_na.corr(), annot=True, cmap="YlGnBu")
plt.title("Mapa de calor actualizado")
plt.show()

# Separar variables independientes y variable objetivo
x = datos_na.drop(["median_house_value"], axis=1)
y = datos_na["median_house_value"]

# División en conjunto de entrenamiento y prueba
x_ent, x_pru, y_ent, y_prue = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Shape de y_prue: {y_prue.shape}")

# Crear y entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(x_ent, y_ent)

# Predicciones sobre conjunto de prueba
predicciones = modelo.predict(x_pru)

# Comparación predicción vs valor real
comparativa = pd.DataFrame({"Predicción": predicciones, "Valor Real": y_prue})
print(comparativa.head())

# Evaluación del modelo (score - R^2)
print(f"Score entrenamiento: {modelo.score(x_ent, y_ent)}")
print(f"Score prueba: {modelo.score(x_pru, y_prue)}")

# Cálculo del error cuadrático medio
mse = mean_squared_error(y_prue, predicciones)
print(f"Mean Squared Error: {mse}")

# Escalar características
scaler = StandardScaler()
x_ent_esc = scaler.fit_transform(x_ent)
x_pru_esc = scaler.transform(x_pru)

# Vista previa de datos escalados
print(pd.DataFrame(x_ent_esc).head())
