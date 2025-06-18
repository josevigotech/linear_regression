import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Leer archivo CSV
datos = pd.read_csv("../data/celsius.csv")

# Limpiar espacios en los nombres de columnas
datos.columns = datos.columns.str.strip()

# Mostrar info del DataFrame
print("\n--- Información del DataFrame ---")
datos.info()

# Mostrar nombres de columnas con comillas para detectar espacios
print("\n--- Columnas del DataFrame ---")
for col in datos.columns:
    print(f"Column: '{{col}}'")

# Mostrar primeras filas
print("\n--- Primeras filas del DataFrame ---")
print(datos.head())

# Variables para análisis y modelado
x = datos["celsius"]
y = datos["fahrenheit"]

# Preparar variables para sklearn (arrays 2D)
x_procesada = x.values.reshape(-1, 1)
y_procesada = y.values.reshape(-1, 1)

# Visualización gráfica con paleta de colores según fahrenheit
print("\n--- Mostrando gráfico de dispersión con paleta ---")
sb.scatterplot(x="celsius", y="fahrenheit", data=datos,
               hue="fahrenheit", palette="coolwarm")
plt.title("Relación Celsius-Fahrenheit con color por Fahrenheit")
plt.xlabel("Temperatura en Celsius")
plt.ylabel("Temperatura en Fahrenheit")
plt.grid(True)
plt.show()

# Crear y entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(x_procesada, y_procesada)

# Predicción para valor dado
prediccion = modelo.predict([[7900]])
print(f"\nPredicción para 7900: {prediccion[0][0]}")

# Evaluar precisión del modelo (R^2)
score = modelo.score(x_procesada, y_procesada)
print(f"Precisión del modelo (R^2): {score:.4f}")