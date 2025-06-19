from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def main():
    # Cargar el dataset de cáncer de mama
    datos = datasets.load_breast_cancer(as_frame=True)

    # Exploración inicial (puedes comentar estas líneas en producción)
    print(datos.frame.head())
    print(datos.frame.info())
    print(datos.frame.isna().sum())
    print(datos.frame.describe())
    print("Forma del dataset:", datos.data.shape)

    # Separar datos y etiquetas
    x = datos.data
    y = datos.target

    # Separar en conjuntos de entrenamiento y prueba
    x_ent, x_pru, y_ent, y_pru = train_test_split(x, y, test_size=0.3, random_state=42)

    # Probar distintos kernels y valores de gamma
    kernels = ["linear", "rbf", "sigmoid"]
    gammas = [1, 0.01, 0.001, 0.0001, 0.00001]

    for kernel in kernels:
        for gamma in gammas:
            modelo = svm.SVC(kernel=kernel, gamma=gamma)
            modelo.fit(x_ent, y_ent)
            predicciones = modelo.predict(x_pru)
            
            print(f"Kernel: {kernel}, Gamma: {gamma}")
            print("  - Exactitud :", metrics.accuracy_score(y_pru, predicciones))
            print("  - Precisión :", metrics.precision_score(y_pru, predicciones))
            print()

    # Usar modelo final con kernel linear para evaluación final
    modelo_final = svm.SVC(kernel="linear")
    modelo_final.fit(x_ent, y_ent)
    predicciones_final = modelo_final.predict(x_pru)

    print("Reporte de Clasificación (modelo final):")
    print(classification_report(y_pru, predicciones_final))

    print("Matriz de Confusión:")
    print(pd.DataFrame(confusion_matrix(y_pru, predicciones_final)))

if __name__ == "__main__":
    main()
