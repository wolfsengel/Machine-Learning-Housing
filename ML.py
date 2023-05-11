# Importar librerías
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar conjunto de datos
from sklearn.datasets import load_boston
boston = load_boston()

# Crear un dataframe con los datos
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

# Definir variables
X = df['RM'].values.reshape(-1,1)  # variable independiente
y = df['target'].values.reshape(-1,1)  # variable dependiente

# Crear modelo de regresión lineal
reg = LinearRegression().fit(X, y)

# Realizar predicciones
y_pred = reg.predict(X)

# Graficar resultados
plt.scatter(X, y, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.title('Relación entre el tamaño de una casa y su precio de venta')
plt.xlabel('Número medio de habitaciones por vivienda')
plt.ylabel('Precio medio de la vivienda ($1000s)')
plt.show()
