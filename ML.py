import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO']

# cargar el conjunto de datos
raw_df.columns = column_names
# definir variables
X = raw_df['RM'].values.reshape(-1, 1) # variable independiente
y = raw_df['MEDV'].values.reshape(-1, 1) # variable dependiente


# crear modelo de regresión lineal
reg = LinearRegression().fit(X, y)

# realizar predicciones
y_pred = reg.predict(X)

# graficar resultados
plt.scatter(X, y, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.title('Relación entre el tamaño de una casa y su precio de venta')
plt.xlabel('Número medio de habitaciones por vivienda')
plt.ylabel('Precio medio de la vivienda ($1000s)')
plt.show()
