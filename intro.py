import numpy as np
import matplotlib.pyplot as plt

# Generar datos aleatorios
x = np.linspace(0, 10, 100)
y = 2*x + 1 + np.random.randn(100)

# Graficar los datos
plt.plot(x, y, 'o')
plt.show()


from sklearn.linear_model import LinearRegression

# Crear un objeto de regresión lineal
lr = LinearRegression()

# Ajustar el modelo a los datos
lr.fit(x.reshape(-1, 1), y)

# Obtener las predicciones de la línea ajustada
y_pred = lr.predict(x.reshape(-1, 1))

# Graficar los datos y la línea ajustada
plt.plot(x, y, 'o')
plt.plot(x, y_pred, '-')
plt.show()
