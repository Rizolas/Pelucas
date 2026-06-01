import pandas as pd
from sklearn.linear_model import LinearRegression
datos = pd.read_csv('datos.csv')
X = datos[['Publicidad']]
y = datos['Ventas']
modelo = LinearRegression()
modelo.fit(X, y)
prediccion = modelo.predict([[120]])
print(prediccion)
