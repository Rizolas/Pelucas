import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

# Leer CSV y normalizar cabeceras
datos = pd.read_csv('datos.csv', encoding='utf-8-sig', skip_blank_lines=True)
datos.columns = datos.columns.str.strip().str.lower()

required = {'publicidad', 'ventas'}
if not required.issubset(set(datos.columns)):
	print('Columnas disponibles:', list(datos.columns))
	raise SystemExit("Faltan columnas 'publicidad' o 'ventas' en datos.csv")

# Seleccionar y limpiar
datos = datos[list(required)].dropna()

X = datos[['publicidad']]
y = datos['ventas']

modelo = LinearRegression()
modelo.fit(X, y)
prediccion = modelo.predict([[120]])
print(prediccion)