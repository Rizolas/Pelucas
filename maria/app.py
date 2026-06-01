from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
datos = pd.read_csv('datos.csv')
X = datos[['Publicidad']]
y = datos['Ventas']
modelo = LinearRegression()
modelo.fit(X, y)
@app.route('/')
def inicio():
 return render_template('index.html')
@app.route('/predecir', methods=['POST'])
def predecir():
 publicidad = float(request.form['publicidad'])
 prediccion = modelo.predict([[publicidad]])
 resultado = round(prediccion[0], 2)
 return render_template(
 'index.html',
 resultado=resultado
 )
if __name__ == '__main__':
 app.run(debug=True)
