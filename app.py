import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import dash_bootstrap_components as dbc
import json


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

filename_in='data.json'
with open(filename_in, "r", encoding="utf8") as json_file:
    data = json.load(json_file)

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"])

app.layout = html.Div([
    html.H2("PRONÓSTICO EJECUCIÓN DE UN PROYECTO USANDO REGRESIÓN LINEAL", style={"textAlign":"center", "color":"#002060"}),
    html.P("Se está utilizando una base de datos donde se registra el avance planeado y ejecutado día a día, a la fecha.",
           style={'display': 'block','fontSize': 18,'margin': '25px', 'text-align': 'center'}),
    html.Label("Introduce el % planeado segun la fecha a predecir avance (número entre 1 y 100): ", style={'fontSize': 20,"textAlign": "center",'margin': '25px'}),
    dcc.Input(id='input-numero', type='number', value=1, min=1, max=100,
              style={'display':'inline-block','border':'1px solid #ccc', 'border-radius': '4px','box-sizing': 'border-box', 'justify-content': 'center','align-items': 'center'}),
    html.Hr(),
    html.Div(id='resultado', style={'fontSize': 20, "border": "2px solid blue", "background-color": "lightblue","textAlign": "center", 'margin': '25px'}),
    dcc.Graph(id="graph"),
    html.Div(id='mse', style={'fontSize': 20,"textAlign": "center", 'margin': '25px'}),
    html.Div(id='r2', style={'fontSize': 20,"textAlign": "center", 'margin': '25px'})
])


@app.callback(
    Output("graph", "figure"),
    Output('resultado', 'children'),
    Output('mse', 'children'),
    Output('r2', 'children'),
    Input('input-numero', 'value')
    )
def train_and_display(valor):
    X = df[['Programado']].values
    y = df['Ejecutado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    y_pred = model.predict(X_test)

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='prediction')
    ],
        go.Layout(title="Relación entre Programado y Ejecutado", xaxis=dict(title="Programado"), yaxis=dict(title="Ejecutado"))
    )

    if valor is None or valor < 1 or valor > 100:
        return "Por favor ingresa un número entero positivo"
    else:
        planeado = valor
        prediccion = model.predict([[planeado]])
        entero = float(prediccion[0])

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return fig, f"El pronostico, segun el porcentaje planeado es: {entero :.2f}%", f"MSE: {mse}", f"r2: {r2}"


if __name__ == '__main__':
    app.run(debug=True)