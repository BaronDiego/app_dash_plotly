import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H2("PRONÓSTICO EJECUCIÓN DE UN PROYECTO USANDO REGRESIÓN LINEAL", style={"textAlign":"center", "color":"#002060"}),
    html.P("Se está utilizando una base de datos donde se registra el avance planeado y ejecutado día a día, hasta el 21 de septiembre del presente año. El avance planeado al 21/09/2024 es de 14.21%, mientras que el avance ejecutado a esa misma fecha es de 7.20%.",
           style={'display': 'block','fontSize': 18,'margin': '25px', 'text-align': 'center'}),
    html.Label("Introduce el % planeado segun la fecha a predecir avance (número entre 1 y 100): ", style={'fontSize': 20,"textAlign": "center",'margin': '25px'}),
    dcc.Input(id='input-numero', type='number', value=1, min=1, max=100,
              style={'display':'inline-block','border':'1px solid #ccc', 'border-radius': '4px','box-sizing': 'border-box', 'justify-content': 'center','align-items': 'center'}),
    html.Hr(),
    html.Div(id='resultado', style={'fontSize': 20, "border": "2px solid blue", "background-color": "lightblue","textAlign": "center", 'margin': '25px'}),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Output('resultado', 'children'),
    Input('input-numero', 'value')
    )
def train_and_display(valor):
    df = df = pd.read_excel("data.xlsx")
    X = df[['Programado']].values
    X_train, X_test, y_train, y_test = train_test_split(X, df['Ejecutado'], test_size=0.25,random_state=42)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='prediction')
    ],
        go.Layout(title="Relación entre Programado y Ejecutado (Datos del 01/01/2024 al 21/09/2024)", xaxis=dict(title="Programado"), yaxis=dict(title="Ejecutado"))
    )

    if valor is None or valor < 1 or valor > 100:
        return "Por favor ingresa un número entero positivo"
    else:
        planeado = valor
        prediccion = model.predict([[planeado]])
        entero = float(prediccion[0])
        print(entero)
        print(valor)

    return fig, f"El pronostico, segun el porcentaje planeado es: {entero :.2f}%"


if __name__ == '__main__':
    app.run(debug=True)