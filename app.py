import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import dash_bootstrap_components as dbc
import json
from datetime import date

import locale
# Setea la variable LC_ALL al conjunto de código UTF-8 con descripción español España
locale.setlocale(locale.LC_ALL,'es_ES.UTF-8')


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

data_fechas = pd.read_excel("fechas_prog.xlsx")
data_fechas["Fecha"] = pd.to_datetime(data_fechas['Fecha'])

filename_in='data.json'
with open(filename_in, "r", encoding="utf8") as json_file:
    data = json.load(json_file)

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"])

app.layout = html.Div([
    html.H3("REVERGY COLOMBIA PROYECTO PUERTA DE ORO | PV | PRONÓSTICO EJECUCIÓN USANDO REGRESIÓN LINEAL", style={
        "textAlign":"center", "color":"#094780","border": "2px solid #094780", 'margin': '25px', "padding": "10px"}),
    html.Label("Introduce la fecha a predecir avance: ", style={'fontSize': 20,"textAlign": "center",'margin': '25px'}),
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=date(2024, 1, 1),
        max_date_allowed=date(2026, 12, 31),
        initial_visible_month=date(2025, 11, 15),
        date=date(2025, 11, 15)
    ),
    html.Hr(),
    html.H3(id='resultado', style={'fontSize': 22, "border": "2px solid blue", "background-color": "lightblue","textAlign": "center", 'margin': '25px',"padding": "5px"}),
    dcc.Graph(id="graph"),
    html.Div(id='mse', style={'fontSize': 20,"textAlign": "center", 'margin': '25px'}),
    html.Div(id='r2', style={'fontSize': 20,"textAlign": "center", 'margin': '25px'}),
    html.A('Dashboard', href = 'https://app.powerbi.com/view?r=eyJrIjoiMTA1OTgyNjQtOTlkMy00NzEyLTg4MjItMzQxMTY5ODI5ZDhkIiwidCI6IjhmZmYyZTJmLWIwOTEtNGNhMi05NTdmLWE2M2U4NWM0ZTU0MiJ9', target = '_blank',
        style = {'color':'blue', 'fontSize':'25px','fontFamily':'Times New Roman', 'textAlign':'center','align':'center'}),
])


@app.callback(
    Output("graph", "figure"),
    Output('resultado', 'children'),
    Output('mse', 'children'),
    Output('r2', 'children'),
    Input('my-date-picker-single', 'date')
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

    planeado = data_fechas[data_fechas["Fecha"] == valor]["Programado LB"].values
    planaedo_float = float(planeado)
    prediccion = model.predict([[planaedo_float]])
    entero = float(prediccion[0]*100)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    valor = date.fromisoformat(valor)
    valor = valor.strftime("%d %B del %Y")

    return fig, f"El pronóstico de avance para el {valor}  es del {entero :.2f}%", f"MSE: {mse}", f"r2: {r2}"

server = app.server

if __name__ == '__main__':
    app.run()