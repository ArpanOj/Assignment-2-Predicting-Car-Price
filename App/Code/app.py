from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import pickle

from car_model import CarPriceModel          
from car_model_v2 import CarPriceModelV2     
from regression_models import LinearRegression, Lasso, Ridge, NoRegularization  # noqa: F401


app = Dash(__name__, suppress_callback_exceptions=True)

with open("Car_Price_Model.pkl", "rb") as file:
    loaded_model_v1 = pickle.load(file)

with open("Car_Price_Model_V2.pkl", "rb") as file:
    loaded_model_v2 = pickle.load(file)


# ---------------------------------------------------------------------------
# Shared nav bar, shown on every page
# ---------------------------------------------------------------------------
nav_bar = html.Div([
    dcc.Link("Original Model (v1)", href="/", style={"marginRight": "20px"}),
    dcc.Link("New Model (v2)", href="/v2"),
], style={"padding": "10px", "borderBottom": "1px solid #ccc", "marginBottom": "20px"})


# ---------------------------------------------------------------------------
# Page 1: the original A1 model 
# ---------------------------------------------------------------------------
def layout_v1():
    return html.Div([
        nav_bar,

        html.H1("Car Price Prediction System (Original Model)"),
        html.P("Enter the details of the used car below to predict its selling price."),

        html.Label("Car Brand"), dcc.Input(id="name", type="text", placeholder="e.g. Maruti"),
        html.Br(), html.Br(),

        html.Label("Year"), dcc.Input(id="year", type="number", placeholder="e.g. 2018"),
        html.Br(), html.Br(),

        html.Label("Kilometers Driven"), dcc.Input(id="km_driven", type="number", placeholder="e.g. 40000"),
        html.Br(), html.Br(),

        html.Label("Fuel"),
        dcc.Dropdown(id="fuel", options=[
            {"label": "Diesel", "value": "Diesel"},
            {"label": "Petrol", "value": "Petrol"},
        ], placeholder="Select fuel type"),
        html.Br(),

        html.Label("Seller Type"),
        dcc.Dropdown(id="seller_type", options=[
            {"label": "Individual", "value": "Individual"},
            {"label": "Dealer", "value": "Dealer"},
            {"label": "Trustmark Dealer", "value": "Trustmark Dealer"},
        ], placeholder="Select seller type"),
        html.Br(),

        html.Label("Transmission"),
        dcc.Dropdown(id="transmission", options=[
            {"label": "Manual", "value": "Manual"},
            {"label": "Automatic", "value": "Automatic"},
        ], placeholder="Select transmission"),
        html.Br(),

        html.Label("Owner"),
        dcc.Dropdown(id="owner", options=[
            {"label": "First Owner", "value": 1},
            {"label": "Second Owner", "value": 2},
            {"label": "Third Owner", "value": 3},
            {"label": "Fourth & Above Owner", "value": 4},
            {"label": "Test Drive Car", "value": 5},
        ], placeholder="Select owner type"),
        html.Br(),

        html.Label("Mileage (kmpl)"), dcc.Input(id="mileage", type="number", placeholder="e.g. 20.0"),
        html.Br(), html.Br(),

        html.Label("Engine (CC)"), dcc.Input(id="engine", type="number", placeholder="e.g. 1197"),
        html.Br(), html.Br(),

        html.Label("Max Power (bhp)"), dcc.Input(id="max_power", type="number", placeholder="e.g. 82"),
        html.Br(), html.Br(),

        html.Label("Seats"), dcc.Input(id="seats", type="number", placeholder="e.g. 5"),
        html.Br(), html.Br(),

        html.Button("Predict Price", id="predict_button", n_clicks=0),
        html.Br(), html.Br(),

        html.Div(id="prediction_output"),
    ])


@app.callback(
    Output("prediction_output", "children"),
    Input("predict_button", "n_clicks"),
    State("name", "value"), State("year", "value"), State("km_driven", "value"),
    State("fuel", "value"), State("seller_type", "value"), State("transmission", "value"),
    State("owner", "value"), State("mileage", "value"), State("engine", "value"),
    State("max_power", "value"), State("seats", "value"),
)
def predict_price_v1(n_clicks, name, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):
    if n_clicks == 0:
        return ""

    new_car = pd.DataFrame({
        "name": [name], "year": [year], "km_driven": [km_driven], "fuel": [fuel],
        "seller_type": [seller_type], "transmission": [transmission], "owner": [owner],
        "mileage": [mileage], "engine": [engine], "max_power": [max_power], "seats": [seats],
    })

    predicted_price = loaded_model_v1.predict(new_car)
    return f"Predicted selling price (original model): \u20b9{predicted_price[0]:,.2f}"


# ---------------------------------------------------------------------------
# Page 2: the new Task 2 model. 
# ---------------------------------------------------------------------------
def layout_v2():
    return html.Div([
        nav_bar,

        html.H1("Car Price Prediction System (New Model)"),
        html.P(
            "This page uses the Task 2 model: a linear regression implemented from scratch "
            "with gradient descent, cross-validated and tuned across regularization type "
            "(Lasso/Ridge/none), weight initialization, momentum, batch method, and learning "
            "rate using MLflow. Compared to the original model (a tuned SVR), this one is "
            "fully interpretable -- every prediction is a transparent weighted sum of the "
            "input features, and you can inspect exactly which features drive the price via "
            "the feature-importance plot in the Task 2 notebook."
        ),

        html.Label("Car Brand"), dcc.Input(id="name_v2", type="text", placeholder="e.g. Maruti"),
        html.Br(), html.Br(),

        html.Label("Year"), dcc.Input(id="year_v2", type="number", placeholder="e.g. 2018"),
        html.Br(), html.Br(),

        html.Label("Kilometers Driven"), dcc.Input(id="km_driven_v2", type="number", placeholder="e.g. 40000"),
        html.Br(), html.Br(),

        html.Label("Fuel"),
        dcc.Dropdown(id="fuel_v2", options=[
            {"label": "Diesel", "value": "Diesel"},
            {"label": "Petrol", "value": "Petrol"},
        ], placeholder="Select fuel type"),
        html.Br(),

        html.Label("Seller Type"),
        dcc.Dropdown(id="seller_type_v2", options=[
            {"label": "Individual", "value": "Individual"},
            {"label": "Dealer", "value": "Dealer"},
            {"label": "Trustmark Dealer", "value": "Trustmark Dealer"},
        ], placeholder="Select seller type"),
        html.Br(),

        html.Label("Transmission"),
        dcc.Dropdown(id="transmission_v2", options=[
            {"label": "Manual", "value": "Manual"},
            {"label": "Automatic", "value": "Automatic"},
        ], placeholder="Select transmission"),
        html.Br(),

        html.Label("Owner"),
        dcc.Dropdown(id="owner_v2", options=[
            {"label": "First Owner", "value": 1},
            {"label": "Second Owner", "value": 2},
            {"label": "Third Owner", "value": 3},
            {"label": "Fourth & Above Owner", "value": 4},
        ], placeholder="Select owner type"),
        html.Br(),

        html.Label("Mileage (kmpl)"), dcc.Input(id="mileage_v2", type="number", placeholder="e.g. 20.0"),
        html.Br(), html.Br(),

        html.Label("Engine (CC)"), dcc.Input(id="engine_v2", type="number", placeholder="e.g. 1197"),
        html.Br(), html.Br(),

        html.Label("Max Power (bhp)"), dcc.Input(id="max_power_v2", type="number", placeholder="e.g. 82"),
        html.Br(), html.Br(),

        html.Label("Seats"), dcc.Input(id="seats_v2", type="number", placeholder="e.g. 5"),
        html.Br(), html.Br(),

        html.Button("Predict Price", id="predict_button_v2", n_clicks=0),
        html.Br(), html.Br(),

        html.Div(id="prediction_output_v2"),
    ])


@app.callback(
    Output("prediction_output_v2", "children"),
    Input("predict_button_v2", "n_clicks"),
    State("name_v2", "value"), State("year_v2", "value"), State("km_driven_v2", "value"),
    State("fuel_v2", "value"), State("seller_type_v2", "value"), State("transmission_v2", "value"),
    State("owner_v2", "value"), State("mileage_v2", "value"), State("engine_v2", "value"),
    State("max_power_v2", "value"), State("seats_v2", "value"),
)
def predict_price_v2(n_clicks, name, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):
    if n_clicks == 0:
        return ""

    new_car = pd.DataFrame({
        "name": [name], "year": [year], "km_driven": [km_driven], "fuel": [fuel],
        "seller_type": [seller_type], "transmission": [transmission], "owner": [owner],
        "mileage": [mileage], "engine": [engine], "max_power": [max_power], "seats": [seats],
    })

    predicted_price = loaded_model_v2.predict(new_car)
    return f"Predicted selling price (new model): \u20b9{predicted_price[0]:,.2f}"



app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/v2":
        return layout_v2()
    return layout_v1()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
