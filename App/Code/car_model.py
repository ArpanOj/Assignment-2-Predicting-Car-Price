import pandas as pd
import numpy as np


class CarPriceModel:

    def __init__(self, model, scaler, medians, modes, columns):
        self.model = model
        self.scaler = scaler
        self.medians = medians
        self.modes = modes
        self.columns = columns

    def preprocess(self, data):

        data = data.copy()

        num_features = [
            'year',
            'km_driven',
            'owner',
            'mileage',
            'engine',
            'max_power',
            'seats'
        ]

        cat_features = [
            'name',
            'fuel',
            'seller_type',
            'transmission'
        ]

        # Fill missing numerical values
        for column, median in self.medians.items():
            data[column] = data[column].fillna(median)

        # Fill missing categorical values
        for column, mode in self.modes.items():
            data[column] = data[column].fillna(mode)

        # Scale numerical features
        data[num_features] = self.scaler.transform(
            data[num_features]
        )

        # One-hot encode categorical features
        data = pd.get_dummies(
            data,
            columns=cat_features,
            drop_first=False
        )

        # Make the columns match the training data
        data = data.reindex(
            columns=self.columns,
            fill_value=0
        )

        return data

    def predict(self, data):

        data_processed = self.preprocess(data)

        # Model predicts log(price)
        predicted_log_price = self.model.predict(
            data_processed
        )

        # Convert back to actual price
        predicted_price = np.exp(predicted_log_price)

        return predicted_price