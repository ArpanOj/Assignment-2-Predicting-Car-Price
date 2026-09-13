import numpy as np
import pandas as pd
 
 
class CarPriceModelV2:
 
    def __init__(self, model, scaler, medians, modes, columns, num_features, cat_features,
                 poly=None, poly_scaler=None):
        self.model = model                  
        self.scaler = scaler                
        self.medians = medians              
        self.modes = modes                  
        self.columns = columns              
        self.num_features = num_features    
        self.cat_features = cat_features    
 

        self.poly = poly
        self.poly_scaler = poly_scaler
 
    def preprocess(self, data):
        data = data.copy()
 
        # Fill any missing numeric fields with the training-set median 
        for column, median in self.medians.items():
            data[column] = data[column].fillna(median)
 
        # Fill any missing categorical fields with the training-set mode
        for column, mode in self.modes.items():
            data[column] = data[column].fillna(mode)
 
        # Scale numeric features using the scaler fit during training 
        data[self.num_features] = self.scaler.transform(data[self.num_features])
 

        data = pd.get_dummies(data, columns=self.cat_features, drop_first=False)
 
        # Force the exact column set/order the (pre-polynomial-expansion) training data had.
        data = data.reindex(columns=self.columns, fill_value=0)
 
        if self.poly is not None:
            # 1. Expand only the numeric columns.
            numeric_expanded = self.poly.transform(data[self.num_features])
            if self.poly_scaler is not None:
                numeric_expanded = self.poly_scaler.transform(numeric_expanded)
 
            cat_cols = [c for c in self.columns if c not in self.num_features]
            other_part = data[cat_cols].to_numpy()
 
            X = np.concatenate([numeric_expanded, other_part], axis=1).astype(float)
        else:
            X = data.to_numpy().astype(float)
 
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X
 
    def predict(self, data):
        X = self.preprocess(data)
 
        # The model predicts log(selling_price) -- convert back to an actual rupee figure.
        predicted_log_price = self.model.predict(X)
        predicted_price = np.exp(predicted_log_price)
 
        return predicted_price