import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Abstract Model class
class Model(ABC):
    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def compute(self, test_data):
        pass

# Example implementation of the Model class
class MovingAverageModel(Model):
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.train_data = None

    def train(self, train_data):
        self.train_data = train_data['Close']

    def compute(self, test_data):
        combined_data = pd.concat([self.train_data, test_data['Close']])
        predictions = combined_data.rolling(window=self.window_size).mean().shift(1)
        predictions = predictions.loc[test_data.index]
        return predictions

class ARIMAModel(Model):
    def __init__(self, frac=0.5):
        self.frac = frac
        self.train_data = None
        self.model_fit = None


    def train(self, train_data):
        from statsmodels.tsa.stattools import adfuller
        self.train_data = train_data['Close']
        adf_result = adfuller(self.train_data)
        print(f'ADF Statistic (Differenced Data): {adf_result[0]}')
        print(f'p-value (Differenced Data): {adf_result[1]}')
        model = ARIMA(self.train_data, order=(2, 0, 2))
        # Fit the model
        self.model_fit = model.fit()
        # Model summary
        print(self.model_fit.summary())

    def compute(self, test_data):
        forecast = self.model_fit.forecast(steps=len(test_data))
        return forecast

class LSTMModel(Model):
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.trainX = None
        self.trainY = None

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def train(self, train_data):
        # Extract 'Close' prices and scale them
        close_prices = train_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)

        # Create training datasets
        self.trainX, self.trainY = self.create_dataset(scaled_data)
        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], self.trainX.shape[1], 1))

        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(self.trainX, self.trainY, epochs=10, batch_size=32, verbose=1)

    def compute(self, test_data):
        # Prepare test data
        total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - self.look_back:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        X_test = []
        for i in range(self.look_back, len(inputs)):
            X_test.append(inputs[i - self.look_back:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict stock prices
        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)

        # Create a DataFrame with predictions
        predictions = pd.Series(predicted_stock_price.flatten(), index=test_data.index[self.look_back:])
        return predictions
