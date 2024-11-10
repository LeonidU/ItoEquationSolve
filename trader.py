# Trader class for trading decisions and performance estimation
class Trader:
    def __init__(self, model, initial_capital=10000.0, commission=0.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.positions = 0  # Number of shares held
        self.trade_history = []
        self.model = model

    def trade(self, data, model):
        # Predict future prices
        predictions = model.compute(data)
        
        # Align predictions with data
        data = data.iloc[len(data) - len(predictions):]
        data = data.copy()
        data['Prediction'] = predictions

        for idx in data.index:
            current_price = data.loc[idx, 'Close']
            predicted_price = data.loc[idx, 'Prediction']
            decision = 'hold'

            # Simple decision rule: Buy if predicted price > current price, Sell if predicted price < current price
            if predicted_price > current_price:
                if self.capital >= current_price:
                    # Buy one share
                    self.capital -= current_price * (1 + self.commission)
                    self.positions += 1
                    decision = 'buy'
            elif predicted_price < current_price:
                if self.positions > 0:
                    # Sell one share
                    self.capital += current_price * (1 - self.commission)
                    self.positions -= 1
                    decision = 'sell'

            # Record the trade
            self.trade_history.append({
                'Date': idx,
                'Decision': decision,
                'Price': current_price,
                'Predicted Price': predicted_price,
                'Positions': self.positions,
                'Capital': self.capital
            })

        return pd.DataFrame(self.trade_history).set_index('Date')

    def performance(self):
        final_capital = self.capital + self.positions * self.trade_history[-1]['Price']
        profit_loss = final_capital - self.initial_capital
        return profit_loss, final_capital
