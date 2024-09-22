import yfinance as yf
import pandas as pd
import numpy as np
from models import LSTMModel

def get_user_input():
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    return ticker, start_date, end_date

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker symbol and date range.")
    return data

def split_data(data, train_ratio=0.8):
    split_point = int(len(data) * train_ratio)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    return train_data, test_data

def calculate_profit_loss(test_data, predictions):
    # Simulate a simple trading strategy
    test_data = test_data.copy()
    print(predictions)
    test_data['Prediction'] = predictions.values
    print(test_data)
    test_data.dropna(inplace=True)
    
    test_data['Signal'] = np.where(test_data['Prediction'] > test_data['Close'].shift(1), 1, -1)
    test_data['Market Return'] = test_data['Close'].pct_change()
    test_data['Strategy Return'] = test_data['Market Return'] * test_data['Signal'].shift(1)
    
    # Calculate cumulative returns
    test_data['Cumulative Market Return'] = (1 + test_data['Market Return']).cumprod() - 1
    test_data['Cumulative Strategy Return'] = (1 + test_data['Strategy Return']).cumprod() - 1
    
    total_profit_loss = test_data['Strategy Return'].sum()
    return test_data, total_profit_loss

def main():
    # Step 1: Get user input
    ticker, start_date, end_date = get_user_input()
    
    # Step 2: Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Step 3: Split data into training and testing sets
    train_data, test_data = split_data(data)
    
    # Step 4: Initialize and train the model
    model = LSTMModel()
    model.train(train_data)
    
    # Step 5: Compute predictions on the test set
    predictions = model.compute(train_data, test_data)
    
    # Step 6: Calculate benefits or losses
    results, total_profit_loss = calculate_profit_loss(test_data, predictions)
    
    # Step 7: Display the results
    print("\n--- Trading Simulation Results ---")
    print(f"Total Profit/Loss: {total_profit_loss:.2f}")
    print(f"Cumulative Market Return: {results['Cumulative Market Return'][-1]:.2%}")
    print(f"Cumulative Strategy Return: {results['Cumulative Strategy Return'][-1]:.2%}")
    print("\nDetailed Results:")
    print(results[['Close', 'Prediction', 'Signal', 'Strategy Return']])
    
if __name__ == "__main__":
    main()
