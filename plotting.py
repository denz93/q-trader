import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import logging
import numpy as np

def plot_data_and_transactions(data, transactions, total_profit, stock_name="Stock"):
    """
    Plot the stock data and mark transactions on the plot.

    Args:
        data (pd.DataFrame): The stock data.
        transactions (list): List of transactions (action, time, price).
        total_profit (float): Total profit from the evaluation.
    """
    plt.figure(figsize=(12, 6))

    # Plot the stock closing prices
    plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.7)

    # Mark transactions
    for transaction in transactions:
        if transaction['action'] == 'Long':
            plt.scatter(transaction['time'], transaction['price'], color='green', label='Long', marker='^', alpha=0.8)
        elif transaction['action'] == 'Short':
            plt.scatter(transaction['time'], transaction['price'], color='red', label='Short', marker='v', alpha=0.8)
        elif transaction['action'] == 'Sell':
            plt.scatter(transaction['time'], transaction['price'], color='orange', label='Sell', marker='o', alpha=0.8)

    # Add title and labels
    plt.title(f"Stock Data and Transactions (Total Profit: {total_profit:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend([stock_name])
    plt.grid()

    # Show the plot
    plt.show()


def plot_candle(data: pd.DataFrame, transactions, total_profit):
    """
    Plot the stock data as a candlestick chart and mark transactions on the plot.

    Args:
        data (pd.DataFrame): The stock data with columns ['Open', 'High', 'Low', 'Close'].
        transactions (list): List of transactions (action, time, price).
        total_profit (float): Total profit from the evaluation.
    """
    logger = logging.getLogger("q-trader")

    # Prepare the data for mplfinance
    # logger.info(data.index)
    # data["Date"] = pd.to_datetime(data["Date"])
    mpf_data = data[['Open', 'High', 'Low', 'Close']]
    mpf_data.index = pd.DatetimeIndex(data["Date"])
    # Create markers for transactions
    add_plots = []
    transactions = list(map(lambda tr: {
        'action': tr['action'],
        'time': pd.to_datetime(data["Date"][tr['time']]),
        'price': tr['price']
    }, transactions))
    for transaction in transactions:
        # Create a Series with np.nan and set the transaction price at the correct datetime index
        marker_data = pd.Series(np.nan, index=mpf_data.index)
        marker_data.loc[transaction['time']] = transaction['price'] # type: ignore

        if transaction['action'] == 'Long': # type: ignore
            add_plots.append(mpf.make_addplot(
                marker_data, type='scatter', markersize=50, marker='^', color='green', label='Long'
            ))
        elif transaction['action'] == 'Short': # type: ignore
            add_plots.append(mpf.make_addplot(
                marker_data, type='scatter', markersize=50, marker='v', color='red', label='Short'
            ))
        elif transaction['action'] == 'Sell': # type: ignore
            add_plots.append(mpf.make_addplot(
                marker_data, type='scatter', markersize=50, marker='o', color='orange', label='Sell'
            ))

    # Plot the candlestick chart with transactions
    mpf.plot(
        mpf_data,
        type='candle',
        style='charles',
        title=f"Stock Data and Transactions (Total Profit: {total_profit:.2f})",
        ylabel='Price',
        addplot=add_plots,
        volume=False,
        figsize=(20, 8)
    )