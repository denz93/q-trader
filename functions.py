import numpy as np
import math
import pandas as pd

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	data = pd.read_csv("data/" + key + ".csv")
	data = data.drop(columns=["Close", "Date"])
	data = data.rename(columns={"Adj Close": "Close"})

	return data 

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t

cache = {} # type: dict[int, np.ndarray[np.float32]]
data_cache = None # type: np.ndarray[np.float32]
def getState(data, t, n, portfolio=[0, 0, 0, 0]):
	global data_cache
	if data_cache is None:
		_ = data.copy(deep=True) # type: pd.DataFrame
		ohlc = ["Open", "High", "Low", "Close"]
		shifted = _[ohlc].shift(1, axis=0)
		shifted.iloc[0] = _[ohlc].iloc[0]
		_[ohlc] = (_[ohlc] - shifted) / shifted
		_["Volume"] = _["Volume"].apply(lambda x: np.log(x + 1e-6))
		_["Volume"] = (_["Volume"] - _["Volume"].mean()) / _["Volume"].std()
		data_cache = _.to_numpy().tolist()
		padding = np.array([ data_cache[0] for i in range(n - 1)])
		data_cache = [*padding, *data_cache]
	
	t_cache = t + n - 1
	block = data_cache[t_cache - (n-1): t_cache + 1]
	final_state = [block, portfolio]
	return final_state
