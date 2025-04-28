import unittest
from functions import formatPrice, getStockDataVec, getState
import pandas as pd 
import numpy.testing as npt 
import numpy as np

class TestFunctions(unittest.TestCase):
    def test_formatPrice(self):
        self.assertEqual(formatPrice(123.456), "$123.46")
        self.assertEqual(formatPrice(-123.456), "-$123.46")
        self.assertEqual(formatPrice(0), "$0.00")

    def test_getStockDataVec(self):
        data = getStockDataVec("^GSPC_2011")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertEqual(data.columns.tolist(), ["Open", "High", "Low", "Close", "Volume"])
         

    def test_getState(self):
        data = pd.DataFrame({
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1, 2, 3, 4],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [1000, 2000, 3000, 4000, 5000]
        })
        portfolio = [1, 0, 0, 0]
        window = 3
        t = 3
        state, p = getState(data, t, window, portfolio)
        state = np.array(state)
        self.assertEqual(p, [1, 0, 0, 0])
        print(state)
        self.assertEqual(state.shape, (3, 5))
        
if __name__ == "__main__":
    unittest.main()