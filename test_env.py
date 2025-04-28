import unittest
import pandas as pd
import numpy.testing as npt
from env import TradingEnv

whatever = [1,2,3] # doesn't matter what this is
data = pd.DataFrame({
    "Open": [1, 2, 3, 4, 5],
    "High": [2, 3, 4, 5, 6],
    "Low": [0.5, 1, 2, 3, 4],
    "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
    "Volume": [1000, 2000, 3000, 4000, 5000]
})
window_size = 3
env = TradingEnv(
    data=data,
    window_size=window_size,
    opportunity_cost=0.01,
    double_bet_penalty=0.1,
    wait_too_long_penalty=0.2,
    sell_bonus=0.05
)
class TestHoldAction(unittest.TestCase):
    def setUp(self) -> None:
        self.action = 0 #hold
        return super().setUp()
    def test_at_hold(self):
        combined_state = (whatever, [1, 0, 0, 1])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertLess(reward, 0)
    def test_at_long(self):
        combined_state = (whatever, [0, 1, 0, 2/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 1, 0, 1/window_size])
        self.assertEqual(reward, 1/3.5)
    def test_at_short(self):
        combined_state = (whatever, [0, 0, 1, 2/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 0, 1, 1/window_size])
        self.assertEqual(reward, -1/3.5)
    
    def test_at_hold_too_long(self):
        combined_state = (whatever, [1, 0, 0, 1/window_size])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertLess(reward, 0)

    def test_at_long_too_long(self):
        combined_state = (whatever, [0, 1, 0, 1/window_size])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertLess(reward, 0)
    def test_at_short_too_long(self):
        combined_state = (whatever, [0, 0, 1, 1/window_size])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertLess(reward, 0)

class TestLongAction(unittest.TestCase):
    def setUp(self) -> None:
        self.action = 1 #long
        return super().setUp()
    def test_at_hold(self):
        combined_state = (whatever, [1, 0, 0, (window_size-1)/window_size])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [0, 1, 0, (window_size-1)/window_size])
        self.assertEqual(reward, 0)
    def test_at_long(self):
        combined_state = (whatever, [0, 1, 0, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 1, 0, (window_size-2)/window_size])
        self.assertEqual(reward, -env.double_bet_penalty)
    def test_at_short(self):
        combined_state = (whatever, [0, 0, 1, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 0, 1, (window_size-2)/window_size])
        self.assertEqual(reward, -env.double_bet_penalty)    

class TestShortAction(unittest.TestCase):
    def setUp(self) -> None:
        self.action = 2 #short
        return super().setUp()
    def test_at_hold(self):
        combined_state = (whatever, [1, 0, 0, 0])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [0, 0, 1, (window_size-1)/window_size])
        self.assertEqual(reward, 0)
    def test_at_long(self):
        combined_state = (whatever, [0, 1, 0, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 1, 0, (window_size-2)/window_size])
        self.assertEqual(reward, -env.double_bet_penalty)
    def test_at_short(self):
        combined_state = (whatever, [0, 0, 1, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [0, 0, 1, (window_size-2)/window_size])
        self.assertEqual(reward, -env.double_bet_penalty)
class TestSellAction(unittest.TestCase):
    def setUp(self) -> None:
        self.action = 3 #sell
        return super().setUp()
    def test_at_hold(self):
        combined_state = (whatever, [1, 0, 0, (window_size-1)/window_size])
        t = 4
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        npt.assert_almost_equal(next_portfolio, [1, 0, 0, (window_size-2)/window_size])
        self.assertEqual(reward, -env.double_bet_penalty)
    def test_at_long(self):
        combined_state = (whatever, [0, 1, 0, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertEqual(reward, (data["Close"][3] - data["Close"][2]) / data["Close"][2] + env.sell_bonus)
    def test_at_short(self):
        combined_state = (whatever, [0, 0, 1, (window_size-1)/window_size])
        t = 3
        next_state, reward, _, __ = env.step(combined_state, self.action, t)
        next_portfolio = next_state[1]
        self.assertEqual(next_portfolio, [1, 0, 0, (window_size-1)/window_size])
        self.assertEqual(reward, -(data["Close"][3] - data["Close"][2]) / data["Close"][2] + env.sell_bonus)
if __name__ == "__main__":
    unittest.main()