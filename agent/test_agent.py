import unittest
from unittest.mock import MagicMock
import numpy as np
from agent.agent import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 10
        self.agent = Agent(state_size=self.state_size, is_eval=False)

    def test_act_random(self):
        state = (np.random.rand(self.state_size, 5), np.random.rand(4))
        action = self.agent.act(state)
        self.assertIn(action, range(self.agent.action_size), "Action should be within valid range.")

    def test_act_predict(self):
        self.agent.epsilon = 0  # Force prediction
        self.agent.model.predict = MagicMock()
        self.agent.model.predict.return_value = [1, 2, 3, 4]
        state = (np.random.rand(self.state_size, 5), np.random.rand(4))
        action = self.agent.act(state)
        self.agent.model.predict.assert_called_once()
        self.assertIn(action, range(self.agent.action_size), "Action should be within valid range.")

    def test_decay_epsilon(self):
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon, "Epsilon should decay.")

    def test_expReplay(self):
        state = (np.random.rand(self.state_size, 5), np.random.rand(4))
        next_state = (np.random.rand(self.state_size, 5), np.random.rand(4))
        self.agent.memory.append((state, 1, 1.0, next_state, False))
        loss = self.agent.expReplay(batch_size=1)
        self.assertGreaterEqual(loss, 0, "Loss should be a non-negative value.")
        self.assertAlmostEqual(self.agent.epsilon, 1)

    def test_predict(self):
        state = (np.random.rand(self.state_size, 5), np.random.rand(4))
        predictions = self.agent.predict(state)
        self.assertEqual(predictions.shape, (1, self.agent.action_size), "Predictions should match action size.")

    def test_reset(self):
        self.agent.predict_times = [0.1, 0.2, 0.3]
        self.agent.train_times = [0.4, 0.5, 0.6]
        self.agent.reset()
        self.assertEqual(len(self.agent.predict_times), 0, "Predict times should be reset.")
        self.assertEqual(len(self.agent.train_times), 0, "Train times should be reset.")

    def test_get_avg_predict_time(self):
        self.agent.predict_times = [0.1, 0.2, 0.3]
        avg_time = self.agent.get_avg_predict_time()
        self.assertAlmostEqual(avg_time, 0.2, 6, "Average predict time should be calculated correctly.")

    def test_get_avg_train_time(self):
        self.agent.train_times = [0.4, 0.5, 0.6]
        avg_time = self.agent.get_avg_train_time()
        self.assertAlmostEqual(avg_time, 0.5, 6, "Average train time should be calculated correctly.")

if __name__ == "__main__":
    unittest.main()