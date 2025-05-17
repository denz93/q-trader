import unittest
from unittest.mock import MagicMock, patch
from agent.agent import Agent
from env import TradingEnv
from train import train_agent
import pandas as pd
import logging

class TestTrainAgent(unittest.TestCase):
    def setUp(self):
        # Mocking the Agent and TradingEnv
        self.agent: Agent = MagicMock()
        self.agent.model = MagicMock()
        self.agent.memory = []
        self.agent.epsilon = 1.0 
        self.agent.epsilon_decay = 0.99
        self.agent.epsilon_min = 0.01
        self.agent.get_avg_predict_time.return_value = .5
        self.agent.get_avg_train_time.return_value = .5
        self.env = MagicMock(spec=TradingEnv)
        self.env.total_steps = 10
        

        # Mocking data
        self.data = pd.DataFrame({'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]})
        self.window_size = 3
        self.episode_count = 2
        self.save_step = 2
        self.batch_size = 2
        self.metadata = {"current_episode": 1}

        # Mocking logger
        self.logger = logging.getLogger("q-trader")
        self.logger.setLevel(logging.CRITICAL)  # Suppress logging during tests

    @patch('train.getState', return_value=[0, 1, 2])
    def test_train_agent_execution(self, mock_getState):
        # Mocking environment step
        self.env.step.return_value = ([0, 1, 2, [0, 0, 0, 0]], 1, False, {"bet_violation": 0, "hold_violation": 0})

        # Mocking agent methods
        self.agent.act.return_value = 0
        self.agent.expReplay.return_value = 0.1
        self.agent.get_avg_train_time.return_value = 0.01
        self.agent.get_avg_predict_time.return_value = 0.01

        # Run the training function
        train_agent(self.agent, self.env, self.data, self.window_size, self.episode_count, self.save_step, self.batch_size, self.metadata)

        # Assertions
        self.assertEqual(self.metadata["current_episode"], self.episode_count)
        self.assertTrue(self.agent.act.called)
        self.assertTrue(self.env.step.called)
        self.assertTrue(self.agent.expReplay.called)

    @patch('train.getState', return_value=[0, 1, 2])
    def test_train_agent_saves_model(self, mock_getState):
        # Mocking environment step
        self.env.step.return_value = ([0, 1, 2, [0, 0, 0, 0]], 1, False, {"bet_violation": 0, "hold_violation": 0})

        # Mocking agent methods
        self.agent.act.return_value = 0
        self.agent.expReplay.return_value = 0.1
        self.agent.model.save = MagicMock()

        # Run the training function
        train_agent(self.agent, self.env, self.data, self.window_size, self.episode_count, self.save_step, self.batch_size, self.metadata)

        # Check if the model save method was called
        self.agent.model.save.assert_called_with(f"models/model_ep{self.save_step}.keras")

if __name__ == '__main__':
    unittest.main()