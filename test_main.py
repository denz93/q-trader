import unittest
from unittest.mock import patch, MagicMock, mock_open
import yaml
import os
import signal
import sys
from main import validate_config, save_progress, signal_handler, train, resume, evaluate, plot_evaluation

class TestMain(unittest.TestCase):

    @patch("main.yaml.safe_load")
    def test_validate_config_valid(self, mock_yaml_load):
        config = {
            "agent": {"memory_size": 1000, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995},
            "env": {"opportunity_cost": 0.1, "double_bet_penalty": 0.2, "wait_too_long_penalty": 0.3, "sell_bonus": 0.4},
            "train": {"stock_name": "AAPL", "window_size": 10, "episode_count": 100, "save_step": 10, "batch_size": 32}
        }
        validate_config(config)  # Should not raise any exception

    def test_validate_config_invalid(self):
        config = {
            "agent": {"memory_size": 1000, "epsilon": 1.0},
            "env": {"opportunity_cost": 0.1},
            "train": {"stock_name": "AAPL"}
        }
        with self.assertRaises(ValueError):
            validate_config(config)

    @patch("main.logging.getLogger")
    @patch("main.yaml.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_progress(self, mock_open_file, mock_yaml_dump, mock_get_logger):
        save_progress()
        mock_open_file.assert_called_once_with("models/metadata_resume.yml", "w")
        mock_yaml_dump.assert_called_once()
        mock_get_logger.assert_called_once_with("q-trader")

    @patch("main.sys.exit")
    @patch("main.save_progress")
    @patch("main.logging.getLogger")
    def test_signal_handler_train(self, mock_get_logger, mock_save_progress, mock_sys_exit):
        args = MagicMock()
        args.command = "train"
        signal_handler(signal.SIGINT, args)
        mock_save_progress.assert_called_once()
        mock_sys_exit.assert_called_once_with(0)

    @patch("main.sys.exit")
    @patch("main.logging.getLogger")
    def test_signal_handler_non_train(self, mock_get_logger, mock_sys_exit):
        args = MagicMock()
        args.command = "evaluate"
        signal_handler(signal.SIGINT, args)
        mock_sys_exit.assert_called_once_with(0)

    @patch("main.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("main.train_agent")
    @patch("main.getStockDataVec")
    @patch("main.Agent")
    @patch("main.TradingEnv")
    def test_train(self, mock_trading_env, mock_agent, mock_get_stock_data, mock_train_agent, mock_open_file, mock_yaml_load):
        mock_yaml_load.return_value = {
            "agent": {"memory_size": 1000, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995},
            "env": {"opportunity_cost": 0.1, "double_bet_penalty": 0.2, "wait_too_long_penalty": 0.3, "sell_bonus": 0.4},
            "train": {"stock_name": "AAPL", "window_size": 10, "episode_count": 100, "save_step": 10, "batch_size": 32}
        }
        train("config.yml")
        
        mock_open_file.assert_any_call("config.yml", "r")
        mock_train_agent.assert_called_once()

    unittest.skip("resume function subject to change")
    @patch("main.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("main.load_model")
    @patch("main.Agent")
    @patch("main.TradingEnv")
    def test_resume(self, mock_trading_env, mock_agent, mock_load_model, mock_open_file, mock_yaml_load):
        mock_yaml_load.return_value = {
            "memory_size": 1000, "epsilon_min": 0.01, "epsilon_decay": 0.995,
            "opportunity_cost": 0.1, "double_bet_penalty": 0.2, "wait_too_long_penalty": 0.3, "sell_bonus": 0.4,
            "stock_name": "AAPL", "window_size": 10, "episode_count": 100, "save_step": 10, "batch_size": 32
        }
        with patch("os.path.exists", return_value=True):
            resume()
            mock_open_file.assert_called_once_with("models/metadata_resume.yml", "r")
            mock_load_model.assert_called_once()

    @patch("main.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("main.evaluate_agent", return_value=(1500, [{"action": "BUY", "time": 0, "price": 100}], 5, 10))
    @patch("main.getStockDataVec")
    @patch("main.Agent")
    @patch("main.TradingEnv")
    def test_evaluate(self, mock_trading_env, mock_agent, mock_get_stock_data, mock_evaluate_agent, mock_open_file, mock_yaml_load):
        mock_yaml_load.return_value = {
            "train": {"window_size": 10},
            "env": {"opportunity_cost": 0.1, "double_bet_penalty": 0.2, "wait_too_long_penalty": 0.3, "sell_bonus": 0.4}
        }
        evaluate("model_name", "data_name", "config.yml")
        mock_open_file.assert_any_call("config.yml", "r")
        mock_evaluate_agent.assert_called_once()

    @patch("main.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("main.plot_candle")
    @patch("main.getStockDataVec")
    @patch("pandas.read_csv")
    def test_plot_evaluation(self, mock_read_csv, mock_get_stock_data, mock_plot_candle, mock_open_file, mock_yaml_load):
        mock_yaml_load.return_value = {
            "data_name": "data.csv",
            "transactions": [],
            "total_profit": 1000
        }
        plot_evaluation("config.yml")
        mock_open_file.assert_called_once_with("config.yml", "r")
        mock_plot_candle.assert_called_once()

if __name__ == "__main__":
    unittest.main()