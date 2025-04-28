import argparse
import yaml
import signal
import sys
from train import train_agent
from agent.agent import Agent
from env import TradingEnv
from functions import getStockDataVec
import logging
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
from keras.api.models import load_model
import keras 
import os
from evaluate import evaluate as evaluate_agent
from plotting import plot_data_and_transactions, plot_candle
import pandas as pd 

# Global variables to track the agent, environment, and metadata
current_agent: Agent | None = None
current_env: TradingEnv | None = None
current_metadata = {}

def setup_logger(filename='app.log', command=''):

    logger = logging.getLogger("q-trader")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if os.environ.get("DEBUG"):
        handler = StreamHandler(sys.stdout)
        logger.setLevel(logging.DEBUG)
    else:
        if command in ("evaluate", "plot"):
            handler = StreamHandler(sys.stdout)
        else:
            handler = RotatingFileHandler(filename=filename, maxBytes=100 * 1024, backupCount=5)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def validate_config(config):
    """Validate the configuration file to ensure all required parameters are present."""
    required_keys = {
        "agent": ["memory_size", "epsilon", "epsilon_min", "epsilon_decay"],
        "env": ["opportunity_cost", "double_bet_penalty", "wait_too_long_penalty", "sell_bonus"],
        "train": ["stock_name", "window_size", "episode_count", "save_step", "batch_size"]
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing section '{section}' in the configuration file.")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing key '{key}' in section '{section}' of the configuration file.")

def save_progress():
    """Save the current model and metadata when the process is terminated."""
    global current_agent, current_metadata
    logger = logging.getLogger("q-trader")
    logger.info("Saving training progress...")

    if current_metadata:
        # Save metadata (e.g., current episode, epsilon, etc.)
        with open("models/metadata_resume.yml", "w") as file:
            yaml.dump(current_metadata, file)
        logger.info("Metadata saved to 'models/metadata_resume.yml'.")

    logger.info("Training progress saved. Exiting gracefully.")

def signal_handler(sig, args: argparse.Namespace):
    """Handle termination signals."""
    logger = logging.getLogger("q-trader")
    command = args.command if hasattr(args, 'command') else 'unknown'
    if command != "train" and command != "resume":
        logger.info("Signal handler called, but not in training or resume mode. Ignoring signal.")
        sys.exit(0)
        return

    logger.info(f"Received termination signal: {sig}. Saving progress...")
    save_progress()
    sys.exit(0)

def train(config_path):
    global current_agent, current_env, current_metadata

    # Load configuration from YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate the configuration
    validate_config(config)

    # Extract configurations
    agent_config = config['agent']
    env_config = config['env']
    train_config = config['train']

    # Initialize logger
    logger = setup_logger("train.log")

    # Log configuration
    logger.info("Starting training with the following configuration:")
    for section, section_config in config.items():
        logger.info(f"{section.upper()}:")
        for key, value in section_config.items():
            logger.info(f"  {key}: {value}")

    # Load data and initialize components
    data = getStockDataVec(train_config['stock_name'])
    current_agent = Agent(
        state_size=train_config['window_size'],
        is_eval=False,
        model_name="",
        memory_size=agent_config['memory_size']
    )
    current_env = TradingEnv(
        data=data,
        window_size=train_config['window_size'],
        opportunity_cost=env_config['opportunity_cost'],
        double_bet_penalty=env_config['double_bet_penalty'],
        wait_too_long_penalty=env_config['wait_too_long_penalty'],
        sell_bonus=env_config['sell_bonus']
    )

    # Initialize metadata
    current_metadata = {
        "memory_size": agent_config['memory_size'],
        "epsilon_min": agent_config['epsilon_min'],
        "epsilon_decay": agent_config['epsilon_decay'],
        "opportunity_cost": env_config['opportunity_cost'],
        "double_bet_penalty": env_config['double_bet_penalty'],
        "wait_too_long_penalty": env_config['wait_too_long_penalty'],
        "sell_bonus": env_config['sell_bonus'],
        "stock_name": train_config['stock_name'],
        "window_size": train_config['window_size'],
        "episode_count": train_config['episode_count'],
        "save_step": train_config['save_step'],
        "batch_size": train_config['batch_size'],
        "epsilon": current_agent.epsilon,
        "current_episode": 0  # Start from episode 0
    }

    # Train the agent
    train_agent(
        agent=current_agent,
        env=current_env,
        data=data,
        window_size=train_config['window_size'],
        episode_count=train_config['episode_count'],
        save_step=train_config['save_step'],
        batch_size=train_config['batch_size'],
        metadata=current_metadata  # Pass metadata to track progress
    )

def resume():
    global current_agent, current_env, current_metadata
    setup_logger("train.log")
    logger = logging.getLogger("q-trader")
    logger.info("Resuming training from the last saved state...")
    # Load metadata
    with open("models/metadata_resume.yml", "r") as file:
        current_metadata = yaml.safe_load(file)

    # Check if the model file exists
    model_path = "models/checkpoints/backup.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    # Load model
    current_agent = Agent(
        state_size=current_metadata['window_size'],
        is_eval=False,
        model_name="",
        memory_size=current_metadata['memory_size'],
        metadata=current_metadata
    )
    loaded_model = load_model(model_path, custom_objects={'mse': keras.losses.MeanSquaredError()})
    if not isinstance(loaded_model, keras.Model):
        raise ValueError("Model is not a keras model")
    current_agent.model = loaded_model

    # Load data and environment
    data = getStockDataVec(current_metadata['stock_name'])
    current_env = TradingEnv(
        data=data,
        window_size=current_metadata['window_size'],
        opportunity_cost=current_metadata['opportunity_cost'],
        double_bet_penalty=current_metadata['double_bet_penalty'],
        wait_too_long_penalty=current_metadata['wait_too_long_penalty'],
        sell_bonus=current_metadata['sell_bonus']
    )

    # Resume training
    train_agent(
        agent=current_agent,
        env=current_env,
        data=data,
        window_size=current_metadata['window_size'],
        episode_count=current_metadata['episode_count'],
        save_step=current_metadata['save_step'],
        batch_size=current_metadata['batch_size'],
        metadata=current_metadata
    )

def evaluate(model_name, data_name, config_path):
    logger = setup_logger("evaluate.log", command="evaluate")
    config = yaml.safe_load(open(config_path, 'r'))
    train_config = config["train"]
    agent = Agent(state_size=train_config["window_size"], is_eval=True, model_name=model_name)
    data = getStockDataVec(data_name)
    env = TradingEnv(
        data=data,
        window_size=train_config['window_size'],
        opportunity_cost=config['env']['opportunity_cost'],
        double_bet_penalty=config['env']['double_bet_penalty'],
        wait_too_long_penalty=config['env']['wait_too_long_penalty'],
        sell_bonus=config['env']['sell_bonus']
    )
    total_profit, transactions, bet_violation, hold_violation = evaluate_agent(agent, env, data, config["train"]["window_size"])
    logger.info(f"Total profit: {total_profit}")
    logger.info(f"Bet violations: {bet_violation}")
    logger.info(f"Hold violations: {hold_violation}")
    # Save evaluation results to a file
    results = {
        "data_name": data_name,
        "total_profit": float(total_profit),
        "transactions": list(map(lambda tr: {
            "action": str(tr["action"]),
            "time": int(tr["time"]),
            "price": float(tr["price"]),
        }, transactions)),
        "bet_violation": int(bet_violation),
        "hold_violation": int(hold_violation),
    }
    with open("evaluation-output.yml", "w") as file:
        yaml.dump(results, file)
    logger.info("Evaluation results saved to 'evaluation-output.yml'.")

def plot_evaluation(config_path):
    setup_logger("plot.log", command="plot")
    config = yaml.safe_load(open(config_path, 'r'))
    data_name = config["data_name"]
    transactions = config["transactions"]
    total_profit = config["total_profit"]
    data = getStockDataVec(data_name)
    data = pd.read_csv(f"data/{data_name}.csv", parse_dates=True)
    plot_candle(data, transactions, total_profit)
def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, args))
    signal.signal(signal.SIGTERM, lambda sig, frame:  signal_handler(sig, args))

    parser = argparse.ArgumentParser(description="Q-Trader Application")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the trading agent")
    train_parser.add_argument("--config", required=True, help="Path to the configuration YAML file")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training from the last saved state")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the trading agent")
    evaluate_parser.add_argument("--model", required=True, help="Model name")
    evaluate_parser.add_argument("--data", required=True, help="Data name")
    evaluate_parser.add_argument("--config", required=True, help="Path to the configuration YAML file")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot evaluation results")
    plot_parser.add_argument("--config", required=True, help="Path to the evaluation results YAML file")
    
    # Parse arguments
    args = parser.parse_args()

    if args.command == "train":
        train(args.config)
    elif args.command == "resume":
        resume()
    elif args.command == "evaluate":
        evaluate(args.model, args.data, args.config)
    elif args.command == "plot":
        plot_evaluation(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
