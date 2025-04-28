from agent.agent import Agent
from functions import *
from env import TradingEnv
import logging
import time
logger = logging.getLogger("q-trader")
def train_agent(agent: Agent, env, data, window_size, episode_count, save_step, batch_size, metadata):
    """
    Train the trading agent in the given environment.

    Args:
        agent (Agent): The trading agent.
        env (TradingEnv): The trading environment.
        data (pd.DataFrame): The stock data.
        window_size (int): The size of the observation window.
        episode_count (int): The number of episodes to train.
        save_step (int): The interval for saving the model.
        batch_size (int): The batch size for experience replay.
        logger (logging.Logger): The logger for logging training progress.
        metadata (dict): Metadata to track training progress.
    """
    for e in range(metadata["current_episode"], episode_count + 1):
        metadata["current_episode"] = e  # Update current episode
        # Start the timer
        start_time = time.time()
        logger.info("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size)
        agent.reset()

        total_profit = 0
        total_bet_violation = 0
        total_hold_violation = 0
        loss_list = []
        for t in range(env.total_steps):
            action = agent.act(state)

            next_state, reward, done, info = env.step(state, action, t)
            *_, portfolio = next_state
            hold, long, short, t_taken = portfolio

            total_bet_violation += info["bet_violation"]
            total_hold_violation += info["hold_violation"]

            if action == 0:
                logger.debug(f"Hold at t{t}| {formatPrice(data['Close'][t])}")
                if "profit" in info:
                    logger.debug(f"Profit: {formatPrice(info['profit'])} | Start: {info['start']} | End: {info['end']}")
                    total_profit += info["profit"]
            elif action == 1:
                logger.debug(f"Long at t{t} | {formatPrice(data['Close'][t])}")
            elif action == 2:
                logger.debug(f"Short at t{t} | {formatPrice(data['Close'][t])}")
            elif action == 3:
                logger.debug(f"Sell at t{t} | {formatPrice(data['Close'][t])}")
                if "profit" in info:
                    logger.debug(f"Profit: {formatPrice(info['profit'])} | Start: {info['start']} | End: {info['end']}")
                    total_profit += info["profit"]
            logger.debug(f"Reward: {reward}")
            logger.debug(f"Portfolio: {[hold, long, short, t_taken]}")
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(agent.memory) > batch_size:
                loss = agent.expReplay(batch_size)
                loss_list.append(loss)
                agent.decay_epsilon()
                logger.debug(f"Train time: {agent.get_avg_train_time()}")

        # End the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        logger.info("--------------------------------")
        logger.info("Total Profit: " + formatPrice(total_profit))
        logger.info(f"Loss: {loss_list[len(loss_list) - 5:]}")
        logger.info(f"Bet violations: {total_bet_violation} | Hold violations: {total_hold_violation}")
        logger.info(f"Greedy Epsilon: {agent.epsilon}")
        logger.info(f"Average predict time: {agent.get_avg_predict_time():.2f}")
        logger.info(f"Average train time: {agent.get_avg_train_time():.2f}")
        logger.info(f"Episode execution time: {elapsed_time:.2f} seconds")
        logger.info("--------------------------------")
        if e % save_step == 0 and e > 0:
            agent.model.save(f"models/model_ep{e}.keras")
