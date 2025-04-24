from agent.agent import Agent
from functions import *
import sys
import logging
import psutil
import os
from logging.handlers import RotatingFileHandler
import numpy as np
import time

# Set up logger with RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('training.log', maxBytes=100*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if len(sys.argv) < 4:
	print("Usage: python train.py [stock] [window] [episodes] [save_step] - default 100 [memory_size] default 25000")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
save_step = int(sys.argv[4]) if len(sys.argv) > 4 else 100
memory_size = int(sys.argv[5]) if len(sys.argv) > 5 else 25000
agent = Agent(state_size=window_size, is_eval=False, model_name="", memory_size=memory_size)

data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
OPPORTUNITY_COST = 2.75/100
DOUBLE_BET_PENALTY = 1
WAIT_TOO_LONG_PENALTY = OPPORTUNITY_COST * 10
SELL_BONUS = 0.001 

logger.info("--------------------------------")
logger.info(f"REPLAY_BUFFER_SIZE: {memory_size}")
logger.info(f"OPPORTUNITY_COST: {OPPORTUNITY_COST}")
logger.info(f"DOUBLE_BET_PENALTY: {DOUBLE_BET_PENALTY}")
logger.info(f"WAIT_TOO_LONG_PENALTY: {WAIT_TOO_LONG_PENALTY}")
logger.info(f"SELL_BONUS: {SELL_BONUS}")
logger.info(f"SAVE MODEL EACH: {save_step} steps")
logger.info("--------------------------------")


def cap_reward(reward):
	if reward >= 0:
		return min(reward, 1)
	else:
		return max(reward, -1)

def step(combined_state, action, t):
	global window_size
	global l 

	*state, position = combined_state
	hold, long, short, t_taken_percentage = position
	is_idle = True if long == 0 and short == 0 else False
	t_taken = round(t_taken_percentage * window_size + t - window_size)
	is_in_position_too_long = t_taken_percentage <= 1/window_size 
	next_portfolio = [hold, long, short, t_taken_percentage - (1/window_size) if not is_idle else 0]
	reward = 0
	info = {
		"bet_violation": 0,
		"hold_violation": 0
	}

	if action == 0: #hold
		if is_idle: # hold without any open position
			reward = cap_reward(-OPPORTUNITY_COST)
			next_portfolio = [1, 0, 0, 0]
		elif is_in_position_too_long: #hold when position is too long
			reward = -WAIT_TOO_LONG_PENALTY
			next_portfolio = [1, 0, 0, 0]
			info["hold_violation"] += 1
		else:
			now = data["Close"][t]
			then = data["Close"][t_taken]
			gain = now - then
			gain = gain if long == 1 else -gain
			reward = (gain / then)
			
			info["profit"] = gain
			info["start"] = t_taken 
			info["end"] = t
	elif action == 1 or action == 2: #long/short 
		if is_idle: #valid bet
			next_portfolio = [0, 1 if action == 1 else 0, 1 if action == 2 else 0, (window_size-1)/window_size]
		elif is_in_position_too_long: #bet at the end of window
			reward = -WAIT_TOO_LONG_PENALTY
			next_portfolio = [1, 0, 0, 0]
			info["hold_violation"] += 1
		else: #double bet
			reward = -DOUBLE_BET_PENALTY
			info["bet_violation"] += 1
	else: #sell 
		if is_idle: # sell without open position
			reward = -DOUBLE_BET_PENALTY
			info["bet_violation"] += 1
			next_portfolio = [1, 0, 0, 0]
		else: #valid sell
			now = data["Close"][t] 
			then = data["Close"][t_taken]
			gain = (now - then) * (1 if long == 1 else -1)
			reward = gain / then + SELL_BONUS
			next_portfolio = [1, 0, 0, 0]
			info["start"] = t_taken
			info["end"] = t
			info["profit"] = gain

	reward = cap_reward(reward)

	done = t == l - 1

	return (getState(data, t + 1, window_size, next_portfolio), reward, done, info)

for e in range(episode_count + 1):
	# Start the timer
	start_time = time.time()
	logger.info("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size)
	agent.reset()
	agent.decay_epsilon()

	total_profit = 0
	total_bet_violation = 0
	total_hold_violation = 0
	loss_list = []	

	for t in range(l):
		action = agent.act(state)

		next_state, reward, done, info = step(state, action, t)
		*_, portfolio= next_state
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
