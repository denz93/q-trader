from agent.agent import Agent
from functions import *
import sys
import logging
import psutil
import os
from logging.handlers import RotatingFileHandler
import numpy as np


# Set up logger with RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('training.log', maxBytes=100*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if len(sys.argv) < 4:
	logger.info("Usage: python train.py [stock] [window] [episodes] [save_step] - default 100 [memory_size] default 25000")
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
WAIT_TOO_LONG_PENALTY = DOUBLE_BET_PENALTY

print(f"Double bet penalty: {DOUBLE_BET_PENALTY}")

def step(combined_state, action, t):
	global window_size
	global l 

	*state, position = combined_state
	hold, long, short, t_taken_percentage = position
	is_idle = True if long == 0 and short == 0 else False
	t_taken = round(t_taken_percentage * window_size + t - window_size)
	is_in_position_too_long = t_taken_percentage <= 0 
	next_portfolio = [hold, long, short, t_taken_percentage - (1/window_size) if not is_idle else 0]
	reward = 0
	info = {}

	if action == 0: #hold
		if is_idle:
			reward = -OPPORTUNITY_COST
			next_portfolio = [1, 0, 0, 0]
		elif is_in_position_too_long: #force sell
			reward = -WAIT_TOO_LONG_PENALTY
			next_portfolio = [1, 0, 0, 0]
		else:
			now = data["Close"][t]
			then = data["Close"][t_taken]
			gain = now - then
			gain = gain if long == 1 else -gain
			ith_taken = t - t_taken + 1
			reward = (gain / then) * (0.95 ** ith_taken)
			info["profit"] = gain
			info["start"] = t_taken 
			info["end"] = t
	elif action == 1 or action == 2: #long/short 
		if is_idle:
			next_portfolio = [0, 1 if action == 1 else 0, 1 if action == 2 else 0, (window_size-1)/window_size]
		else:
			reward = -DOUBLE_BET_PENALTY
	else: #sell 
		if is_idle:
			reward = -DOUBLE_BET_PENALTY
		else:
			now = data["Close"][t] 
			then = data["Close"][t_taken]
			gain = (now - then) * (1 if long == 1 else -1)
			reward = gain / then
			next_portfolio = [1, 0, 0, 0]
			info = {
				"start": t_taken,
				"end": t,
				"profit": gain,
			}

	done = t == l - 1

	return (getState(data, t + 1, window_size, next_portfolio), reward, done, info)

for e in range(episode_count + 1):
	logger.info("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size)

	total_profit = 0
	loss_list = []	
	for t in range(l):
		action = agent.act(state)

		next_state, reward, done, info = step(state, action, t)
		*_, portfolio= next_state
		hold, long, short, t_taken = portfolio

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

		if done:
			logger.info("--------------------------------")
			logger.info("Total Profit: " + formatPrice(total_profit))
			logger.info("--------------------------------")

		if len(agent.memory) > batch_size:
			loss = agent.expReplay(batch_size)
			loss_list.append(loss)
	logger.info(f"Loss: {loss[len(loss_list) - 5:]}")

	if e % save_step == 0 and e > 0:
		agent.model.save(f"models/model_ep{e}.keras")
