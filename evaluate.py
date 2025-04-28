from agent.agent import Agent
from functions import *
import re 
from env import TradingEnv
import logging

logger = logging.getLogger("q-trader")

def evaluate(agent: Agent, env:TradingEnv, data, window_size):
	action_map = {0: "Hold", 1: "Long", 2: "Short", 3: "Sell"}
	l = len(data) - 1

	state = getState(data, 0, window_size)
	total_profit = 0
	bet_violation = 0
	hold_violation = 0
	total_reward = 0
	transactions = []
	for t in range(l):
		action = agent.act(state)
		next_state, reward, done, info = env.step(state, action, t)
		bet_violation += info["bet_violation"]
		hold_violation += info["hold_violation"]
		total_profit += info["profit"]
		total_reward += reward
		if action != 0:
			transactions.append({
				"action": action_map[int(action)],
				"time": t,
				"price": data["Close"][t],
			})
		
		# if action != 0:
		# 	print(f"\n{action_map[action]} at t{t}")
		# 	print(f"Next portfolio: {[long, short, t_taken]}")
		# 	if action == 3:
		# 		print(f"Profit: {profit}")
		# 		*_, old_long, old_short, t_wanted, _, __ = state
		# 		t_taken_old = round(t_wanted * window_size + t - window_size)
		# 		now = data["Close"][t]
		# 		then = data["Close"][t_taken_old]
		# 		actual_profit = (now - then) * (1 if old_long else -1)
		# 		print(f"Actual profit: ${formatPrice(now)} - {formatPrice(then)} = {formatPrice(actual_profit)}")
		# else:
		# 	print(f"Hold at t{t} {formatPrice(data["Close"][t])}")
		state = next_state

	return total_profit, transactions, bet_violation, hold_violation

def get_model_list(model_names):
	p = r"model_ep(\d+)\.keras"
	
	model_list = []
	for name in model_names:
		m = re.search(p, name)
		if m:
			model_list.append(int(m.group(1)))
	model_list.sort()
	return model_list
