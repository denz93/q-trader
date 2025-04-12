from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
OPPORTUNITY_COST = 2.75/100
DOUBLE_BET_PENALTY = 1000 

def step(combined_state, action, t):
	global window_size
	global l 

	*state, portfolio = combined_state
	long, short, t_taken_percentage, _, __ = portfolio
	is_idle = long == 0 and short == 0
	t_taken = int(t_taken_percentage * window_size + t - window_size)
	is_in_position_too_long = t_taken_percentage <= 0 
	next_portfolio = [long, short, t_taken_percentage - 1/window_size if not is_idle else 0, 0, 0]
	reward = 0
	info = {}

	if action == 0: #hold
		if is_idle:
			reward = -OPPORTUNITY_COST
		elif is_in_position_too_long: #force sell
			now = data["Close"][t] 
			then = data["Close"][t_taken]
			reward = (now - then) / then 
			next_portfolio = [0, 0, 0, 0, 0]
		else:
			gain = data["Close"][t] - data["Close"][t_taken]
			gain = gain if long == 1 else -gain
			ith_taken = t - t_taken
			reward = gain * (0.5 ** ith_taken)
			info["profit"] = reward
			info["start"] = t_taken 
			info["end"] = t
	elif action == 1 or action == 2: #long/short 
		if is_idle:
			next_portfolio = [1 if action == 1 else 0, 1 if action == 2 else 0, (window_size-1)/window_size, 0, 0]
		else:
			reward = -DOUBLE_BET_PENALTY
	else: #sell 
		if is_idle:
			reward = -DOUBLE_BET_PENALTY
		else:
			now = data["Close"][t] 
			then = data["Close"][t_taken]
			reward = (now - then) * (1 if long == 1 else -1)
			next_portfolio = [0, 0, 0, 0, 0]
			info = {
				"start": then,
				"end": now,
				"profit": reward,
			}

	done = t == l - 1

	return (getState(data, t + 1, window_size, next_portfolio), reward, done, info)

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		next_state, reward, done, info = step(state, action, t)
		*_, portfolio = next_state

		if action == 0: 
			print(f"Hold at t{t}| {formatPrice(data['Close'][t])}")
		elif action == 1:
			print(f"Long at t{t} | {formatPrice(data['Close'][t])}")
		elif action == 2:
			print(f"Short at t{t} | {formatPrice(data['Close'][t])}")
		elif action == 3:
			print(f"Sell at t{t} | {formatPrice(data['Close'][t])}")
			if "profit" in info:
				print(f"Profit: {formatPrice(info['profit'])} | Start: {info['start']} | End: {info['end']}")
				total_profit += info["profit"]
		print(f"Reward: {reward}")
		print(f"Portfolio: {portfolio[:3]}")
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
