import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
from agent.agent import Agent
from functions import *
import sys
import os 
import re 
last_n = int(sys.argv[1])
last_n = 1000 if last_n <= 0 else last_n
data = getStockDataVec("^GSPC_2011")
train_data = getStockDataVec("^GSPC")
window_size = 60

def step(data, combined_state, action, t, window_size):

	*_,  portfolio = combined_state
	hold, long, short, t_taken_percentage = portfolio
	is_idle = hold
	t_taken = round(t_taken_percentage * window_size + t - window_size)
	profit = 0 
	double_bet = 0
	hold_too_long = 0
	next_portfolio = [hold, long, short, t_taken_percentage - (1/window_size) if not is_idle else 0]
	is_wait_too_long = t_taken_percentage < 0

	if is_wait_too_long:
		hold_too_long += 1
		next_portfolio = [1, 0, 0, 0]
	else:
		if action == 1 or action == 2:
			if is_idle:
				next_portfolio = [0, 1 if action == 1 else 0, 1 if action == 2 else 0, (window_size-1)/window_size]
			else:
				double_bet += 1
		elif action == 3: #sell 
			if not is_idle:
				now = data["Close"][t] 
				then = data["Close"][t_taken]
				profit = (now - then) * (1 if long == 1 else -1)
				next_portfolio = [1, 0, 0, 0]
			else:
				double_bet += 1
	next_state = getState(data, t+1, window_size, next_portfolio)		
	return next_state, profit, double_bet, hold_too_long
def evaluate(model_name, data):
	global window_size
	action_map = {0: "Hold", 1: "Long", 2: "Short", 3: "Sell"}
	agent = Agent(window_size, True, model_name)
	#window_size = agent.model.layers[0].input.shape.as_list()[1]
	l = len(data) - 1
	batch_size = 32

	state = getState(data, 0, window_size)
	total_profit = 0
	bet_violation = 0
	hold_violation = 0

	for t in range(l):
		action = agent.act(state)
		next_state, profit, double_bet, hold_too_long = step(data,state, action, t, window_size)
		*_, portfolio = next_state
		hold, long, short, t_taken = portfolio
		bet_violation += double_bet
		hold_violation += hold_too_long
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
		total_profit += profit
		state = next_state

	return total_profit, bet_violation, hold_violation

def get_model_list(model_names):
	p = r"model_ep(\d+)\.keras"
	
	model_list = []
	for name in model_names:
		m = re.search(p, name)
		if m:
			model_list.append(int(m.group(1)))
	model_list.sort()
	return model_list

def main():
	global data 
	global train_data
	model_file_names = os.listdir("models/")
	model_ord_list = get_model_list(model_file_names)
	evaluate_models = model_ord_list[len(model_ord_list) - last_n:]
	print(evaluate_models)
	profit_list = []
	for dataset_name, dataset in zip(["train", "test"], [train_data, data]):
		print("======================>")
		print(f"Evaluating {dataset_name} dataset")
		for model_ord in evaluate_models:
			model_name = f"model_ep{model_ord}.keras"
			profit, bet_violation, hold_violation = evaluate(model_name, dataset)
			print("-----------------")
			print(f"Model: {model_name}")
			print(f"Profit: {formatPrice(profit)}")
			print(f"Bet violation: {bet_violation}")
			print(f"Hold violation: {hold_violation}")
			print("-----------------")
		print("<======================")
main()