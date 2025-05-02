from functions import getState

class TradingEnv:
    def __init__(self, data, window_size, opportunity_cost, double_bet_penalty, wait_too_long_penalty, sell_bonus):
        self.data = data
        self.window_size = window_size
        self.opportunity_cost = opportunity_cost
        self.double_bet_penalty = double_bet_penalty
        self.wait_too_long_penalty = wait_too_long_penalty
        self.sell_bonus = sell_bonus
        self.total_steps = len(data)

    def cap_reward(self, reward):
        if reward >= 0:
            return min(reward, 1)
        else:
            return max(reward, -1)

    def step(self, combined_state, action, t):
        *state, position = combined_state
        hold, long, short, t_taken_percentage = position
        hold = int(hold)
        long = int(long)
        short = int(short)
        
        is_idle = True if long == 0 and short == 0 else False
        t_taken = round(t_taken_percentage * self.window_size + t - self.window_size)
        is_in_position_too_long = t_taken_percentage <= 1 / self.window_size
        next_portfolio = [hold, long, short, t_taken_percentage - (1 / self.window_size)]
        reward = 0
        info = {
            "bet_violation": 0,
            "hold_violation": 0,
            "profit": 0,
            "start": 0,
            "end": 0
        }

        if action == 0:  # hold
            if is_idle:
                if is_in_position_too_long:
                    reward = -self.opportunity_cost + -self.wait_too_long_penalty
                    next_portfolio = [1, 0, 0, (self.window_size - 1)/self.window_size]
                    info["hold_violation"] += 1
                else:
                    reward = -self.opportunity_cost 
            else: # in active open position
                if is_in_position_too_long:
                    reward = -self.wait_too_long_penalty
                    next_portfolio = [1, 0, 0, (self.window_size-1)/self.window_size]
                    info["hold_violation"] += 1
                else:
                    now = self.data["Close"][t]
                    then = self.data["Close"][t_taken]
                    gain = now - then
                    gain = gain if long == 1 else -gain
                    reward = (gain / then)
        elif action == 1 or action == 2:  # long/short
            if is_idle:  # valid bet
                next_portfolio = [0, 1 if action == 1 else 0, 1 if action == 2 else 0, (self.window_size - 1) / self.window_size]
            elif is_in_position_too_long:  # bet at the end of window
                reward = -self.wait_too_long_penalty
                next_portfolio = [1, 0, 0, 0]
                info["hold_violation"] += 1
            else:  # double bet
                reward = -self.double_bet_penalty
                info["bet_violation"] += 1
        else:  # sell
            if is_idle:  # sell without open position
                reward = -self.double_bet_penalty
                info["bet_violation"] += 1
            else:  # valid sell
                now = self.data["Close"][t]
                then = self.data["Close"][t_taken]
                gain = (now - then) * (1 if long == 1 else -1)
                reward = gain / then + self.sell_bonus
                next_portfolio = [1, 0, 0, (self.window_size - 1) / self.window_size]
                info["start"] = t_taken
                info["end"] = t
                info["profit"] = gain

        reward = self.cap_reward(reward)

        done = True if t == self.total_steps - 1 else False

        return (getState(self.data, t + 1, self.window_size, next_portfolio), reward, done, info)