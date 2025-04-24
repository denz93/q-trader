import tensorflow as tf
from keras.layers import LSTM
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import time 

class Agent:
	def __init__(self, state_size, is_eval=False, model_name="", memory_size=25000):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 # hold, long, short, sell
		self.memory = deque(maxlen=memory_size)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval
		self.predict_times = []
		self.train_times = []

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.965 # ~130 episodes to epsilon_min
		self.model = load_model("models/" + model_name, custom_objects={'mse': keras.losses.MeanSquaredError()}) if is_eval else self._model()

	def _model(self):
		ohlcv_input = Input(shape=(self.state_size, 5), name="ohlcv")
		lstm_out = LSTM(64)(ohlcv_input)

		position_input = Input(shape=(4,), name="position")
		combined = Concatenate()([lstm_out, position_input])
		dense = Dense(units=64, activation="relu")(combined)
		dense = Dropout(0.2)(dense)
		dense = Dense(units=32, activation="relu")(dense)
		output = Dense(units=self.action_size, activation="linear")(dense)

		model = keras.Model(inputs=[ohlcv_input, position_input], outputs=output)
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
		model.summary()
		
		return model

	def act(self, state, force=False):
		if not self.is_eval and np.random.rand() <= self.epsilon and not force:
			return random.randrange(self.action_size)
		options = self.predict(state)
		return np.argmax(options[0])
	
	def to_model_input(self, state):
		ohlcv, position = state
		return {
			"ohlcv": np.array([ohlcv]),
			"position": np.array([position])
		}
	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		mini_batch = random.sample(self.memory, batch_size)
		loss_list = []

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				next_actions = self.predict(next_state)
				target = reward + self.gamma * np.amax(next_actions[0])

			target_f = self.predict(state)
			target_f[0][action] = target
			start = time.time()
			history = self.model.fit(self.to_model_input(state), target_f, epochs=1, verbose=0)
			end = time.time()
			elapsed = end - start 
			self.train_times.append(elapsed)
			loss_list.append(history.history["loss"][0])

		return np.average(loss_list)
	
	def reset(self):
		self.predict_times = []
		self.train_times = []

	def get_avg_predict_time(self):
		return np.average(self.predict_times)
	
	def get_avg_train_time(self):
		return np.average(self.train_times)
	
	def decay_epsilon(self):
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

	def predict(self, state):
		start = time.time()
		actions = self.model.predict(self.to_model_input(state), verbose=0)
		end = time.time()
		elapsed = end - start
		self.predict_times.append(elapsed)
		return actions

