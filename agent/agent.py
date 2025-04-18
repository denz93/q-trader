import tensorflow as tf
from keras.layers import LSTM
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 # hold, long, short, sell
		self.memory = deque(maxlen=25000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = load_model("models/" + model_name, custom_objects={'mse': keras.losses.MeanSquaredError()}) if is_eval else self._model()

	def _model(self):
		ohlcv_input = Input(shape=(self.state_size, 5), name="ohlcv")
		lstm_out = LSTM(64)(ohlcv_input)
		
		position_input = Input(shape=(4,), name="position")
		combined = Concatenate()([lstm_out, position_input])
		dense = Dense(units=32, activation="relu")(combined)
		output = Dense(units=self.action_size, activation="linear")(dense)

		model = keras.Model(inputs=[ohlcv_input, position_input], outputs=output)
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
		model.summary()
		
		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		options = self.model.predict(self.to_model_input(state), verbose=0)
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

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				next_actions = self.model.predict(self.to_model_input(next_state), verbose=0)
				target = reward + self.gamma * np.amax(next_actions[0])

			target_f = self.model.predict(self.to_model_input(state), verbose=0)
			target_f[0][action] = target
			history = self.model.fit(self.to_model_input(state), target_f, epochs=1, verbose=0)
			return history.history["loss"][0]

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
