import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 # hold, long, short, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = load_model("models/" + model_name, custom_objects={'mse': keras.losses.MeanSquaredError()}) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=128, input_shape=((self.state_size+1)*5,), activation="gelu"))
		model.add(Dense(units=64, activation="gelu"))
		model.add(Dense(units=32, activation="gelu"))
		model.add(Dense(units=8, activation="gelu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.002))
		model.summary()
		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(np.array([state]), verbose=0)
		print(options)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])

			target_f = self.model.predict(np.array([state]), verbose=0)
			#print(target_f)
			#print(f"action: {action}")
			#exit()
			target_f[0][action] = target
			history = self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
			return history.history["loss"][0]

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
