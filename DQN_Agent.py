from collections import deque
import numpy as np
import random
import tensorflow as tf

class DQN_Agent:
    def __init__(self, state_size, action_size, initial_cash=10000, train_interval=10):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.portfolio_value = initial_cash  # Track portfolio value
        self.cash = initial_cash  # Cash available for trading
        self.stock_position = 0  # Number of stocks owned
        self.train_interval = train_interval  # Interval for training the model
        self.steps_since_last_training = 0  # To track steps since last training

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Only replay periodically
        if self.steps_since_last_training >= self.train_interval:
            if len(self.memory) > batch_size:
                minibatch = random.sample(self.memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = (reward + self.gamma * 
                                  np.amax(self.model.predict(next_state)[0]))
                    target_f = self.model.predict(state)
                    target_f[0][action] = target
                    self.model.fit(state, target_f, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                print(f"Model updated at step {self.steps_since_last_training}")
            self.steps_since_last_training = 0  # Reset the training step counter
        else:
            self.steps_since_last_training += 1  # Increment the counter

    def get_position(self):
        return self.stock_position

    def update_portfolio(self, action, price):
        """
        Updates portfolio value based on action taken (buy/sell/hold).
        """
        if action == 0:  # Buy action
            # Buy stock with all available cash
            quantity = self.cash // price
            self.stock_position += quantity
            self.cash -= quantity * price
        elif action == 1:  # Sell action
            # Sell all owned stocks
            self.cash += self.stock_position * price
            self.stock_position = 0
        # Portfolio value is the sum of cash and the value of stocks held
        self.portfolio_value = self.cash + self.stock_position * price
        print(f"Portfolio updated: Cash = {self.cash}, Stock Position = {self.stock_position}, Portfolio Value = {self.portfolio_value}")
