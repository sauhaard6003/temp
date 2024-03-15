import numpy as np
import random
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.replay_memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_freq = 10
        self.target_update_counter = 0

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def choose_actions(self, state):
        return np.argsort(self.model.predict(state)[0])[-5:][::-1]  # Always return top 5 documents

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.replay_memory) < self.batch_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            states.append(state[0])
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.target_update_counter > self.update_target_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        else:
            self.target_update_counter += 1
            
def f(word):
    pass

def main():
    dict = {}
    
    n_documents = 100


    action_size = n_documents
    state_size = n_documents + 1  # Size of state: 1 for search word + n_documents for relevance scores

    agent = DQNAgent(state_size, action_size)

    while True:
        # Prompt the user to input the search word
        search_word = input("Enter the search word (or 'quit' to exit): ")
        if search_word == 'quit':
            break

        # Check if the search word exists in the relevance_scores dictionary
        if search_word not in dict:
            dict.add(search_word)

        scores = f(search_word)

        # Enter the training loop for the selected search word
        while True:
            # Prompt the user to input the search word
            search_word = input("Enter the search word (or 'quit' to exit): ")
            if search_word == 'quit':
                break

            scores = f(search_word)  

            # Train the agent for the selected search word
            state = np.concatenate(([search_word], scores))
            actions = agent.choose_actions(state)
            x = scores[0] - scores[99]

            # Prompt the user to provide relevance feedback for the top 5 documents
            for action in actions:
                relevance_feedback = int(input(f"Is document {action} relevant for '{search_word}'? (1 for yes, 0 for no): "))

                # Update relevance scores based on user feedback
                if relevance_feedback == 1:
                    scores[action] += x  
                else:
                    scores[action] -= x  

            next_state = np.concatenate(([search_word], scores))
            for i, action in enumerate(actions):
                reward = scores[action] 
                done = False  
                agent.remember(state, action, reward, next_state, done)

            agent.train_model()

if __name__ == "__main__":
    main()