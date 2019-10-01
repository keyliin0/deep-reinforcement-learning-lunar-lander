import gym
import numpy as np
from model import DQNmodel
from collections import deque
import random

epsilon = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001


env = gym.make('LunarLander-v2')
env.seed(42)
DQNagent = DQNmodel(8,4)
# store score of last 100 episodes
scores = deque(maxlen=100)

for episode in range(1, 3000):
    score = 0
    current_state = env.reset()
    done = False
    for _ in range(1000):
        if np.random.random() > epsilon:
            action = np.argmax(DQNagent.get_Q_value(current_state))
        else:
            action = random.choice(np.arange(4))
        new_state, reward, done, info = env.step(action)
        # append (state , action , reward , done , new_state)
        DQNagent.append_to_memory((current_state, action, reward, done, new_state))
        score += reward
        current_state = new_state
        # train the model
        DQNagent.train()
        if done:
            break
    scores.append(score)
    if (episode % 100 == 0):
        # print average score of the last 100 episodes
        print("episode " + str(episode) + " : " + str(np.mean(scores)))
        DQNagent.save(episode)
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()