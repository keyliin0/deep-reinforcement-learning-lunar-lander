import gym
import numpy as np
from model import DQNmodel

env = gym.make('LunarLander-v2')
env.seed(42)
DQNagent = DQNmodel(8,4,"my_model_1900.hdf5")
current_state = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(DQNagent.get_Q_value(current_state))
    new_state, reward, done, info = env.step(action)
    current_state = new_state
    env.render()
env.close()
