from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random
from keras.optimizers import Adam
import numpy as np

class DQNmodel:
  def __init__(self, input_size, output_size, model_file_name = False):
    self.model = self.create_model(input_size, output_size)
    self.target_model = self.create_model(input_size, output_size)
    self.target_model.set_weights(self.model.get_weights())
    # load a pretrained model
    if(model_file_name is not False):
        self.model.load_weights(model_file_name)
        self.target_model.load_weights(model_file_name)
    self.game_memory = deque(maxlen=100_000)
    self.batch_size = 64
    self.discount = 0.99
    self.train_count = 0
  def create_model(self, input_size, output_size):
    # create the model
    model = Sequential()
    model.add(Dense(output_dim=64, init='uniform', input_dim=input_size, activation='relu'))
    model.add(Dense(output_dim=64, init='uniform', activation='relu'))
    model.add(Dense(output_dim=64, init='uniform', activation='relu'))
    model.add(Dense(output_dim=output_size, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['accuracy'])
    return model
  def append_to_memory(self, data):
    # append (state , action , reward , done , new_state)
    self.game_memory.append(data)
  def clear_game_memory(self):
    self.game_memory.clear()
  def get_Q_value(self, state):
    return self.model.predict(np.array([state]))
  def train(self):
    # train the model every 5 moves
    self.train_count += 1
    if(len(self.game_memory) < self.batch_size or self.train_count < 5):
      return
    self.train_count = 0

    # get random samples to train on
    samples = random.sample(self.game_memory, self.batch_size)

    current_states = np.array([data[0] for data in samples])
    current_state_qs = self.model.predict(current_states)

    new_states = np.array([data[4] for data in samples])
    new_states_qs = self.target_model.predict(new_states)

    X_train = []
    Y_train = []

    for index, (current_state, action, reward, done, new_state) in enumerate(samples):
      # get new q value
      if not done:
        max_q = np.max(new_states_qs[index])
        new_q = reward + self.discount * max_q
      else:
        new_q = reward

      # update the q value for the current action
      target_qs = current_state_qs[index]
      target_qs[action] = new_q
      X_train.append(current_state)
      Y_train.append(target_qs)
    self.model.fit(np.array(X_train),np.array(Y_train), shuffle=False,verbose=False)
    self.target_model.set_weights(self.model.get_weights())
  def save(self, episode):
    self.target_model.save("my_model_"+str(episode)+".hdf5")