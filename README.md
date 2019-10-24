
Train a neural network using deep [Q learning](https://en.wikipedia.org/wiki/Q-learning).

## Before training 
![alt text](https://github.com/keyliin0/deep-reinforcement-learning-lunar-lander/blob/master/demo1.gif?raw=true)

## After training 
![alt text](https://github.com/keyliin0/deep-reinforcement-learning-lunar-lander/blob/master/demo2.gif?raw=true)

## Score
The mean for the last 100 episodes

![alt text](https://raw.githubusercontent.com/keyliin0/deep-reinforcement-learning-lunar-lander/master/score.jpg)



## The model

3 hidden layers with 64 neurons in each layer using relu as an activation function

The output layer contains 4 (Q value for each action) neurons

## Usage

### train the model
```
python train.py
```

### test the model

Change my_model_1900.hdf5 to your model file name in test.py then run

```
python test.py
```
