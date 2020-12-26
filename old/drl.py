# import shelve
import time
# from utils import *
from network import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import regularizers
# from keras.utils import plot_model
# from keras.utils import multi_gpu_model
from collections import deque
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import random
from keras.models import load_model

from main import reward_fn

# random.seed(3)

REPLAY_MEMORY_SIZE = 3000
MODEL_NAME = reward_fn #
MIN_REPLAY_MEMORY_SIZE =  2_000
MINIBATCH_SIZE =  64
DISCOUNT = 0 #0.9
UPDATE_TARGET_EVERY = 5 # weight transfer after every this many steps

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# set_gpu() # biplav
class DLModel:
    def __init__(self, max_users, action_size):
        # main model, gets trained every step. crazy model
        self.model = self.create_model(max_users, action_size)

        # target model, which will be .predict() every step. stable model
        self.target_model= self.create_model(max_users, action_size)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

       # Custom tensorboard object
        self.Tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self, max_users, action_size):
        first_layer = 2*max_users # states = age at BS and UAV for each user
        last_layer = action_size # actions = combination of updating and sampling capacity number of users in both directions
        self.layers_units = [first_layer, 32, 16, last_layer]
        print(f'NN with {self.layers_units} structure has been created')
        model = Sequential()
        model.add(
                  Dense(units=self.layers_units[1],
                  activation='relu',
                  input_dim=self.layers_units[0],
                  name='ds1')
                 )
        model.add(Dropout(rate=0.5))
        
        model.add(
                  Dense(units=self.layers_units[2],
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  name='ds2')
                  )

        model.add(Dropout(rate=0.5))

        model.add(Dense(units=self.layers_units[3],
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  name='out')
                  )
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])


        # if is_gpu_available(): # biplav
        #     gpus = get_available_gpus()
        #     model = multi_gpu_model(model, gpus)
        # return model

    def load_best_model(self): 
        '''Load the current best model, not used currently'''
        models = os.listdir(Tools.qf_model)
        if len(models) != 0:
            tim = []
            for i in range(len(models)):
                tim.append(int(models[i].split('_')[1]))
            best_model = self.build_model() # builds the default NN
            best_model.load_weights(Tools.qf_model + '/weights_' + str(np.max(tim)) + '_.h5')
            best_model.compile(
                optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                loss=Tools.adjust_loss,
                metrics=[Tools.my_metrics]
            )
            return best_model
        else:
            print('No model currently...')
            return None

    def is_gpu_available():
        is_gpu = tf.test.is_gpu_available(True)
        return is_gpu

     # transition is the new state and other information after taking an action, save to replay memory
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state) [0] is state info, the output of prediction is the q-value ??
    def get_qs(self, state):
        # print("Inside get_qs, state is ", state)

        model_output = self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
        # print(f'inside get_qs, model_output = {model_output}')
        return model_output # predict always returns a list of list so get the first element

    def train(self, terminal_state, step):
        # print(f'terminal state is {terminal_state}')
        # coming true for the last step before the episode ends

        # not enough data to do a training step
        if (len(self.replay_memory)) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print(f'size(minibatch) - {np.shape(minibatch)}') # minibatchsize * 5, for 5 see update_replay_memory
        # print(f'minibatch - {minibatch}')
        # minibatch is sampled from replay_memory, which contains transitions. and tranistion[0] is the state

        # transition format -> (current_state, action, reward, new_state, done)
        # print(f'step = {step}')

        current_states = np.array([transition[0] for transition in minibatch])  # shape = minibatch_size*NN_first_layer
        # all current_state in the minibatch that will be fed together
        # print(f'current_states = {current_states}')
        # print(f'np.shape(current_states) = {np.shape(current_states)}')
        
        current_qs_list = self.model.predict(current_states) # shape = minibatch_size*NN_last_layer
        # print(f'current_qs_list = {current_qs_list}') 
        # print(f'np.shape(current_qs_list) = {np.shape(current_qs_list)}')
        # predict by the crazy model

        new_current_states = np.array([transition[3] for transition in minibatch]) # shape = minibatch_size*NN_first_layer
        # all new_current_state in the minibatch, this is like the actual state seen by the model and current_qs_list is like the state predicted by the model 
        # print(f'new_current_states = {new_current_states}')
        # print(f'np.shape(new_current_states) = {np.shape(new_current_states)}') 

        future_qs_list = self.target_model.predict(new_current_states) # shape = minibatch_size*NN_last_layer
        # print(f'future_qs_list = {future_qs_list}') 
        # print(f'np.shape(future_qs_list) = {np.shape(future_qs_list)}')
        # current_qs_list and future_qs_list are same sized arrays that store state and next state information, but are generated by 2 different models

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # print(f'index={index}, current_state={current_state}, action={action}, reward={reward}, new_current_state={new_current_state}, done={done}')
        # this is how information is stored in minibatch
        
        # prepare the X and y for feeding into the NN
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            
            else:
                new_q = reward # terminal step

            current_qs = current_qs_list[index]
            current_qs[action] = new_q # all other Qs remain same except the chosen action's Q which has to be updated based on the reward

            # both X and y are the same dimension as minibatch size
            # now both current_state and current_qs belong to the same index, and are added as a training point
            X.append(current_state)
            y.append(current_qs)

        # print(f'shape(np.array(X))- {np.shape(np.array(X))}, shape(np.array(y))- {np.shape(np.array(y))}')
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.Tensorboard] if terminal_state else None)


        # check if time to update target model yet, only after having enough data. working properly
        if terminal_state:
            # print(f'terminal_state = {terminal_state}, target_update_counter =  {self.target_update_counter}')
            self.target_update_counter += 1

        
        # working properly
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            # print(f'weights updated')
            self.target_model.set_weights((self.model.get_weights()))
            self.target_update_counter = 0

    

