from PIL.Image import new
from sentdex_tutorial import DISCOUNT, LEARNING_RATE
import numpy as np 
from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 
import pickle 
from matplotlib import style 
import time 

style.use("ggplot")

SIZE = 10 # observation space 10*10 grid where player, enemy and food will be in the grid. this the arena
HM_EPISODES = 25_000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300 
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998 
SHOW_EVERY = 3000 # display stats every

start_q_table = None # is saved q-table exists, load it here




PLAYER_N = 1 # key to identify player
FOOD_N = 2 
ENEMY_N = 3 

# how the player, food and enemy will look like in color
d = {1: (255, 175, 0),
     2: (0,   255, 0),
     3: (0,    0, 255)}

# make a class to which each of the player, enemy and food will belong to. They all have common attributes like moving so better to have a class instead of repeating the code

class Blob:
    def __init__(self):
        # initialize starting locations randomly
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        # an overloaded method to print the blob's location
        return (f'{self.x}, {self.y}')

    def __sub__(self, other):
        # an overloaded method to calculate the distance between the current blob and other blob
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        # define what action means for player, choice is the meaning of the action
        # 4 possible actions, diagonal movement
        if choice == 0:
            self.move(x=1,y=1)
        elif choice == 0:
            self.move(x=-1,y=-1)
        elif choice == 2:
            self.move(x=-1,y=1)
        elif choice == 3:
            self.move(x=1,y=-1)
        

    def move(self, x=False, y=False):
        # move randomly if a value passed, else move as requested
        if not x: 
            # x is false
            self.x = np.random.randint(-1,2)
        else:
            self.x += x
        if not y: 
            # y is false
            self.y = np.random.randint(-1,2)
        else:
            self.y += y

        # take care of blobs trying to move out of the arena - stop them at the boundary
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

if start_q_table is None:
    # no q-table has been loaded
    """
    state space will contain 2 things, distance to the food and distance to the enemy.
    so any state space will look have 2 tuples 
    (x1,y1), (x2,y2)
    so each state is a tuple of tuples
    And each state has 4 possible actions
    """
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    # randomly initialize 4 different values that will act as q-values for the actions at this state
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0, size = 4)]

else:
    # q-table was saved before
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food   = Blob()
    enemy  = Blob() 
    
    if episode%SHOW_EVERY==0:
        print(f'on episode number {episode}')
        print(f'{SHOW_EVERY} ep mean reward = {np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True

    else:
        show = False

    # fix number of steps in 1 episode until it goes to next episode
    for i in range(200):
        obs = (player-food, player-enemy) # s

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        
        else:
            action = np.random.randint(0, 4)

        player.action(action)
        # action taken

        if player.x == enemy.x and player.y == enemy.y:
            # game over
            reward = -ENEMY_PENALTY

        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD

        else:
            reward = MOVE_PENALTY
        # reward collected

        new_obs = (player-food, player-enemy) # s'
        max_future_q = np.max(q_table[new_obs]) # max q at s'
        current_q = q_table(obs)[action] # q at s

        # the terminal states have fixed reward so the q formula doesn't there
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q


