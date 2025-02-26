import torch
import random
import numpy as np
from collections import deque #memory storage
from AI_Snake_game.AI_snake_game import SnakeGameAI, Direction, Point
from AI_Snake_game.DQN_model import Linear_QNet, QTrainer
from game_plot_helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001 #will get changed later

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls randomness
        self.gamma = 0.9  # discount rate - value smaller than 1 (usually somehwere btw 0.8-0.9)
        self.memory = deque(maxlen=MAX_MEMORY)  # will popleft() if memory exceeds
        #self.model = Linear_QNet(<input_size>, <hidden_size>, <output_size>)
        # in this case <input_size = State>, <hidden_size = try diff. #s>, <output_size = Actions>)
        #therefore the "State" and "Actions" are fixed
        self.model = Linear_QNet(11, 256, 3)
        self.trainer= QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int) #converts the food location bools to an int np value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY is reached
        #NOTE: the "self.memory.append((state, action, reward, next_state, done))" is a single tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #return a list of tuples
        else:
            mini_sample = self.memory

        states, actions,rewards,next_states, dones = zip(*mini_sample) # "zip" unpacks the 5 tuples into 5 sep. lists
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        ### another method instead of using "zip(*mini_sample)"
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) #can take multiple short memory tensors
                                                                        # and use as a batch for the NN model

    def get_action(self, state):
        # random moves: tradeoff exploration (more random moves in the beginning to learn)
        # & exploitation (as our Agent improves the agent exploits the model) in deep learning
        self.epsilon = 80 - self.n_games #change later
        final_move = [0,0,0]
        # this IF statement says the smaller our self.epsilon -> the less freq. the agent's moves become random
        #can even become neg meaning no more moves
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #note: "self.model.predict(state0)" used for tensorflow but for torch use just "self.model()"
            _prediction = self.model(state0) #will execute the "def forward" fcn in the model for predictions
            move = torch.argmax(_prediction).item() #note: ".item()" converts the prediction to one #
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 #keeps high scores
    agent = Agent()
    game = SnakeGameAI()
    #create training loop - to run forever until we quit the script
    while True:
        # get old/current state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory - NOTE: we got the param hints once we include args in "def train_short_memory(<args>)"
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory & plot results - replays all the moves and stores the moves for improvement
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

                print('Game: ',agent.n_games, ' Score: ', score, ' Record: ', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()


