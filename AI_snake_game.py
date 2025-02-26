#RL for AI Snake Game using PyTorch (will use lightning later)
#RL is teaching a sw agent how to behave in an environment by telling it how good it's doing

import os
import git


#1. impport the game from git
#
# # Define the path where you want to clone the repository
# REPO_PATH = os.path.join('AI_games')
#
# # Check if the directory exists, if not, create it
# if not os.path.exists(REPO_PATH):
#     os.makedirs(REPO_PATH)
#
# # Clone the Git repository
# REPO_URL = 'https://github.com/patrickloeber/python-fun.git'  # Use .git at the end
# repo = git.Repo.clone_from(REPO_URL, REPO_PATH)

import pygame
import random
from enum import Enum
from collections import namedtuple

import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)


# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20


class SnakeGameAI:

    def __init__(self, w=1080, h=720):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        #now that we created a "def reset" we need to call a self.reset() fcn
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        #to keep track of frame iterations it's initialized at 0 or reset (after playing) to 0
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1 #update the frame iteration by 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            ### BECAUSE its AI so no need for key usage
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        #now we include the reward counter
        reward = 0
        game_over = False
        #included the condition "or" instead of using " | "
        #also saying if the snake is too long (time) then the game is over
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 #included with the reward score
            return reward, game_over, self.score #once the game is over display the reward score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 #one of the AI reward additions
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    #we are going to calculate the actions to be determined (straight, right turn, left turn)
    def _move(self, action):
        # [straight, right, left]

        #define all possible direction in "clockwise order" with an index
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        #we use numpy array to compare the action to the index
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx] #no change - straight
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx + 1) % 4 #mod math to indicate the next move
            new_dir = clock_wise[next_idx] #right turn (r->d->l->up) b/c it's clock_wise the snake keeps to the order
        else: #[0,0,1]
            next_idx = (idx -1 ) % 4
            new_dir=clock_wise[next_idx] #going CCW so its a left turn (r->u->l->d)

        self.direction = new_dir

        #NOTE: "BLOCK_SIZE = 20" and is in pixels
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

#this no longer works with user input so it will have to be updated (TEMPORARILY COMMENTED)
# if __name__ == '__main__':
#     game = SnakeGameAI()
#
#     # game loop
#     while True:
#         game_over, score = game.play_step()
#
#         if game_over == True:
#             break
#
#     print('Final Score', score)
#
#     pygame.quit()


