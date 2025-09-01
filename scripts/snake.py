import numpy as np
from scripts.constants import GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT, ACTIONS, INITIAL_SNAKE_POS, NUM_FOOD, NUM_EPISODES, \
    MAX_STEPS
import random

from scripts.utils import distance_to_wall, select_action


def fitness_func(ga_instance, solution, solution_idx):
    total_reward = 0

    for _ in range(NUM_EPISODES):
        game = SnakeGame()
        state = game.reset()
        episode_reward = 0
        for _ in range(MAX_STEPS):
            action = select_action(state, solution)
            state, reward, done = game.step(action)
            episode_reward += reward
            if done:
                break
        total_reward += episode_reward


    return total_reward / NUM_EPISODES

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.done = False
        self.direction = ACTIONS[1]
        self.score = 0
        self.snake = INITIAL_SNAKE_POS.copy()
        self.frame = 0
        self.food = self.spawn_food(NUM_FOOD)

        return self.get_state()

    def spawn_food(self, n_apples):
        all_positions = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
        available_positions = list(set(all_positions) - set(self.snake))
        food_positions = random.sample(available_positions, n_apples)
        return food_positions

    def is_collision(self, pos: tuple[int, int]) -> int:
        x, y = pos
        return int(x < 0 or x >= GRID_W or y < 0 or y >= GRID_H or pos in self.snake)

    def body_in_steps(self, head, direction, steps):
        x, y = head
        dx, dy = direction
        for step in range(1, steps + 1):
            pos = (x + dx * step, y + dy * step)
            if pos in self.snake:
                return 1
        return 0

    def tail_distance(self) -> float:
        head_x, head_y = self.snake[0]
        tail_x, tail_y = self.snake[-1]
        dist = abs(head_x - tail_x) + abs(head_y - tail_y)
        return dist / (GRID_W + GRID_H)

    def pick_nearest_apple(self):
        head_x, head_y = self.snake[0]
        if self.food:
            dists = [abs(fx - head_x) + abs(fy - head_y) for (fx, fy) in self.food]
            min_idx = np.argmin(dists)
            food_x, food_y = self.food[min_idx]
        else:
            food_x, food_y = head_x, head_y

        return food_x,food_y

    def manhattan_distance_to_all_apples(self):
        head_x, head_y = self.snake[0]
        if self.food:
            all_dists = [abs(head_x - fx) + abs(head_y - fy) for fx, fy in self.food]
            sum_dist_to_apples = sum(all_dists) / (len(self.food) * (GRID_W + GRID_H))
            avg_dist_to_apples = np.mean(all_dists) / (GRID_W + GRID_H)
        else:
            sum_dist_to_apples = 0.0
            avg_dist_to_apples = 0.0

        return sum_dist_to_apples,avg_dist_to_apples

    def apples_per_quadrant(self):
        apples_q1 = apples_q2 = apples_q3 = apples_q4 = 0
        for fx, fy in self.food:
            if fx < GRID_W / 2 and fy < GRID_H / 2:
                apples_q1 += 1
            elif fx >= GRID_W / 2 and fy < GRID_H / 2:
                apples_q2 += 1
            elif fx < GRID_W / 2 and fy >= GRID_H / 2:
                apples_q3 += 1
            else:
                apples_q4 += 1
        n_food = max(len(self.food), 1)
        apples_q1 /= n_food
        apples_q2 /= n_food
        apples_q3 /= n_food
        apples_q4 /= n_food

        return apples_q1,apples_q2,apples_q3,apples_q4

    def fraction_of_empty_cells(self):
        n_empty = GRID_W * GRID_H - len(self.snake) - len(self.food)
        frac_empty = n_empty / (GRID_W * GRID_H)
        return frac_empty

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction

        food_x, food_y = self.pick_nearest_apple()

        straight = (head_x + dir_x, head_y + dir_y)
        left = (head_x - dir_y, head_y + dir_x)
        right = (head_x + dir_y, head_y - dir_x)

        danger_straight = self.is_collision(straight)
        danger_left = self.is_collision(left)
        danger_right = self.is_collision(right)


        dist_straight = distance_to_wall((head_x, head_y), (dir_x, dir_y))
        dist_left = distance_to_wall((head_x, head_y), (-dir_y, dir_x))
        dist_right = distance_to_wall((head_x, head_y), (dir_y, -dir_x))


        body_2_straight = self.body_in_steps((head_x, head_y), (dir_x, dir_y), 2)
        body_3_left = self.body_in_steps((head_x, head_y), (-dir_y, dir_x), 3)
        body_4_right = self.body_in_steps((head_x, head_y), (dir_y, -dir_x), 4)


        dx = (food_x - head_x) / GRID_W
        dy = (food_y - head_y) / GRID_H


        dir_up = int(self.direction == UP)
        dir_down = int(self.direction == DOWN)
        dir_left = int(self.direction == LEFT)
        dir_right = int(self.direction == RIGHT)


        num_apples_left = len(self.food)


        tail_dist = self.tail_distance()


        sum_dist_to_apples,avg_dist_to_apples = self.manhattan_distance_to_all_apples()

        apples_q1,apples_q2,apples_q3,apples_q4 = self.apples_per_quadrant()

        frac_empty = self.fraction_of_empty_cells()

        state = np.array([
            danger_straight, danger_left, danger_right,
            dist_straight, dist_left, dist_right,
            body_2_straight, body_3_left, body_4_right,
            dx, dy,
            dir_up, dir_down, dir_left, dir_right,
            num_apples_left,
            tail_dist,
            sum_dist_to_apples,
            avg_dist_to_apples,
            apples_q1, apples_q2, apples_q3, apples_q4,
            frac_empty
        ], dtype=float)

        return state

    def step(self, action_idx):
        if self.done:
            return self.get_state(), 0, True

        new_dir = ACTIONS[action_idx]
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        reward = -0.2
        done = False

        if self.is_collision(new_head):
            done = True
            reward = -10
            self.done = True
            return self.get_state(), reward, done

        self.snake.insert(0, new_head)

        if new_head in self.food:
            self.food.remove(new_head)
            self.score += 1
            reward = 10
            if not self.food:
                done = True  # Win!
                reward = 50
                self.done = True
        else:
            self.snake.pop()

        self.frame += 1
        return self.get_state(), reward, done