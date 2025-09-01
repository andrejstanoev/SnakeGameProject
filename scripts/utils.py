import numpy as np

from scripts.constants import GRID_W, GRID_H

def distance_to_wall(pos, direction):
    x, y = pos
    dx, dy = direction
    distance = 0
    while 0 <= x < GRID_W and 0 <= y < GRID_H:
        x += dx
        y += dy
        distance += 1
    return distance / max(GRID_W, GRID_H)


def select_action(state, weights):
    num_features = len(state)
    num_actions = 4
    weights_matrix = weights.reshape((num_features, num_actions))
    scores = np.dot(state, weights_matrix)
    return np.argmax(scores)

