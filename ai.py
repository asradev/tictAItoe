from math import inf as infinity


# checks the play grid to see if the game has ended
def check_victory(g):
    v = 0

    # check if all spaces have been selected
    if g.min() != 0:
        v = -1  # the game is a draw

    # check if there is any horizontal, vertical or diagonal line
    win_state = [
        [g[0][0], g[0][1], g[0][2]],
        [g[1][0], g[1][1], g[1][2]],
        [g[2][0], g[2][1], g[2][2]],
        [g[0][0], g[1][0], g[2][0]],
        [g[0][1], g[1][1], g[2][1]],
        [g[0][2], g[1][2], g[2][2]],
        [g[0][0], g[1][1], g[2][2]],
        [g[2][0], g[1][1], g[0][2]],
    ]

    if [1, 1, 1] in win_state:
        return 1  # white wins
    elif [2, 2, 2] in win_state:
        return 2  # black wins
    return v


# heuristic for the minimax algorithm
def evaluate(state):
    if check_victory(state) == 1:
        score = +1
    elif check_victory(state) == 2:
        score = -1
    else:
        score = 0

    return score


def minimax(state, depth, white_turn):
    if white_turn:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 9:
        return [1, 1, 0]  # always pick the middle cell at the start of the game

    if depth == 0 or evaluate(state) != 0:
        score = evaluate(state)
        return [-1, -1, score]

    # iterate through every empty cell
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i][j] == 0:
                if white_turn:
                    state[i][j] = 1
                else:
                    state[i][j] = 2
                score = minimax(state, depth - 1, not white_turn)
                state[i][j] = 0
                score[0], score[1] = i, j

                if white_turn:
                    if score[2] > best[2]:
                        best = score  # max value
                else:
                    if score[2] < best[2]:
                        best = score  # min value

    return best
