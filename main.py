import os.path
import random

import numpy as np
import pygame as pg

import utils as ut
import nn
import ai


# returns a rect object that is centered on x and y
def centered_rect(x, y, w, h):
    rect = pg.Rect(0, 0, w, h)
    rect.center = (x, y)
    return rect


# updates the AI type to play against
def change_ai(text):
    global aiType
    aiType = text


# updates the global gameState with the text provided
def update_game_state(text):
    global gameState
    gameState = text
    restart()


# restarts the game
def restart():
    global grid, victory, white_turn, player_turn, player_first, model, modelHistory, modelTrained, trainingCount, best
    grid = np.zeros((grid.shape[0], grid.shape[1]))
    victory = None
    white_turn = True
    player_turn = bool(random.getrandbits(1))
    player_first = player_turn
    best = None  # used for playing against the minimax AI

    if logging and (modelTrained or not modelTrained and trainingCount >= 50):
        model, modelHistory = nn.train_model(model, 100)
        modelTrained = True


if __name__ == '__main__':
    pg.init()
    model = nn.create_model()
    trainingCount = 0
    modelTrained = False
    logging = True

    if os.path.isfile("xvalues.txt") and os.path.isfile("yvalues.txt"):
        with open('xvalues.txt') as file:
            trainingCount = sum(1 for line in file)
            if trainingCount >= 50:
                model, modelHistory = nn.train_model(model, 500)
                modelTrained = True

    size = width, height = 1024, 600  # size of the screen
    screen = pg.display.set_mode(size)
    pg.display.set_caption('tictAItoe')  # title of the game window

    '''
        The possible gameStates are the following:
        
        gameState = "title"         -> Main menu
        gameState = "twoPlayerGame" -> Two player game
        gameState = "aiGame"        -> Game against the AI
    '''
    gameState = "title"

    '''
        The possible aiTypes are the following:
        
        aiType = "nn"       -> AI based on a neural-network that learns with previous player inputs
        aiType = "minimax"  -> AI based on the minimax algorithm, intended to be unbeatable
    '''
    aiType = "nn"

    '''
        The possible victory values are the following:
    
        victory = None     -> Game has not ended
        victory = "white"  -> White has won
        victory = "black"  -> Black has won
        victory = "draw"   -> Game has ended in a draw
    '''
    victory = None

    largeFont = pg.font.Font(None, 48)
    mediumFont = pg.font.Font(None, 24)

    white = (255, 255, 255)  # constant for white color
    black = (0, 0, 0)  # constant for black color

    '''
        The play grid is a 3x3 matrix. The cell values mean the following:
        cell = 0 -> Empty cell
        cell = 1 -> White cell
        cell = 2 -> Black cell
    '''
    grid = np.zeros((3, 3))
    cellMargin = 20  # Margin between cells
    grid_W = 80  # Width of each cell
    grid_H = 80  # Height of each cell
    margin_X = width // 2 - ((cellMargin + grid_W) * grid.shape[1] + cellMargin) / 2  # Horizontal margin of the grid
    margin_Y = height // 2 - ((cellMargin + grid_H) * grid.shape[0] + cellMargin) / 2  # Vertical margin of the grid
    white_turn = True  # Indicates if it's White's turn
    player_turn = True  # Indicates if it's the player's turn (in a game vs AI)
    player_first = player_turn

    # game loop
    while True:
        mouse = pg.mouse.get_pos()  # mouse position
        screen.fill((66, 134, 244))  # set screen background color

        for event in pg.event.get():
            if event.type == pg.QUIT:
                # exit the game
                pg.quit()
                quit()
            elif event.type == pg.MOUSEBUTTONDOWN:
                # a click has been registered
                if gameState == "twoPlayerGame" or (gameState == "aiGame" and player_turn):
                    # detect which cell the user has clicked
                    row = (mouse[1] - margin_Y) // (grid_H + cellMargin)
                    column = (mouse[0] - margin_X) // (grid_W + cellMargin)
                    # check if it's an empty cell inside the grid and if the game has not finished yet
                    if 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1] and grid[int(row)][int(column)] == 0 \
                            and victory is None:
                        if logging:
                            with open('xvalues.txt', 'a+') as outfile:
                                grid.tofile(outfile, sep=" ")
                                outfile.write("\n")
                        if white_turn:
                            grid[int(row)][int(column)] = 1
                        else:
                            grid[int(row)][int(column)] = 2
                        if logging:
                            with open('yvalues.txt', 'a+') as outfile:
                                outfile.write(str(3 * row + column))
                                outfile.write("\n")
                        trainingCount = trainingCount + 1
                        white_turn = not white_turn  # The turn passes to the other player
                        player_turn = not player_turn  # The turn passes to the AI
                        victory = ai.check_victory(grid)  # Check if the game has ended

        if gameState == "title":
            ut.display_text("tictAItoe", largeFont, (0, 0, 255), width // 2, height // 4, screen)

            if aiType == "nn":
                if modelTrained:
                    ut.update_button("play against the AI", centered_rect(width // 2 + 150, height // 2 - 60, 175, 50),
                                     (0, 195, 255), (18, 206, 255), white, black, mediumFont, screen, mouse,
                                     action=update_game_state, arg="aiGame")
                    ut.update_button("plot training results",
                                     centered_rect(width // 2 + 150, height // 2 + 120, 175, 50), (120, 0, 255),
                                     (140, 40, 255), white, black, mediumFont, screen, mouse,
                                     action=nn.plot_training, arg=modelHistory)
                else:
                    ut.display_text("there are not enough training samples to train the network!", mediumFont,
                                    white, width // 2, height // 2 - 30, screen)
                    ut.display_text("to play against the AI, play against a friend until " + str(50 - trainingCount) +
                                    " more moves are made.", mediumFont, white, width // 2, height // 2 - 10, screen)
                ut.display_text("AI based on a neural network, will play better as the training samples grow in size.",
                                mediumFont, white, width // 2, height - 65, screen)
            if aiType == "minimax":
                ut.update_button("play against the AI", centered_rect(width // 2 + 150, height // 2 - 60, 175, 50),
                                 (0, 195, 255), (18, 206, 255), white, black, mediumFont, screen, mouse,
                                 action=update_game_state, arg="aiGame")
                ut.display_text("AI based on the minimax algorithm, intended to be unbeatable.",
                                mediumFont, white, width // 2, height - 65, screen)

            ut.update_button("play against a friend", centered_rect(width // 2 + 150, height // 2, 175, 50),
                             (0, 0, 215), (0, 0, 255), white, black, mediumFont, screen, mouse,
                             action=update_game_state, arg="twoPlayerGame")
            ut.update_button("quit", centered_rect(width // 2 + 150, height // 2 + 60, 175, 50), (120, 0, 255),
                             (140, 40, 255), white, black, mediumFont, screen, mouse, action=quit)
            ut.update_button("Neural Network", centered_rect(width // 2 - 150, height // 2 - 60, 175, 50), (0, 0, 215),
                             (0, 0, 255), white, white, mediumFont, screen, mouse, bw=1, action=change_ai, arg="nn")
            ut.update_button("Minimax", centered_rect(width // 2 - 150, height // 2, 175, 50), (0, 0, 215),
                             (0, 0, 255), white, white, mediumFont, screen, mouse, bw=1, action=change_ai,
                             arg="minimax")
            ut.display_text("AI type", mediumFont, white, width // 2 - 150, height // 2 - 105, screen)
            ut.display_text(str(trainingCount) + " training samples have been recorded so far.", mediumFont,
                            white, width // 2, height - 30, screen)
        elif gameState == "twoPlayerGame" or gameState == "aiGame":
            if victory is not None:
                turnMsg = victory + " wins!"
                if victory == "white":
                    turnMsgColor = white
                elif victory == "black":
                    turnMsgColor = black
                else:
                    turnMsg = "Draw"
                    turnMsgColor = (0, 0, 255)

                ut.update_button("restart", centered_rect(width // 2, height - 110, 175, 50), (0, 0, 215), (0, 0, 255),
                                 white, black, mediumFont, screen, mouse, action=restart)
                ut.update_button("back to main menu", centered_rect(width // 2, height - 50, 175, 50), (120, 0, 215),
                                 (140, 40, 255), white, black, mediumFont, screen, mouse, action=update_game_state,
                                 arg="title")
            elif white_turn:
                turnMsg = "it's White's turn"
                turnMsgColor = white
            else:
                turnMsg = "it's Black's turn"
                turnMsgColor = black
            ut.display_text(turnMsg, largeFont, turnMsgColor, width // 2, height // 7, screen)

            if gameState == "aiGame":
                if player_first:
                    indicatorMsg = "you play as White"
                    indicatorMsgColor = white
                else:
                    indicatorMsg = "you play as Black"
                    indicatorMsgColor = black
                ut.display_text(indicatorMsg, mediumFont, indicatorMsgColor, width // 2, 40, screen)

                if aiType == "minimax" and best is not None and victory is None:
                    if (best[2] == 1 and not player_first) or (best[2] == -1 and player_first):
                        ut.display_text("The AI thinks that you will lose", mediumFont, white, width // 2, height - 110,
                                        screen)
                    elif best[2] == 0:
                        ut.display_text("The AI thinks that the game will end in a draw", mediumFont, white, width // 2,
                                        height - 110, screen)
                    else:
                        ut.display_text("The AI thinks that you will win", mediumFont, white, width // 2, height - 110,
                                        screen)

                if not player_turn and victory is None:
                    if aiType == "nn":
                        rawPrediction = nn.make_prediction(model, grid.reshape(1, 9))
                        prediction = np.argsort(-rawPrediction)
                        aux = 0
                        for i in prediction[0]:
                            aux = aux + 1
                            row = i // 3
                            column = i % 3
                            if grid[int(row)][int(column)] == 0:
                                if white_turn:
                                    grid[int(row)][int(column)] = 1
                                else:
                                    grid[int(row)][int(column)] = 2
                                break
                        print("Prediction number", aux, "selected.")
                    elif aiType == "minimax":
                        best = ai.minimax(grid, (grid == 0).sum(), white_turn)
                        if white_turn:
                            grid[best[0]][best[1]] = 1
                        else:
                            grid[best[0]][best[1]] = 2
                    white_turn = not white_turn  # The turn passes to the other player
                    player_turn = not player_turn  # The turn passes to the human
                    victory = ai.check_victory(grid)  # Check if the game has ended

            ut.display_grid(grid, screen, mouse, cellMargin, grid_W, grid_H, margin_X, margin_Y)

        pg.display.flip()
