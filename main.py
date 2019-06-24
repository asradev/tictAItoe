import numpy as np
import pygame as pg
import random
import os.path
import ai


'''
    Function that displays an interactive button.
    msg:    Text inside the button.
    rect:   pygame.Rect object of the size of the desired button.
    ic:     Color of the button when the mouse is not hovering it.
    ac:     Color of the button when the mouse is hovering it.
    tc:     Color of the text inside the button.
    font:   Font that the text inside the color will use (pygame.font.Font).
    action: Method to be called when the button is pressed.
    arg:    Argument to be passed to the method determined by the action parameter.
'''
def update_button(msg, rect, ic, ac, tc, font, action=None, arg=None):
    click = pg.mouse.get_pressed()

    if rect.x + rect.w > mouse[0] > rect.x and rect.y + rect.h > mouse[1] > rect.y:
        pg.draw.rect(screen, ac, rect)

        if click[0] == 1 and action is not None:
            if arg is not None:
                action(arg)
            else:
                action()
    else:
        pg.draw.rect(screen, ic, rect)

    textsurf, textrect = text_objects(msg, font, tc)
    textrect.center = ((rect.x + (rect.w / 2)), (rect.y + (rect.h / 2)))
    screen.blit(textsurf, textrect)


# function that simplifies text rendering
def text_objects(text, font, color):
    textsurface = font.render(text, True, color)
    return textsurface, textsurface.get_rect()


# updates the global gameState with the text provided
def update_gamestate(text):
    global gameState
    gameState = text
    restart()


# restarts the game
def restart():
    global grid, victory, white_turn, player_turn, player_first, model, modelHistory, modelTrained, trainingCount
    grid = np.zeros((grid.shape[0], grid.shape[1]))
    victory = 0
    white_turn = True
    player_turn = bool(random.getrandbits(1))
    player_first = player_turn

    if logging and (modelTrained or not modelTrained and trainingCount >= 50):
        model, modelHistory = ai.trainModel(model, 100)
        modelTrained = True


# checks the play grid to see if the game has ended
def check_victory(g):
    v = 0

    # check if all spaces have been selected
    if g.min() != 0:
        v = -1  # the game is a draw

    # check if there is an horizontal line
    for i in range(g.shape[0]):
        aux = True
        for j in range(1, g.shape[1]):
            if g[i][j] != g[i][0]:
                aux = False
                break
        if aux:
            v = g[i][0]
            break

    # check if there is a vertical line
    for j in range(g.shape[1]):
        aux = True
        for i in range(1, g.shape[0]):
            if g[i][j] != g[0][j]:
                aux = False
                break
        if aux:
            v = g[0][j]
            break

    # check if there is a diagonal line
    if g[0][0] == g[1][1] == g[2][2] and g[0][0] != 0:
        v = g[0][0]
    if g[2][0] == g[1][1] == g[0][2] and g[2][0] != 0:
        v = g[2][0]

    return v


# Display the play grid
def displayGrid(grid, screen, cellMargin, grid_W, grid_H, margin_X, margin_Y):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = (84, 145, 255)
            if grid[i][j] == 1:
                color = (255, 255, 255)
            elif grid[i][j] == 2:
                color = (0, 0, 0)
            pg.draw.rect(screen, color,
                         [(cellMargin + grid_W) * j + cellMargin + margin_X,
                          (cellMargin + grid_H) * i + cellMargin + margin_Y,
                          grid_W,
                          grid_H])


pg.init()
model = ai.createModel()
trainingCount = 0
modelTrained = False
logging = True

if os.path.isfile("xvalues.txt") and os.path.isfile("yvalues.txt"):
    with open('xvalues.txt') as file:
        trainingCount = sum(1 for line in file)
        if trainingCount >= 50:
            model, modelHistory = ai.trainModel(model, 500)
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
    The possible victory values are the following:

    victory = 0     -> Game has not ended
    victory = 1     -> White has won
    victory = 2     -> Black has won
    victory = -1    -> Game has ended in a draw
'''
victory = 0

largeFont = pg.font.Font(None, 48)
mediumFont = pg.font.Font(None, 24)

titleSurf, titleRect = text_objects('tictAItoe', largeFont, (0, 0, 255))
titleRect.center = (width // 2, height // 3)

aiPlayRect = pg.Rect(0, 0, 175, 50)
aiPlayRect.center = (width // 2, height // 2)       # Play against AI button location
playRect = pg.Rect(0, 0, 175, 50)
playRect.center = (width // 2, height // 2 + 60)    # Play against a friend button location
quitRect = pg.Rect(0, 0, 175, 50)
quitRect.center = (width // 2, height // 2 + 120)   # Quit button location
plotRect = pg.Rect(0, 0, 175, 50)
plotRect.center = (width // 2, height // 2 + 180)   # Quit button location
restartRect = pg.Rect(0, 0, 175, 50)
restartRect.center = (width // 2, height - 110)     # Restart button location
menuRect = pg.Rect(0, 0, 175, 50)
menuRect.center = (width // 2, height - 50)         # Back to menu button location

'''
    The play grid is a 3x3 matrix. The cell values mean the following:
    cell = 0 -> Empty cell
    cell = 1 -> White cell
    cell = 2 -> Black cell
'''
grid = np.zeros((3, 3))
cellMargin = 20  # Margin between cells
grid_W = 80      # Width of each cell
grid_H = 80      # Height of each cell
margin_X = width // 2 - ((cellMargin + grid_W) * grid.shape[1] + cellMargin) / 2    # Horizontal margin of the grid
margin_Y = height // 2 - ((cellMargin + grid_H) * grid.shape[0] + cellMargin) / 2   # Vertical margin of the grid
white_turn = True    # Indicates if it's White's turn
player_turn = True   # Indicates if it's the player's turn (in a game vs AI)
player_first = player_turn

while True:
    mouse = pg.mouse.get_pos()  # mouse position

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
                        and victory == 0:
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
                    white_turn = not white_turn     # The turn passes to the other player
                    player_turn = not player_turn   # The turn passes to the AI
                    victory = check_victory(grid)   # Check if the game has ended

    screen.fill((66, 134, 244))  # set screen background color

    if gameState == "title":
        screen.blit(titleSurf, titleRect)  # display title
        if modelTrained:
            update_button("play against the AI", aiPlayRect, (0, 195, 255), (18, 206, 255), (255, 255, 255), mediumFont,
                          action=update_gamestate, arg="aiGame")
            update_button("plot training results", plotRect, (120, 0, 255), (140, 40, 255), (255, 255, 255), mediumFont,
                          action=ai.plotTraining, arg=modelHistory)
        else:
            trainingSurf1, trainingRect1 = text_objects("there are not enough training examples to train the network!",
                                                        mediumFont, (255, 255, 255))
            trainingSurf2, trainingRect2 = text_objects("to play against the AI, play against a friend until "
                                                        + str(50-trainingCount) + " more moves are made.",
                                                        mediumFont, (255, 255, 255))
            trainingRect1.center = (width // 2, height // 2 - 30)
            trainingRect2.center = (width // 2, height // 2 - 10)
            screen.blit(trainingSurf1, trainingRect1)
            screen.blit(trainingSurf2, trainingRect2)
        update_button("play against a friend", playRect, (0, 0, 215), (0, 0, 255), (255, 255, 255), mediumFont,
                      action=update_gamestate, arg="twoPlayerGame")
        update_button("quit", quitRect, (120, 0, 255), (140, 40, 255), (255, 255, 255), mediumFont, action=quit)
        countSurf, countRect = text_objects(str(trainingCount) + " training examples have been recorded so far.",
                                            mediumFont, (255, 255, 255))
        countRect.center = (width // 2, height - 30)
        screen.blit(countSurf, countRect)
    elif gameState == "twoPlayerGame" or gameState == "aiGame":
        if victory != 0:
            if victory == 1:
                turnSurf, turnRect = text_objects("White wins!", largeFont, (255, 255, 255))
            elif victory == 2:
                turnSurf, turnRect = text_objects("Black wins!", largeFont, (0, 0, 0))
            else:
                turnSurf, turnRect = text_objects("Draw", largeFont, (0, 0, 255))

            update_button("restart", restartRect, (0, 0, 215), (0, 0, 255), (255, 255, 255), mediumFont, action=restart)
            update_button("back to main menu", menuRect, (120, 0, 215), (140, 40, 255), (255, 255, 255), mediumFont,
                          action=update_gamestate, arg="title")
        elif white_turn:
            turnSurf, turnRect = text_objects("it's White's turn", largeFont, (255, 255, 255))
        else:
            turnSurf, turnRect = text_objects("it's Black's turn", largeFont, (0, 0, 0))
        turnRect.center = (width // 2, height // 7)
        screen.blit(turnSurf, turnRect)

        if gameState == "aiGame":
            if player_first:
                indicatorSurf, indicatorRect = text_objects("you play as White", mediumFont, (255, 255, 255))
            else:
                indicatorSurf, indicatorRect = text_objects("you play as Black", mediumFont, (0, 0, 0))
            indicatorRect.center = (width // 2, 40)
            screen.blit(indicatorSurf, indicatorRect)

            if not player_turn and victory == 0:
                rawPrediction = ai.makePrediction(model, grid.reshape(1, 9))
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
                        white_turn = not white_turn    # The turn passes to the other player
                        player_turn = not player_turn  # The turn passes to the human
                        victory = check_victory(grid)  # Check if the game has ended
                        break
                print("Prediction number", aux, "selected.")

        displayGrid(grid, screen, cellMargin, grid_W, grid_H, margin_X, margin_Y)

    pg.display.flip()
