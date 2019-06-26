# tictAItoe
Small tic-tac-toe game made with pygame that lets you play against a friend or against an AI that uses different algorithms. **The player can choose the algorithm that the AI will use**, acting as some sort of difficulty selection, as certain algorithms will pose more of a challenge to the player than others. The possible AI types are the following:

## Neural Network

Simple neural network trained with previous player inputs. The database of training samples is located in the xvalues.txt file, and the labels are in the yvalues.txt file. An example database with ~200 training samples comes included.

The neural network has the following architecture:

```python
    model = Sequential()
    model.add(Dense(10, input_shape=(9,), activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(9, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
```

Everytime a human player inputs a move, the board state and the cell that the player selected are logged into the database files for later training. Each line in the xvalues.txt represents a board state (0 = empty cell, 1 = white cell, 2 = black cell), and each corresponding line in yvalues.txt represents the cell selected (values from 0 to 8, 0 represents the top-left cell, and 8 represents the bottom-right cell)

**The AI is constantly learning, everytime a game is restarted, the neural network is trained with the new data.**

## Minimax

Algorithm that prioritizes minimizing the possible loss for a worst case (maximum loss) scenario. Perfect for zero-sum games like tic-tac-toe, in which each participant's gain or loss of utility is exactly balanced by the losses or gains of the utility of the other participants.

**In the context of this game, this algorithm is unbeatable, and the most you can expect to achieve is a draw.**

![Preview image](https://raw.githubusercontent.com/alvarosaulrodriguezaleman/tictAItoe/master/preview.png)
