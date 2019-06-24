# tictAItoe
Small tic-tac-toe game made with pygame that lets you play against a friend or against an AI trained with previous player inputs.
The database of training samples is located in the xvalues.txt file, and the labels are in the yvalues.txt file. A default database with ~200 training examples comes included.

The AI is composed by a neural network with the following architecture:

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

#### The AI is constantly learning, everytime a game is restarted, the neural network is trained with the new data.
