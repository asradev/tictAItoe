# tictAItoe
Small tic-tac-toe game made with pygame that lets you play against a friend or against an AI that uses different algorithms. **The player can choose the algorithm that the AI will use**, acting as some sort of difficulty selection, as certain algorithms will pose more of a challenge to the player than others. The possible AI types are the following:

## Neural Network

Simple neural network trained with previous player inputs. The database of training samples is located in the xvalues.txt file, and the labels are in the yvalues.txt file. An example database with ~800 training samples comes included.

The neural network has the following architecture (located in nn.py):

```python
    model = Sequential()
    model.add(Dense(18, input_shape=(9,), activation="relu"))
    model.add(Dense(18, activation="relu"))
    model.add(Dense(9, activation="relu"))
    model.add(Dense(9, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Everytime a human player inputs a move, the board state and the cell that the player selected are logged into the database files for later training. Each line in the xvalues.txt represents a board state (0 = empty cell, 1 = white cell, 2 = black cell), and each corresponding line in yvalues.txt represents the cell selected (values from 0 to 8, 0 represents the top-left cell, and 8 represents the bottom-right cell)

**The AI does not know how to win, or even what a win means. It just tries to mimic how the player would play at any given state of the game.** The AI is constantly learning, everytime a game is restarted, the neural network is trained with the new data.

## Minimax

Algorithm that prioritizes minimizing the possible loss for a worst case (maximum loss) scenario. Perfect for zero-sum games like tic-tac-toe, in which each participant's gain or loss of utility is exactly balanced by the losses or gains of the utility of the other participants.

The code of this algorithm is located in the ai.py file

**In the context of this game, this algorithm is unbeatable, and the most you can expect to achieve is a draw.**

## Q-learning

Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances. Q-learning finds a policy that is optimal in the sense that it maximizes the expected value of the total reward over any and all successive steps, starting from the current state. "Q" names the function that returns the reward used to provide the reinforcement and can be said to stand for the "quality" of an action taken in a given state. This algorithm has an OOP implementation, in the rl.py file.

This AI learns every state's value by visiting all of them many times until it learns the full value function. **Keeping in mind that tic-tac-toe is a game with not that many possible states, this algorithm is well suited for this situation.**

An example database of qvalues comes included. **The performance of this AI will improve everytime a match is played. As such, an option to train the AI by making it play against itself for 200 matches comes enabled.**

## Deep Reinforcement Learning

Deep Reinforcement Learning combines neural networks and reinforcement learning. Instead of having a value function that maps every state to a value, we use a neural network that takes states as inputs and outputs values. This way, the neural network can learn the similarities between states and achieve better performances in more complex situations. This algorithm also has an OOP implementation, in the rl.py file.

In order to train the network, for every state we calculate a target value: target = v(s) + α(v(s’)+R-v(s)) where v(s) and v(s’) are calculated from the neural network itself. After each target calculation, one iteration of stochastic gradient descent is executed.

Two models (one for each player, with pre-trained weights) come included. **The performance of this AI will also improve everytime a match is played. As such, an option to train the AI by making it play against itself for 50 matches comes enabled.**

![Preview image](https://raw.githubusercontent.com/alvarosaulrodriguezaleman/tictAItoe/master/preview.png)
