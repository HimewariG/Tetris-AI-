# Tetris-AI-

This is a project is an attempt to implement Genetic Algorithm onto Tetris. The goal is to create an AI that is able to recognize patterns. It is written in Python and the  main library used is Pytorch and Pygame.

# Structure of the code

This project is constructed using pytorch and pygame, and it consists of two parts. The first part is the training platform. It is used to optimize the value in the neuronetwork. It is tuned using genetic algorithm. The second part is the performing platform. The platform is built to visualize the gameplay of the AI. It also consists of a human playable part which allows comparison of performance which is tracked using a score system.

# Training platform

The training platform is split into 3 parts. The first part is the brain part, which uses Pytorch to build an NN it is responsible for making the decision. The second part is the genetic algorithm, which adjusts the value stored in the NN to adjust it into a value that can make a good decision. The final part is the game part, which consists of the logic of the game, and it is run to simulate the game the AI played. It is used for generating a score to indicate how well the AI is performing so we can evolve it in the genetic algorithm.

# Game platform part

The AI trained in the training platform can be tested here. It has two parts: the human playable part and the AI part, which is used to visualize and show how the AI plays.

# How it works

For the training platform. First, a list of NN with random values is created. Then each NN will be used to simulate gameplay using the game class. We first create a field, then generate a move list consisting of all positions a block can place, then pass the value to the NN and use Probability function to return a value that is used to pick which position it should place. After iteration, using the fitness score guide that is created with functions such as checking how many holes each block placed. It determines how good each NN does and how good they are and ranks them. We then keep the elite ones and mix up their internal value (split them) and combine the value to create a new NN with value from each parent. Also, to add diversity, there is a mutate function that mutates the child. Then we have a new list of candidates and we loop it again. After a number of generations, we load the one with the highest score in the list of candidates.
