
import torch
import pygame
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
import random
import copy

import torch.nn as nn
class TetrisNN(nn.Module):
    def __init__(self):
        super(TetrisNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(48, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GeneticAlgorithm:
    def __init__(self, population,device,percentage_of_parents_to_keep,num_generations,mutation_rate,population_size,percentage_of_children_to_keep):
        self.model = TetrisNN()
        self.population = population
        self.device = device
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        model_architecture = {}
        for key_name, weights in self.model.state_dict().items():
            model_architecture[key_name] = weights.shape
        self.model_architecture = model_architecture
        self.percentage_of_parents_to_keep = percentage_of_parents_to_keep
        self.fitness_scores = []
        self.percentage_of_children_to_keep = percentage_of_children_to_keep
        self.count_of_initial_model_weights = sum(param.numel() for param in self.model.state_dict().values())
    def game_function(self, model):
        game = Game(self.model,self.device )
        score = game.run()
        return score
    def evaluate_population(self, population):
        self.fitness_scores = []
        for i in range(len(population)):

              state_dict = self.reconstruct_statedict(population[i])
              self.model.load_state_dict(state_dict)
              score = self.game_function(self.model)
              self.fitness_scores.append(score)




    def deconstruct_statedict(self, model: torch.nn.Module):
        one_dim_statedict = torch.Tensor()
        for key_name, weights in model.state_dict().items():
            flattend_weights = torch.flatten(weights)
            one_dim_statedict = torch.cat((one_dim_statedict, flattend_weights), dim=0)
        return one_dim_statedict
    def reconstruct_statedict(self, flattend_weights: torch.Tensor):

      state_dict = {}
      pointer = 0
      for key_name, weights_shape in self.model_architecture.items():
          if len(weights_shape) > 1:
              count_of_weights_this_module_needs = math.prod(weights_shape)
          else:
              count_of_weights_this_module_needs = weights_shape[0]
          slice_of_selected_weights = flattend_weights[
              pointer : pointer + count_of_weights_this_module_needs
          ]

          state_dict[key_name] = torch.reshape(
              slice_of_selected_weights, self.model_architecture[key_name]
          )
          pointer = count_of_weights_this_module_needs + pointer
      return state_dict

    def selection(self,population): #Can use bubble sort for alternative method
        for i in range(len(population)):
              state_dict = self.reconstruct_statedict(population[i])
              self.model.load_state_dict(state_dict)
              score = self.game_function(self.model)
              self.fitness_scores.append(score)
        sorted_population = sorted(
        zip(self.fitness_scores, population), key = lambda x: x[0])
        sorted_population = [x[1] for x in sorted_population]
        sorted_population = sorted_population[::-1]

        threshold = round(len(sorted_population) * self.percentage_of_parents_to_keep)
        best_individuals_in_population = sorted_population[0:threshold]
        return best_individuals_in_population

    def crossover(self,population):
        children = []
        threhold = 940
        for i in range(len(population)):
          male = random.sample(population, k=1)[0].to(self.device)
          female = random.sample(population, k=1)[0].to(self.device)
          male_first_part = male[0:threhold].to(self.device)
          male_second_part = male[threhold::].to(self.device)
          female_first_part = female[0:threhold].to(self.device)
          female_second_part = female[threhold::].to(self.device)
          child1 = torch.cat((male_first_part, female_second_part), dim=0)
          child2 = torch.cat((female_first_part, male_second_part), dim=0)
          children.append(child1)
          children.append(child2)
        return children

    def mutation(self,children):
        mutated_children = []
        for child in  children:
          mutation_base_values = torch.rand(self.count_of_initial_model_weights, device=self.device)
          mutation_values = (torch.rand(self.count_of_initial_model_weights, device=self.device) * 2 - 1) *0.1
          mutation_mask = (torch.rand(self.count_of_initial_model_weights, device=self.device) < self.mutation_rate).float()
          mutation_values = mutation_values * mutation_mask
          mutated_child = torch.add(child, mutation_values)
          mutated_children.append(mutated_child)
          if torch.isnan(mutated_child).any():
            print("NaN detected in mutated child")
            exit()
        return mutated_children


    def Glorious_Evolution(self):
        population = []
        for i in range(self.population_size):
            random_weights = torch.randn(self.count_of_initial_model_weights, device=self.device)
            population.append(random_weights)
        for generation in range(self.num_generations):
            self.fitness_scores = []
            print(f"********************\nStarting Generation {generation + 1}\n********************")

            population = self.selection(population)
            children = self.crossover(population)
            mutated_children = self.mutation(children)
          
            population = mutated_children + population
            population = population[:self.population_size]
        self.evaluate_population(population)
        sorted_population = sorted(zip(self.fitness_scores, population), key = lambda x: x[0],reverse = True)
        return sorted_population[0]


class Game:
    def __init__(self, model,device):
        self.model = model
        self.dead = False
        self.score = 0
        self.width = 10
        self.height = 20
        self.device = device
        self.fitness_scores = []
        self.movement_list = []
        self.L_piece = ((1, 1), (0, 1), (0, 0), (2, 1))
        self.J_piece = ((1, 1), (0, 1), (2, 1), (2, 0))
        self.T_piece = ((1, 1), (0, 1), (1, 0), (2, 1))
        self.I_piece = ((2, 1), (0, 1), (1, 1), (3, 1))
        self.z_piece = ((1, 1), (0, 1), (1, 2), (2, 2))
        self.s_piece = ((1, 1), (0, 2), (1, 2), (2, 1))
        self.o_piece = ((1, 1), (0, 1), (1, 2), (0, 2))
        self.block_shape = {
        "o": self.o_piece,
        "L": self.L_piece,
        "J": self.J_piece,
        "I": self.I_piece,
        "T": self.T_piece,
        "z": self.z_piece,
        "s": self.s_piece,}
    def valid_position(self,block,dy,dx):
        for x,y in block:
            if x + dx < 0 or x + dx >= self.width  or  y + dy >= self.height:
                
                return False
            if self.field[y + dy][x + dx] == 1:             
                return False
            if y + dy < 0:
              return False
        return True
    def generate_random_block(self):
        block_key = random.choice(list(self.block_shape.keys()))
        return self.block_shape[block_key]
    def field_generator(self):
        self.field_corr = [[0 for i in range(self.width)] for j in range(self.height)]
        return self.field_corr
    def rotation(self,corr):
        rotate_corr = corr
        centre = corr[0]
        rotate_corr = [((y - centre[1] + centre[0]), (-x + centre[0] + centre[1])) for x, y in
                                  rotate_corr]
        min_x = min(x for x, y in rotate_corr)
        min_y = min(y for x, y in rotate_corr)
        rotate_corr = [(x - min_x,y) for x, y in rotate_corr]
        if min_y < 0:
            offset = abs(min_y)
            shifted_corr = [(x, y + offset) for x, y in rotate_corr]
            return shifted_corr
        else:
            return rotate_corr

        return rotate_corr
    def generate_position(self):
        self.rando = self.generate_random_block()
        self.corr = self.rando
        self.movement_list = []
        for dx in range(10):
          rotate_corr = self.corr
          for _ in range(4):
              rotate_corr = self.rotation(rotate_corr)
              if self.death_detect(rotate_corr,dx) == True:
                  self.dead = True
                  return []
              if self.valid_position( rotate_corr,0,dx) == True:
                  for x , y in rotate_corr:
                      self.movement_list.append((x + dx,y))

        # while len(self.movement_list) < 40:
        #      self.movement_list.append((0, 0))
        return self.movement_list
    def splitter(self,movement_list):
        movement_list = [movement_list[i:i + 4] for i in range(0, len(movement_list), 4)]
        return movement_list
    def drop_piece(self,decision,field_corr):
        field_copy = copy.deepcopy(field_corr)
        dy = 0
        while not self.collision_detect(decision,field_copy,dy + 1) :
                dy += 1

        for x,y in decision:

            if  y + dy < self.height and 0 <= x < self.width:
              field_copy[y + dy][x] = 1
        return field_copy


    def collision_detect(self,decision,field_corr,dy):
        for x, y in decision:
            new_y = y + dy
            if not (0 <= x < self.width and  0 <= new_y < self.height):
                return True

            if field_corr[new_y][x] == 1:

                return True


        return False
           
 


    def clear_lines(self,field): # atempting  to clear lines

        for i in range(self.height):
            counter_1 = 0
            for j in range(self.width):
                if field[i][j] == 1:
                    counter_1 += 1
            if counter_1 == 10:
                print("cleared line")
                for k in range(i, 0, -1):
                    for j in range(self.width):
                        field[k][j] = field[k - 1][j]
                for j in range(self.width):
                    field[0][j] = 0
        return field
    def death_detect(self,block,dx):

        for x , y in block:
            if (x+dx) <=(self.width-1) and self.field[y][x + dx] != 0:

                self.dead = True
                return True
        return False
    def death(self):
       
        if self.dead == True:


            self.field = self.field_generator()
            self.score -= 10000000
            self.dead = False


    def hole(self,field):
        total_penalty = 0
        for i in range(self.width):

            penalty = 0
            cover = False

            for j in range(self.height ):
                if field[j][i] == 1:
                    cover = True
                elif field[j][i] == 0 and cover:
                    penalty += 1
            total_penalty += penalty
        return 100 - total_penalty
    def flatness(self,field):
        height_list = []
        for i in range(self.width):

            dy = 0
            while dy < self.height and  field[dy][i] != 1 :

                dy += 1
            height_list.append(self.height - dy)
        avg = sum(height_list)/self.width
        return 100 - ((sum((h - avg) ** 2 for h in height_list) )/self.width) ** 0.5 #standard deviation formula

    def bumbiness(self, field):
        height = len(field)
        heights = [height - next((y for y, cell in enumerate(column) if cell), height) for column in zip(*field)]
        return 100 -  sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
   
    def fit_height(self, field):
        score = 0
        for i in range(self.height):
            for j in range(self.width):
                if field[i][j] == 1:
                   score += i
        return    score
    def tetris(self, field):
        line_clears = sum(1 for row in field if all(row))
        return 5 if line_clears >= 4 else line_clears
    def fitness(self, field,line_cleared_bonus):
    
        return  1000 * line_cleared_bonus + 5 *self.flatness(field) + 10 * self.hole(field) + 5 * self.bumbiness(field) + 10 * self.fit_height(field)

    def run(self):
        self.m_input = []
        self.field = self.field_generator()
        self.death()
        self.score = 0
        for i in range(100):
            self.buffer = []
            self.m_input = []
            self.move_list = self.generate_position ()

            self.splited_movelist = self.splitter(self.move_list)
            for move in self.splited_movelist:
                simulated_field = self.drop_piece(move,self.field)
                holes = self.hole(simulated_field)
                self.m_input.append(simulated_field)
                self.buffer.append(move)
            if len(self.buffer) == 0:
                print("ERROR: Buffer is empty before indexing! Stopping execution and death.")
                print(self.field)
                #self.death
                return self.score





            else:
              field_tensor = [torch.tensor(field, dtype=torch.float32).unsqueeze(0)for field in self.m_input]
              self.m_input = torch.stack(field_tensor, dim=0)
              outputs = self.model(self.m_input)
              outputs = outputs.view(-1)
              T = 0.5
              probabilities = F.softmax((outputs/T), dim=0)
              selected_index = torch.multinomial(probabilities, 1).item()
              selected_move = self.buffer[selected_index]
              self.field = self.drop_piece(selected_move,self.field)
              line_cleared_bonus = self.tetris(self.field)
              self.field = self.clear_lines(self.field)
              self.score += self.fitness(self.field,line_cleared_bonus)

        return self.score
        outputs = self.model(self.m_input)  # outputs shape: [N, 1]






gn = GeneticAlgorithm([],torch.device("cpu"),0.6,15,0.04,40,0.6)
best_fitness, best_weights = gn.Glorious_Evolution()
torch.save(best_weights, "tetris_ai2.pth")
print(best_weights)












