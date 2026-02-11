import torch
import pygame
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

pygame.init()
input_size, hidden_size, output_size =  7000 , [128,64,32] , 1
import pygame

import random
pygame.init()
pygame.font.init()
from pygame.locals import (
    K_DOWN,
    K_z,
    K_LEFT,
    K_RIGHT,
    K_SPACE,
    K_UP,
    KEYDOWN,
    KEYUP,
    QUIT,
    USEREVENT,
)
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


    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def reconstruct_statedict(self, flattend_weights: torch.Tensor):

        state_dict = {}
        pointer = 0
        model_architecture = {}
        for key_name, weights in self.state_dict().items():
            model_architecture[key_name] = weights.shape


        for key_name, weights_shape in model_architecture.items():
            if len(weights_shape) > 1:
                count_of_weights_this_module_needs = math.prod(weights_shape)
            else:
                count_of_weights_this_module_needs = weights_shape[0]
            slice_of_selected_weights = flattend_weights[
                                        pointer: pointer + count_of_weights_this_module_needs
                                        ]

            state_dict[key_name] = torch.reshape(
                slice_of_selected_weights, model_architecture[key_name]
            )
            pointer = count_of_weights_this_module_needs + pointer
        return state_dict
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
class block_ai():
    def __init__(self):
        self.dy = 0
        self.dx = 0

        self.rotate_clockwise = False
        self.rotate_anticlockwise = False
        self.L_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/L_piece.png')
        self.o_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/O_piece.png')
        self.J_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/J_piece.png')
        self.T_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/T_piece.png')
        self.I_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/I_piece.png')
        self.z_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/z_piece.png')
        self.s_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/s_piece.png')

        self.L_piece = (((1, 1), (0, 1), (0, 0), (2, 1)), self.L_image)
        self.J_piece = (((1, 1), (0, 1), (2, 1), (2, 0)), self.J_image)
        self.T_piece = (((1, 1), (0, 1), (1, 0), (2, 1)), self.T_image)
        self.I_piece = (((2, 1), (0, 1), (1, 1), (3, 1)), self.I_image)
        self.z_piece = (((1, 1), (0, 1), (1, 2), (2, 2)), self.z_image)
        self.s_piece = (((1, 1), (0, 2), (1, 2), (2, 1)), self.s_image)
        self.o_piece = (((1, 1), (0, 1), (1, 2), (0, 2)), self.o_image)
        self.block_shape = {
            "o": self.o_piece,
            "L": self.L_piece,
            "J": self.J_piece,
            "I": self.I_piece,
            "T": self.T_piece,
            "z": self.z_piece,
            "s": self.s_piece}
    def random_block(self):
        self.L_piece = (((1, 1), (0, 1), (0, 0), (2, 1)),self.L_image)
        self.J_piece = (((1, 1), (0, 1), (2, 1), (2, 0)),self.J_image)
        self.T_piece = (((1, 1), (0, 1), (1, 0), (2, 1)),self.T_image)
        self.I_piece = (((2, 1), (0, 1), (1, 1), (3, 1)),self.I_image)
        self.z_piece = (((1, 1), (0, 1), (1, 2), (2, 2)),self.z_image)
        self.s_piece = (((1, 1), (0, 2), (1, 2), (2, 1)),self.s_image)
        self.o_piece = (((1,1),(0,1),(1,2),(0,2)),self.o_image)
        block_key = random.choice(list(self.block_shape.keys()))
        return self.block_shape[block_key]

class showing_game:
    def __init__(self,model,screen):
        self.model = model
        self.movement_list = []
        self.level = 0
        self.dead = False
        self.score = 0
        self.line_cleared = 0
        self.block_ai = block_ai()
        self.width = 10
        self.height = 20
        self.block_size = 30
        self.board_corrx = 150
        self.board_corry = 40
        self.field = self.field_generator()
        self.field_img = copy.deepcopy(self.field)
        self.WHITE = (255,255,255)
        self.surface_size = (self.width * self.block_size), (self.height * self.block_size)
        self.surface = pygame.Surface(self.surface_size)
        self.screen = screen




    def draw_grid(self):

         for i in range(self.width):
             for j in range(self.height):  # drawing the grid
                 pygame.draw.rect(self.surface, self.WHITE, pygame.Rect((i * self.block_size, j * self.block_size, self.block_size, self.block_size)),1)

         self.screen.blit(self.surface, (self.board_corrx, self.board_corry))
    def death(self):
        print("Dead ")
        print(self.field)
        self.field = self.field_generator()
        self.field_img = self.field_generator()

        self.dead = False


        return False

    def clear_lines_img(self,field):  # atempting  to clear lines

        for i in range(self.height):
            counter_1 = 0
            for j in range(self.width):
                if isinstance(field[i][j], tuple) and field[i][j][0] == 1:
                    counter_1 += 1
            if counter_1 == 10:

                for k in range(i, 0, -1):
                    for j in range(self.width):
                        field[k][j] = field[k - 1][j]
                for j in range(self.width):
                    field[0][j] = [0, None]
        return field
    def clear_lines(self,field): # atempting  to clear lines
        self.line_cleared = 0
        for i in range(self.height):
            counter_1 = 0
            for j in range(self.width):
                if field[i][j] == 1:
                    counter_1 += 1
            if counter_1 == 10:
                self.line_cleared += 1
                for k in range(i, 0, -1):
                    for j in range(self.width):
                        field[k][j] = field[k - 1][j]
                for j in range(self.width):
                    field[0][j] = 0
        return field

    def draw_field(self,field,piece_img):
        for i in range(self.height):
            for j in range(self.width):
                if isinstance(field[i][j], tuple) and field[i][j][0] == 1:
                    self.screen.blit(field[i][j][1],
                                     (j * self.block_size + self.board_corrx, i * self.block_size + self.board_corry))
    
    def generate_random_block(self):
        return self.block_ai.random_block()
    def splitter(self,movement_list):
        movement_list = [movement_list[i:i + 4] for i in range(0, len(movement_list), 4)]
        return movement_list
    def field_generator(self):
        self.field_corr = [[0 for i in range(self.width)] for j in range(self.height)]
        return self.field_corr
    def collision_detect(self,decision,field_corr,dy):
        for x, y in decision:
            new_y = y + dy
            if not (0 <= x < self.width and 0 <= new_y < self.height):
                return True

            if isinstance(field_corr[new_y][x], tuple) and field_corr[new_y][x][0] == 1:
                return True

        return False
    def collision_detect_value(self,decision,field_corr,dy):
        for x, y in decision:
            new_y = y + dy
            if not (0 <= x < self.width and 0 <= new_y < self.height):
                return True

            if field_corr[new_y][x] == 1:
                return True

        return False

    def drop_piece(self,decision,field_corr,img):

        dy = 0
        while not self.collision_detect(decision,field_corr,dy + 1) :
                dy += 1

        for x,y in decision:

            if   0 <= x < self.width:
              field_corr[y + dy][x] = (1,img)


        return field_corr
    def drop_piece_value(self,decision,field_corr):
        field_copy = copy.deepcopy(field_corr)
        dy = 0
        while not self. collision_detect_value(decision,field_copy,dy + 1) :
                dy += 1

        for x,y in decision:

            if  0 <= x < self.width:
              field_copy[y + dy][x] = 1


        return field_copy


    def valid_position(self, block, dy, dx):
        for x, y in block:
            if x + dx < 0 or x + dx >= self.width:
                print(f"Move rejected: Out of bounds -> {block} at ({dx}, {dy})")
                return False


        return True
    def death_detect(self,block,dx):
        print(self.field)
        for x , y in block:
            if (x+dx) <=(self.width-1) and self.field[y][x + dx] != 0:

                self.dead = True
                return True
    def rotation(self, corr):
        rotate_corr = corr
        centre = corr[0]
        rotate_corr = [((y - centre[1] + centre[0]), (-x + centre[0] + centre[1])) for x, y in
                       rotate_corr]
        min_x = min(x for x, y in rotate_corr)
        min_y = min(y for x, y in rotate_corr)
        rotate_corr = [(x - min_x, y) for x, y in rotate_corr]
        if min_y < 0:
            offset = abs(min_y)
            shifted_corr = [(x, y + offset) for x, y in rotate_corr]
            return shifted_corr
        else:
            return rotate_corr
    def generate_position(self):
        self.rando = self.generate_random_block()
        self.corr = self.rando[0]
        piece_img = self.rando[1]
        self.movement_list = []
        for dx in range(10):
            rotate_corr = self.corr
            for _ in range(4):
                rotate_corr = self.rotation(rotate_corr)
                if self.death_detect(rotate_corr, dx):
                    self.death()
                if self.valid_position(rotate_corr, 0, dx) == True:
                    for x, y in rotate_corr:
                        self.movement_list.append((x + dx, y))
        return self.movement_list,piece_img


    def display_score(self):
        if self.line_cleared != 0:
            self.score += (self.level + 1)*(40 * (2.5 +((self.line_cleared - 1) * 0.5)))
        self.level = int(self.score / 100)
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(str(int(self.score)), True, (255, 255, 255))
        self.screen.blit(text_surface, (5, 0))
    def show_game(self):


        self.draw_grid()
        self.buffer = []
        self.buffer2 = []
        self.m_input = []
        self.movement_list , img = self.generate_position()
        print("Generated candidate moves:", self.movement_list)


        self.splited_movelist = self.splitter(self.movement_list)
        for move in self.splited_movelist:
            simulated_field = self.drop_piece_value(move, self.field)
            self.m_input.append(self.clear_lines(simulated_field))
            self.buffer.append(move)


        if len(self.buffer) == 0:
            print("ERROR: Buffer is empty before indexing! Stopping execution.DEAD")
            print(self.field)
            self.death()
            print(self.buffer)

        field_tensor = [torch.tensor(field, dtype=torch.float32).unsqueeze(0) for field in self.m_input]
        self.m_input = torch.stack(field_tensor, dim=0)
        outputs = self.model(self.m_input)  # outputs has shape [N, 1]
        outputs = outputs.view(-1) 

        probabilities = F.softmax(outputs/0.005 , dim=0)
        selected_index = torch.multinomial(probabilities, 1).item()



        selected_move = self.buffer[selected_index]


        self.field = self.clear_lines(self.field)


        self.field_img = self.drop_piece(selected_move, self.field_img,img)
        self.field = self.drop_piece_value(selected_move, self.field)



        self.field = self.clear_lines(self.field)
        self.field_img  = self.clear_lines_img(self.field_img)

        self.draw_field(self.field_img,img)
       
        self.display_score()
        




    def run(self):

        while True:

            self.show_game()



class Board():
    def __init__(self,game):
        self.game = game
        self.width = 10
        self.height = 21
        self.block_size = 30
        self.board_corrx = 700
        self.board_corry = 40
        self.WHITE = (255,255,255)
        self.surface_size = (self.width * self.block_size), (self.height * self.block_size)
        self.surface = pygame.Surface(self.surface_size)

    def collision(self,block_corr):

        for x, y in block_corr:


            if y >= (self.height - 1) or self.game.field.field_corr[y + 1][x] == 1:
                return True
        return False

    def clear_lines(self):  # atempting  to clear lines

        for i in range(self.height):
            counter_1 = 0
            for j in range(self.width):
                if isinstance(self.game.field.field_corr_img[i][j], list) and self.game.field.field_corr_img[i][j][0] == 1:
                    counter_1 += 1
            if counter_1 == 10:

                for k in range(i, 0, -1):
                    for j in range(self.width):
                        self.game.field.field_corr_img[k][j] = self.game.field.field_corr_img[k - 1][j]
                for j in range(self.width):
                    self.game.field.field_corr_img[0][j] =[0,None]


class Scoring_system():
    def __init__(self,game):
        self.game = game
        self.level = 0
        self.score = 0
        self.lineclear = 0
    def upgrade_level(self):
        self.level = int(self.score/ 100)
    def clearing_line_score(self):

        if self.lineclear != 0:
            self.score += (self.level + 1)*(40 * (2.5 +((self.lineclear - 1) * 0.5)))
            self.lineclear = 0



class block():
    def __init__(self):

        self.dy = 0
        self.dx = 0
        self.rotate_clockwise = False
        self.rotate_anticlockwise = False
        self.L_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/L_piece.png')
        self.o_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/O_piece.png')
        self.J_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/J_piece.png')
        self.T_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/T_piece.png')
        self.I_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/I_piece.png')
        self.z_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/z_piece.png')
        self.s_image = pygame.image.load('/Users/wingcheungpong/PycharmProjects/tetrisAI/Tetrimino/s_piece.png')
        self.L_piece = (((1, 1), (0, 1), (0, 0), (2, 1)), self.L_image)
        self.J_piece = (((1, 1), (0, 1), (2, 1), (2, 0)), self.J_image)
        self.T_piece = (((1, 1), (0, 1), (1, 0), (2, 1)), self.T_image)
        self.I_piece = (((2, 1), (0, 1), (1, 1), (3, 1)), self.I_image)
        self.z_piece = (((1, 1), (0, 1), (1, 2), (2, 2)), self.z_image)
        self.s_piece = (((1, 1), (0, 2), (1, 2), (2, 1)), self.s_image)
        self.o_piece = (((1, 1), (0, 1), (1, 2), (0, 2)), self.o_image)
        # ((1, 0), (0, 0), (1, 0), (2, 1))
        self.block_shape = {
            "o": self.o_piece,
            "L": self.L_piece,
            "J": self.J_piece,
            "I": self.I_piece,
            "T": self.T_piece,
            "z": self.z_piece,
            "s": self.s_piece,}


    def random_block(self):
        self.L_piece = (((1, 1), (0, 1), (0, 0), (2, 1)),self.L_image)
        self.J_piece = (((1, 1), (0, 1), (2, 1), (2, 0)),self.J_image)
        self.T_piece =(((1, 1), (0, 1), (1, 0), (2, 1)),self.T_image)
        self.I_piece = (((2, 1), (0, 1), (1, 1), (3, 1)),self.I_image)
        self.z_piece = (((1, 1), (0, 1), (1, 2), (2, 2)),self.z_image)
        self.s_piece = (((1, 1), (0, 2), (1, 2), (2, 1)),self.s_image)
        self.o_piece = (((1,1),(0,1),(1,2),(0,2)),self.o_image)
        block_key = random.choice(list(self.block_shape.keys()))
        return self.block_shape[block_key]



    def rotation(self):
        if self.rotate_clockwise is True:  # rotation system
            centre = self.corr[0]
            self.corr = [((y - centre[1] + centre[0]), (-x + centre[0] + centre[1])) for x, y in
                                  self.corr]
        if self.rotate_anticlockwise is True:
            centre = self.corr[0]
            self.corr = [((-y + centre[1] + centre[0]), (x - centre[0] + centre[1])) for x, y in
                                  self.corr]
        self.rotate_clockwise = False
        self.rotate_anticlockwise = False
        if  self.Boardline_check() is False:
            self.corr = self.old_corr

    def Boardline_check(self):
        for x, y in self.corr:
            if x >= 10 or x < 0 or y > 20:
                return False
        return True
    def x_collision(self,dx):
        for x, y in self.corr:
            if 0 <= x+dx <= (self.game.board.width-1) and self.game.field.field_corr[y][x + dx] == 1 :
                return True
        return False
    def new_block_checker(self):
        for x, y in self.corr:
            if y >= 20 or self.game.field.field_corr[y][x] == 1 :
                return True
        return False



class current_block(block):
    def __init__(self,game):
        super().__init__()
        self.game = game
        self.corr = []
        self.block_img = None
        self.l_time = 0
        self.delay =  2000
        self.c_time = 0
        self.g_time = 0
    def generate(self):

        self.rando = self.random_block()
        self.corr = self.rando[0]
        self.block_img = self.rando[1]
        self.corr = [(x + 3, y) for x, y in self.corr]
        for x,y in self.corr:
            if self.game.field.field_corr[y][x] == 1:
                self.game.death = True


    def soft_drop(self):
        self.c_time = pygame.time.get_ticks()

        if self.c_time - self.g_time  > self.delay:

            self.dy = 1
            self.g_time  = self.c_time



    def position(self):
        self.old_corr = self.corr
        self.corr = [(x + self.dx , y + self.dy) for x, y in self.corr]
        self.dx, self.dy = 0, 0






class field(Board):
    def __init__(self,game):
        super().__init__(game)
        self.field_corr = [[0 for i in range(self.width)] for j in range(self.height)]
        self.field_corr_img = [[0 for i in range(self.width)] for j in range(self.height)]
    def store(self):
        if self.game.board.collision(self.game.current_block.corr) is True:
            for x, y in self.game.current_block.corr:
                self.field_corr[y][x] = 1

    def store_img(self):
        if self.game.board.collision(self.game.current_block.corr) is True:
            for x, y in self.game.current_block.corr:
                self.field_corr_img[y][x] = [1,self.game.current_block.block_img]
            self.game.current_block.generate()
    def clear(self):
        for i in range(self.height):
            self.counter_1 = 0
            for j in range(self.width):
                if self.field_corr[i][j] == 1:
                    self.counter_1 += 1
            if self.counter_1 == 10:
                self.game.scoring_system.lineclear +=1
                for k in range(i, 0, -1):
                    for j in range(self.width):
                        self.field_corr[k][j] = self.field_corr[k - 1][j]
                for j in range(self.width):
                    self.field_corr[0][j] = 0
    def clear_field(self):
        self.field_corr = [[0 for i in range(self.width)] for j in range(self.height)]
        self.field_corr_img = [[0 for i in range(self.width)] for j in range(self.height)]
    def draw_field(self):
        for i in range(self.game.board.height):
            for j in range(self.game.board.width):
                if isinstance(self.field_corr_img[i][j], list) and  self.field_corr_img[i][j][0] == 1:
                    self.game.screen.blit(self.field_corr_img[i][j][1],
                                     (j * self.game.board.block_size + self.game.board.board_corrx, i * self.game.board.block_size + self.game.board.board_corry))
class grid(Board):

    def __init__(self,game):
        super().__init__(game)

    def draw_grid(self):

        for i in range(self.width):
            for j in range(self.height):  # drawing the grid
                pygame.draw.rect(self.surface, self.WHITE, pygame.Rect((i * self.block_size, j * self.block_size, self.block_size, self.block_size)),1)


class Game():
    def __init__(self,screen):
        self.screen = screen
        self.board = Board(self)
        self.grid = grid(self)
        self.scoring_system = Scoring_system(self)
        self.current_block = current_block(self)
        self.field = field(self)
        self.death = False
    def new_game(self):
        self.grid = grid(self)
    def gameover(self):
        if self.death == True:
            self.field.clear_field()
            self.death = False
    def new_block(self):
        self.current_block = current_block(self)

    def new_blockGen(self):
        if self.board.collision(self.current_block.corr) is True:
            self.current_block.generate()
    def display_score(self):
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(str(int(self.scoring_system.score)), True, (255, 255, 255))
        self.screen.blit(text_surface, (700, 0))
    def draw_block(self):
        for x, y in self.current_block.corr:
            print(x,y)
        for x, y in self.current_block.corr:  # drawing the block
            self.screen.blit(self.current_block.block_img, (self.board.board_corrx  + x  * self.board.block_size , self.board.board_corry + y * self.board.block_size))



    def update(self):

        self.grid.draw_grid()
        self.screen.blit(self.grid.surface, (self.board.board_corrx, self.board.board_corry ))
        self.control()
        self.current_block.soft_drop()
        self.current_block.rotation()
        self.field.store()
        self.field.store_img()
        self.current_block.position()
        self.draw_block()
        self.field.draw_field()
        self.board.clear_lines()
        self.field.clear()
        self.scoring_system.clearing_line_score()
        self.scoring_system.upgrade_level()
        self.display_score()
        self.gameover()


    def control(self):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():# key handling
            if event.type == pygame.QUIT:

                pygame.quit()
            if event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    self.current_block.dx = 1
                elif event.key == K_UP:
                    self.current_block.rotate_anticlockwise = True
                elif event.key == K_z:
                    self.current_block.rotate_clockwise = True
                elif event.key == K_LEFT:
                    self.current_block.dx = -1
                elif event.key == K_SPACE:
                    self.hard_drop = True

        if  self.current_block.c_time - self.current_block.l_time > 100:  # soft drop
            if keys[K_DOWN]:
                self.current_block.dy = 1
            self.current_block.l_time = self.current_block.c_time

        if  (self.current_block.Boardline_check() is False) or self.current_block.x_collision(self.current_block.dx) is True:
            self.current_block.corr = self.current_block.old_corr
    def run(self):
        self.current_block.generate()
        while True:

            self.update()
class Tetris_game:
    def __init__(self):
        self.model = TetrisNN()
        load = torch.load("/Users/wingcheungpong/PycharmProjects/tetrisAI/.venv/tetris_ai2.0.pth")
        reconstructed_load = self.model.reconstruct_statedict(load)
        self.model.load_state_dict(reconstructed_load)
        self.model.eval()
        self.screen = pygame.display.set_mode((1200, 2400))
        self.human_game = Game(self.screen)
        self.Ai_game = showing_game(self.model,self.screen)
    def update_ai(self):
        self.Ai_game.show_game()
    def update_human(self):
        self.human_game.update()

    def run(self):
        clock = pygame.time.Clock()
        human_update_interval = 0
        ai_update_interval = 200
        human_timer = 0
        ai_timer = 0

        self.human_game.current_block.generate()
        while True:
            dt = clock.tick(60)
            human_timer += dt
            ai_timer += dt
            if ai_timer >= ai_update_interval:
                self.screen.fill((0, 0, 0))
                self.update_ai()  # Update AI board


                ai_timer = 0

            if human_timer >= human_update_interval:

                self.update_human()  # Update human board
                human_timer = 0

            pygame.display.flip()

tetris = Tetris_game()
tetris.run()

