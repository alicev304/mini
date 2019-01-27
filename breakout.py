import math
import pygame
 
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
 
block_width = 23
block_height = 15
 
class Block(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()
        self.image = pygame.Surface([block_width, block_height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Ball(pygame.sprite.Sprite):
    speed = 10.0
    x = 0.0
    y = 180.0
    direction = 200
    width = 10
    height = 10
 
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(white)
        self.rect = self.image.get_rect()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
 
    def _bounce(self, diff):
        self.direction = (180 - self.direction) % 360
        self.direction -= diff
 
    def _update(self):
        direction_radians = math.radians(self.direction)
        self.x += self.speed * math.sin(direction_radians)
        self.y -= self.speed * math.cos(direction_radians)
        self.rect.x = self.x
        self.rect.y = self.y
        if self.y <= 0:
            self._bounce(0)
            self.y = 1
        if self.x <= 0:
            self.direction = (360 - self.direction) % 360
            self.x = 1
        if self.x > self.screenwidth - self.width:
            self.direction = (360 - self.direction) % 360
            self.x = self.screenwidth - self.width - 1
        if self.y > 600:
            return True
        else:
            return False
 
class Player(pygame.sprite.Sprite):
 
    def __init__(self):
        super().__init__()
        self.width = 75
        self.height = 15
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((white))
        self.rect = self.image.get_rect()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
        self.rect.x = 0
        self.rect.y = self.screenheight-self.height
 
    def _update(self):
        pos = pygame.mouse.get_pos()
        self.rect.x = pos[0]
        if self.rect.x > self.screenwidth - self.width:
            self.rect.x = self.screenwidth - self.width
 
pygame.init()
 
screen = pygame.display.set_mode([800, 600])
pygame.display.set_caption('BREAKOUT')
pygame.mouse.set_visible(0)
font = pygame.font.Font(None, 36)
background = pygame.Surface(screen.get_size())
 
blocks = pygame.sprite.Group()
balls = pygame.sprite.Group()
allsprites = pygame.sprite.Group()
 
player = Player()
allsprites.add(player)
 
ball = Ball()
allsprites.add(ball)
balls.add(ball)
 
top = 80
blockcount = 32
 
for row in range(5):
    for column in range(0, blockcount):
        block = Block(blue, column * (block_width + 2) + 1, top)
        blocks.add(block)
        allsprites.add(block)
    top += block_height + 2

clock = pygame.time.Clock()
game_over = False
exit_program = False
 
while not exit_program:
 
    clock.tick(30)
    screen.fill(black)
 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
 
    if not game_over:
        player._update()
        game_over = ball._update()
 
    if game_over:
        text = font.render("Game Over", True, white)
        textpos = text.get_rect(centerx = background.get_width() / 2)
        textpos.top = 300
        screen.blit(text, textpos)
 
    if pygame.sprite.spritecollide(player, balls, False):
        diff = (player.rect.x + player.width / 2) - (ball.rect.x+ball.width / 2)
 
        ball.rect.y = screen.get_height() - player.rect.height - ball.rect.height - 1
        ball._bounce(diff)
 
    deadblocks = pygame.sprite.spritecollide(ball, blocks, True)
 
    if len(deadblocks) > 0:
        ball._bounce(0)
 
        if len(blocks) == 0:
            game_over = True
            
    allsprites.draw(screen)
    pygame.display.flip()
 
pygame.quit()





































