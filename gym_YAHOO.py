# https://www.secmem.org/blog/2020/02/08/snake-dqn/
class AnomalyAction :
    A_ANOMALY = 0
    A_NORMAL = 1
class Snake:
    ACTIONS = {
        AnomalyAction.A_ANOMALY: 'a anomaly',
        AnomalyAction.A_NORMAL: 'a normal'
    }

    def __init__(self, level_loader, block_pixels=30):
        self.level_loader = level_loader
        self.block_pixels = block_pixels

        self.field_height, self.field_width = self.level_loader.get_field_size()

        pygame.init()
        self.screen = pygame.display.set_mode((
            self.field_width * block_pixels,
            self.field_height * block_pixels
        ))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.state_transition = SnakeStateTransition( ### 수정!!
            self.level_loader.get_field_size(),
            self.level_loader.get_field(),
            self.level_loader.get_num_feed(),
            self.level_loader.get_initial_head_position(),
            self.level_loader.get_initial_tail_position(),
            self.level_loader.get_initial_snake()
        )
        self.tot_reward = 0
        return self.state_transition.get_state()

    def step(self, action):
        reward, done = getattr(self.state_transition, Snake.ACTIONS[action])()
        self.tot_reward += reward
        return self.state_transition.get_state(), reward, done

    def get_length(self):
        return self.state_transition.get_length()

    def quit(self):
        pygame.quit()

    def render(self, fps):
        pygame.display.set_caption('length: {}'.format(self.state_transition.get_length()))
        pygame.event.pump()
        self.screen.fill((255, 255, 255))

        for i in range(self.field_height):
            for j in range(self.field_width):
                cp = get_color_points(self.state_transition.field[i][j])
                if cp is None:
                    continue
                pygame.draw.polygon(
                    self.screen,
                    cp[0],
                    (cp[1] + [j, i])*self.block_pixels
                )

        pygame.display.flip()
        self.clock.tick(fps)

    def save_image(self, save_path):
        pygame.image.save(self.screen, save_path)