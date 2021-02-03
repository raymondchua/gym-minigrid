from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class ClassicGridWorldEnv(MiniGridEnv):
    """
    Classic grid world environment with four possible actions: 
    move up, move down, move left and move right. Agent has no direction
    in this env. 
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Move left, right, upward or downward
        left = 0
        right = 1
        up = 2
        down = 3

    def __init__(self, size, max_steps=200, agent_view=3, agent_pos=None, goal_pos=None):

        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view,
        )

        # Action enumeration for this environment
        self.actions = self.Actions

    def _gen_grid(self, width, height):
    
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)
        else:
            # self.agent_pos = (1, height-2)
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
             # Place a goal square in the top-right corner
            self.goal_pos = np.array((width - 2, 1))
            self.put_obj(Goal(), *self.goal_pos)

        self.mission = 'Reach the goal'

    def _reward(self):
        """
        Return a reward of 1 when the agent reaches the goal site
        """

        return 1

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the position on the left of the agent
        left_pos = self.left_pos

        # Get the position on the right of the agent
        right_pos = self.right_pos

        # Get the position on the upwards of the agent
        up_pos = self.up_pos

        # Get the position on the downwards of the agent
        down_pos = self.down_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Get the contents of the cell in left of the agent
        left_cell = self.grid.get(*left_pos)

        # Get the contents of the cell in right of the agent
        right_cell = self.grid.get(*right_pos)

         # Get the contents of the cell in downwards of the agent
        up_cell = self.grid.get(*up_pos)

        # Get the contents of the cell in downwards of the agent
        down_cell = self.grid.get(*down_pos)

        # Move left
        if action == self.actions.left:
            if left_cell == None or left_cell.can_overlap():
                self.agent_pos = left_pos
            if left_cell != None and left_cell.type == 'goal':
                done = True
                reward = self._reward()

        # Move right
        elif action == self.actions.right:
            if right_cell == None or right_cell.can_overlap():
                self.agent_pos = right_pos
            if right_cell != None and right_cell.type == 'goal':
                done = True
                reward = self._reward()

        # Move upward
        elif action == self.actions.up:
            if up_cell == None or up_cell.can_overlap():
                self.agent_pos = up_pos
            if up_cell != None and up_cell.type == 'goal':
                done = True
                reward = self._reward()

        # Move downward
        elif action == self.actions.down:
            if down_cell == None or down_cell.can_overlap():
                self.agent_pos = down_pos
            if down_cell != None and down_cell.type == 'goal':
                done = True
                reward = self._reward()

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render_wo_agent_dir(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def gen_obs(self):
        """
        Generate the agent's view which is x-y coordinates and head direction
        """

        # grid, vis_mask = self.gen_obs_grid()

        # # Encode the partially observable view into a numpy array
        # image = grid.encode(vis_mask)

        # assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # # Observations are dictionaries containing:
        # # - an image (partially observable view of the environment)
        # # - the agent's direction/orientation (acting as a compass)
        # # - a textual mission string (instructions for the agent)
        obs = {
            'x': self.agent_pos[0],
            'y': self.agent_pos[1],
        }

        return obs

    



class ClassicGridWorldS7Env(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=7)

class ClassicGridWorldS7EnvGoalBL(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=7, goal_pos=(1,5))

class ClassicGridWorldS9Env(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=9)

class ClassicGridWorldS9EnvGoalBL(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=9, goal_pos=(1,7))

class ClassicGridWorldS11Env(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=11)

class ClassicGridWorldS11EnvGoalBL(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=11, goal_pos=(1,9))

class ClassicGridWorldS12Env(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=12)

class ClassicGridWorldS12EnvGoalBL(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=12, goal_pos=(1,10))

class ClassicGridWorldS12max20kEnv(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=12, max_steps=20000)

class ClassicGridWorldS12max20kEnvGoalBL(ClassicGridWorldEnv):
    def __init__(self):
        super().__init__(size=12, max_steps=20000, goal_pos=(1,10))

register(
    id='MiniGrid-ClassicGridWorldS7-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS7Env'
)

register(
    id='MiniGrid-ClassicGridWorldS7BLG-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS7EnvGoalBL'
)

register(
    id='MiniGrid-ClassicGridWorldS9-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS9Env'
)

register(
    id='MiniGrid-ClassicGridWorldS9BLG-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS9EnvGoalBL'
)

register(
    id='MiniGrid-ClassicGridWorldS11-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS11Env'
)

register(
    id='MiniGrid-ClassicGridWorldS11BLG-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS11EnvGoalBL'
)

register(
    id='MiniGrid-ClassicGridWorldS12-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS12Env'
)

register(
    id='MiniGrid-ClassicGridWorldS12BLG-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS12EnvGoalBL'
)

register(
    id='MiniGrid-ClassicGridWorldS12max20k-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS12max20kEnv'
)

register(
    id='MiniGrid-ClassicGridWorldS12max20kBLG-v0',
    entry_point='gym_minigrid.envs:ClassicGridWorldS12max20kEnvGoalBL'
)




