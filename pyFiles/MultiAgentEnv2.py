from pettingzoo.utils.env import ParallelEnv
import networkx as nx
import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import matplotlib.pyplot as plt
from pettingzoo.test import parallel_api_test

class CustomEnvironment(ParallelEnv):
    
    metadata = {
        "name": "custom_graph_environment_v1",
    }
    def __init__(self):
        # this is the graph
        self.g_env = nx.read_graphml('g1.gml')
        self.g_no_node = len(self.g_env.nodes())
        self.node_list = list(self.g_env.nodes())
        
        # A dictionary that relates each discrete value in the observation
        # space to a node obtained from the graph
        
            # 1. Create empty dictionary
        self.node_dict = {}
        self.node_inv_dict = {}
        
            # 2. relating the key to the value of the node
        for key,value in enumerate(self.node_list):
            self.node_dict[key] = value 
            
            # 3. relating the value to the key ( used later on )
        self.node_inv_dict = {value: key for key, value in self.node_dict.items()}
        
        # sets the maximum steps after which the program will terminate 
        self.max_steps = 100
        self.step_now = 0
        
        self.no_of_thieves = 1
        self.no_of_police = 3 
        self.possible_thieves = ['thief_'+str(r) for r in range (self.no_of_thieves)]
        self.possible_police = ['police_'+str(r) for r in range (self.no_of_police)]
        self.possible_agents = self.possible_thieves + self.possible_police
        
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # {'player_0': 0, 'player_1': 1}

        
        self._action_spaces = {agent: Discrete(4) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(self.g_no_node**len(self.possible_agents)) for agent in self.possible_agents
        }
        
        self.agents = [i for i in self.possible_agents]
        self.infos = {agent: {} for agent in self.possible_agents}
        self.state = {agent: None for agent in self.possible_agents}
        
        # not utilized
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        
                
        # this is extra things for visualization you do not need to know
        graph = self.g_env
        position = list(graph.nodes())
        position = [self.str_to_tuple(name) for name in position]
        pos = dict(zip(graph.nodes(), position))
        self.node_positions = pos
        
        # sets the current state of agents
        # sets the thief to position 6
#         self.state['player_0'] = self.node_dict[6]
        
        # sets the police to position 9
#         self.state['player_1'] = self.node_dict[9]
        self.terminations = {agent:False for agent in self.possible_agents}

        for thief in self.possible_thieves:
            self.state[thief]= self.node_dict[6]
            
        for police in self.possible_police:
            self.state[police]= self.node_dict[9]
            
        #forced by testing api
        self.action_spaces = self._action_spaces
        self.observation_spaces = self._observation_spaces
        self.terminations = {a:False for a in self.possible_agents}
        
        
    def reset(self, seed=None, options=None):
        self.agents = [i for i in self.possible_agents]
        self.timestep = None
        self.state = {agent: None for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.step_now = 0
        self.terminations = {agent:False for agent in self.possible_agents}

#         ran = np.random.randint(0,self.g_no_node)     # later on if you have to start at random places
        for thief in self.possible_thieves:
            self.state[thief]= self.node_dict[6]
            
        for police in self.possible_police:
            self.state[police]= self.node_dict[9]
            
        return self.state

    def step(self, actions):
        
        # observation returns the next state of the agents
        # for each action selected for the agent the observations sould be sent back
        # the impletentation is bad change if possible
        terminations = self.terminations
        
#         rewards = {"player_0":None, "player_1":None}
        rewards = {}
        for agent in self.possible_agents:
            rewards[agent] = None  
        
        # movement of the thief and the police according to the action and their rewards       
        for agent in self.possible_agents:
            temp_neighbours = []
            for neighbour in self.g_env.neighbors(self.state[agent]):
                temp_neighbours.append(neighbour)
    
            if actions[agent] < len([i for i in self.g_env.neighbors(self.state[agent])]):
                rewards[agent] = -2;
                self.state[agent] = temp_neighbours[actions[agent]]
                

            elif actions[agent] == len([i for i in self.g_env.neighbors(self.state[agent])]):
                rewards[agent] = -1;
                
            else:
                rewards[agent] = -10;
                
        # termination if on the same place
        for police in self.possible_police:
            for thief in self.possible_thieves:
                if self.state[police] == self.state[thief]:
                    terminations[police]: True
                    rewards[police] = 10 
                    print('terminating step')

#         extra reward for thief for running away
#         if (self.step_now>7):
#             for thief in self.possible_thieves:
#                 rewards[thief] += 20
#             for police in self.possible_police:
#                 rewards[police] -= 20
#             for police in self.possible_police:
#                 for thief in self.possible_thieves:
#                     terminations[thief]: True

        self.step_now += 1
        self.terminations  = terminations
        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}
        observations = {a: self.state[a] for a in self.agents}
        truncations = {a: None for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        pass
    
    # only works for 2 players
    def temp_render(self,episode):
        
        nx.draw(self.g_env, self.node_positions,node_size=300)
        nx.draw_networkx_labels(self.g_env, self.node_positions,labels = self.node_inv_dict,font_color='black' )
        
        # drawing the agents
        for agent in self.agents:
            x,y = self.node_positions[str(self.state[agent])]
            if agent in self.possible_police:
                plt.scatter(x, y, s=550, c='yellow')
            else:
                plt.scatter(x, y, s=450, c='red')    
        
        
        filename = f"images/Multi_Env_2{episode}_{self.step_now}.png"
        
         
        plt.savefig(filename)
        plt.show()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def str_to_tuple(self,string):
        return tuple(float(x) for x in string.strip('()').split(','))
    
    def possible_move_range(self,player):
        return len(list(self.g_env.neighbors(self.state[player])))