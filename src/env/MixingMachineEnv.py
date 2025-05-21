import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Discrete, Dict
import numpy as np
from .Reward import get_reward
from gymnasium.envs.registration import register


class InterfaceEnv(gym.Env):
    
    def __init__(self, df):
        super(InterfaceEnv, self).__init__()
        
        self.num_mix_slots = 16 # Number of mix slots
        self.num_additives = 6 #Number of possible additives amount

        
        self.action_space = MultiDiscrete([self.num_mix_slots + 1] * self.num_mix_slots + [self.num_additives, 2]) #Action space: (16 possible mix + 1 for hidden mix)*16 mix slots + 6 additives + 2 containers

        obs_dict = {
            'Time': Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32),  #Time in decimal hours
            'User': Discrete(30)   #Total of users
        }
        self.observation_space = Dict(obs_dict) #Conversion to dictionary space
        self.state = None   #Initialize state
        self.max_steps = 1
        self.df = df
######################################################################
    def _state_to_obs(self, state):
        #Function to convert the state to the observation dictionary
        obs = {
            'Time': np.array([state[-2]], dtype=np.float32),
            'User': int(state[-1])
        }
        return obs
    
    def get_initial_state(self):
        #Function to get the initial state from data
        sample = self.df[['encoded_user', 'hora_decimal']].sample(1)
        initepoch = sample.iloc[0, 1]
        user = sample.iloc[0, 0]
        initial_mixes = list(range(self.num_mix_slots))
        state = np.array(initial_mixes + [3, 0, initepoch, user])
        return state
    
    def get_final_selection(self):
        #Function to get the final selection of the user from data
        selection = list(self.df[(self.df['encoded_user'] == self.state[-1]) & (self.df['hora_decimal'] == self.state[-2])][['encoded_mixture','additive','encoded_container']].iloc[0])

        return selection
    
######################################################################   
    def step(self, action):
        
        selection = self.get_final_selection()

        reward = float(get_reward(selection, action))

        new_state = np.concatenate((action, self.state[-2:]))   #create new state with action [mix, additive, container] and [time, user]

        self.state = new_state
        done = True

        obs =  self._state_to_obs(self.state)   #return ¡new state as observation

        return obs, reward, done, False, {}
    


    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.state = self.get_initial_state() #Get initial state from data

        obs = self._state_to_obs(self.state)
            
        return obs, {}        
    
    def render(self):
        pass


if __name__ == "__main__":

#Register the environment in Gymnasium
    register(
        id="InterfaceEnv-v0",  
        entry_point="RLHF_adaptation.env.MixingMachineEnv:InterfaceEnv",  
    )
    print("Environment registered successfully.")
