# Helper to fold two DDPG agents into one

from agent import Agent

import torch

class DualDDPGAgent:
    def __init__(self, state_size, action_size, random_seed):
        self.agents = [Agent(state_size, action_size, 1, random_seed) for _ in range(2)]
    
    def step(self, states, actions, rewards, next_states, dones):
        for i in range(2):
            self.agents[i].step(states[i],actions[i],rewards[i],next_states[i],dones[i])
    
    def act(self, states, eps=0, add_noise=True):
        result = []
        for i in range(2):
            action = self.agents[i].act(states[i],eps,add_noise)
            result.append(action)
        return result
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def save(self):
        for i in range(len(self.agents)):       
            torch.save(self.agents[i].actor_local.state_dict(), str(i) + 'st_agent_checkpoint_actor.pth')
            torch.save(self.agents[i].critic_local.state_dict(), str(i) + 'st_agent_checkpoint_critic.pth')