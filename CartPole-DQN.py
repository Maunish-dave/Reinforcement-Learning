from numpy.__config__ import show
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import gym


class DeepQNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
        super(DeepQNetwork,self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims,self.n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions
    

class Agent():
    def __init__(self,gamma,epsilon,lr,input_dims,fc1_dims,fc2_dims,batch_size,n_actions,
    max_memory_size = 100000,epsilon_dec = 5e-4,epsilon_end=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr  = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(self.lr,self.input_dims,self.fc1_dims,self.fc2_dims,self.n_actions)

        self.state_memory = np.zeros((self.max_mem,*self.input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem,*self.input_dims),dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem,dtype=np.int32)
        self.reward_memory= np.zeros(self.max_mem,dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem,dtype=np.bool)

    
    def store_transition(self,state,state_,reward,done,action):
        index = self.mem_cntr % self.max_mem
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.action_memory[index] = action
        self.mem_cntr += 1
    
    def choose_action(self,observation):
        state = torch.tensor([observation],dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_size = min(self.max_mem,self.mem_cntr)

        batch = np.random.choice(max_size,self.batch_size,replace=False)

        batch_index = np.arange(self.batch_size,dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next,dim=1)[0]

        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()
        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_end \
            else self.epsilon_end


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    show_every = 50
    episodes = 500
    brain = Agent(gamma=0.99,epsilon=0.1,lr=0.1,input_dims=[len(env.observation_space.high)],fc1_dims=256,fc2_dims=256,batch_size=64,n_actions=env.action_space.n)
    scores = [0]
    for episode in range(episodes):
        render = False
        if episode % show_every == 0:
            print(f"episode {episode} avg score {np.mean(scores[-show_every:])} max score {np.max(scores[-show_every:])} min score {np.min(scores[-show_every:])}")
            render = True
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action  = brain.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            brain.store_transition(observation,observation_,reward,done,action)
            score += reward
            if render:
                env.render()
            brain.learn()
            observation = observation_
        scores.append(score)
    env.close()













