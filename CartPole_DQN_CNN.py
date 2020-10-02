import gym
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt



class DeeepQNetworkCNN(nn.Module):
    def __init__(self,lr,batch_size,input_channels,w,h,conv_dim1,conv_dim2,conv_dim3,n_actions):
        super(DeeepQNetworkCNN,self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.w = w 
        self.h = h
        self.input_channels = input_channels
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(self.input_channels,self.conv_dim1,kernel_size=5,stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_dim1)
        self.conv2 = nn.Conv2d(self.conv_dim1,self.conv_dim2,kernel_size=5,stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_dim2)
        self.conv3 = nn.Conv2d(self.conv_dim2,self.conv_dim3,kernel_size=5,stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_dim3)

        def conv2d_size_out(size,kernel_size=5,stride=2,layers=3):
            for _ in range(layers):
                size = (size - (kernel_size -1) -1) // stride + 1
            return size

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.convh = conv2d_size_out(self.h)
        self.convw = conv2d_size_out(self.w)
        # print(self.convh)
        self.linear1 = nn.Linear(self.conv_dim3 * self.convw * self.convh,self.n_actions)
        self.to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),self.lr)


    def forward(self,xb):
        x = F.relu(self.bn1(self.conv1(xb)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.linear1(x.view(x.size(0),-1))



class Agent:
    def __init__(self,lr,gamma,batch_size,input_channels,w,h,conv_dim1,conv_dim2,conv_dim3,n_actions,
        max_mem_size=10000,epsilon=1.0,eps_dec=5e-4,eps_end=0.01):
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.input_channes = input_channels
        self.w = w
        self.h = h
        self.conv_dim1= conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.n_actions = n_actions
        self.action_space = [x for x in range(n_actions)]
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_end
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.iter_cntr = 0

        self.Q_eval = DeeepQNetworkCNN(self.lr,self.batch_size,self.input_channes
        ,self.w,self.h,self.conv_dim1,self.conv_dim2,self.conv_dim3,self.n_actions)

        self.state_memory = torch.zeros((self.mem_size,self.input_channes,self.w,self.h),dtype=torch.float)
        self.new_state_memory = torch.zeros((self.mem_size,self.input_channes,self.w,self.h),dtype=torch.float)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)

    def store_memory(self,state,state_,reward,action,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            observation  = state.unsqueeze(0).to(self.Q_eval.device)
            actions = self.Q_eval.forward(observation)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        mem = min(self.mem_cntr,self.mem_size)

        batch = np.random.choice(mem,self.batch_size,replace=False)
        batch_index = np.arange(self.batch_size,dtype=np.int32)

        state_batch = self.state_memory[batch].to(self.Q_eval.device)
        new_state_batch = self.new_state_memory[batch].to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next  = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min       


if __name__ == "__main__":
    env = gym.make("MountainCar-v0").unwrapped

    resize = T.Compose([T.ToPILImage(),T.Resize((100,100)),T.ToTensor()])
    
    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)

    def get_screen():
        screen = env.render(mode='rgb_array').transpose((2,0,1))
        _,screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)

        screen = screen[:, :, slice_range]

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        image = resize(screen)
        return image , screen_height, screen_width
    
    brain = Agent(lr=0.003,gamma=0.99,batch_size=32,w=100,h=100,input_channels=3,
    conv_dim1 = 16,conv_dim2 = 32,conv_dim3= 32,n_actions=env.action_space.n)
    
    score = 0
    scores = list()
    n_games = 500
    show_every = 20
    max_avg_score = 0

    for episode in range(n_games):
        
        if episode % show_every == 0 and episode > 0:
            avg_score = np.mean(scores[-show_every:])
            if avg_score > max_avg_score:
                max_avg_score = avg_score
                torch.save(brain.Q_eval.state_dict(),f"best_model_{episode}.pth")

            print("episode ",episode, "score ",score,"avg score ", avg_score)
        else:
            print("episode ",episode,"score ", score)

        env.reset()
        last_state, height, width = get_screen()
        current_state , _, _ = get_screen()
        state = current_state - last_state
        done = False
        score = 0
        while not done:
            action = brain.choose_action(state)
            _,reward,done,_  = env.step(action)
            score+= reward

            last_state = current_state
            current_state,_,_ = get_screen()

            next_state = current_state - last_state
            
            brain.store_memory(state,next_state,reward,action,done)

            state = next_state
            brain.learn()
        
        scores.append(score)
    
    env.close()
    
    plt.plot([x for x in range(n_games)],scores)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()

            









