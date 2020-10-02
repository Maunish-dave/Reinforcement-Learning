import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount = 0.95
episodes = 25000
show_every = 3000

epsilon = 1
start_epislon_decaying = 1
end_epislon_decaying = episodes//2
epislon_decay_value = epsilon / (end_epislon_decaying - start_epislon_decaying)

discrete_os_size = [20] * len(env.observation_space.high)
discrete_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size
q_table = np.random.uniform(low=-2,high=0,size=discrete_os_size + [env.action_space.n])

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(episodes):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False
    render = True if episode % show_every == 0 else False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)

        new_state,reward ,done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        episode_reward+= reward
        if render:
            env.render()

        if not done:
            current_q = q_table[discrete_state + (action,)]
            max_q = np.max(q_table[new_discrete_state + (action,)])  * discount
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_q)
            q_table[discrete_state+ (action,)] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = new_discrete_state
    
    if end_epislon_decaying >= episode >= start_epislon_decaying:
        epsilon -= epislon_decay_value
    
    ep_rewards.append(episode_reward)
    if not episode % show_every:
        average_reward = sum(ep_rewards[-show_every:])/show_every
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))


env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()