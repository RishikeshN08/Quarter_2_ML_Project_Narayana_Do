import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")


LEARNING_RATE = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
BATCH_SIZE = 1024 
MINIBATCH_SIZE = 64
EPOCHS = 10
TOTAL_UPDATES = 200     


class ActorCritic(nn.Module):
   def __init__(self, obs_dim, action_dim):
       super().__init__()
       self.shared = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
       self.actor = nn.Linear(128, action_dim)
       self.critic = nn.Linear(128, 1)
  
   def forward(self, x):
       shared = self.shared(x)
       logits = self.actor(shared)
       value = self.critic(shared).squeeze(-1)
       return logits, value




class PPO:
   def __init__(self, env):
       self.env = env
       obs_dim = env.observation_space.shape[0]
       action_dim = env.action_space.n
       self.network = ActorCritic(obs_dim, action_dim).to(device)
       self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
      
   def get_action(self, obs):
       obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
       with torch.no_grad():
           logits, value = self.network(obs)
           dist = torch.distributions.Categorical(logits=logits)
           action = dist.sample()
           log_prob = dist.log_prob(action)
       return action.item(), log_prob.item(), value.item()
  
   def compute_gae(self, rewards, values, dones, next_value):
       advantages = []
       gae = 0
       for t in reversed(range(len(rewards))):
           if t == len(rewards) - 1:
               next_val = next_value
               next_done = 0
           else:
               next_val = values[t + 1]
               next_done = dones[t + 1]
           delta = rewards[t] + GAMMA * next_val * (1 - next_done) - values[t]
           gae = delta + GAMMA * LAMBDA * (1 - next_done) * gae
           advantages.insert(0, gae) 
       advantages = torch.FloatTensor(advantages).to(device)
       returns = advantages + torch.FloatTensor(values).to(device)
       return returns, advantages
  
   def update(self, obs, actions, old_log_probs, returns, advantages, clipEpsilon):
       advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
       obs = torch.FloatTensor(obs).to(device)
       actions = torch.LongTensor(actions).to(device)
       old_log_probs = torch.FloatTensor(old_log_probs).to(device)
       for _ in range(EPOCHS):
           indices = np.random.permutation(len(obs))
           for start in range(0, len(obs), MINIBATCH_SIZE):
               idx = indices[start:start + MINIBATCH_SIZE]
               logits, values = self.network(obs[idx])
               dist = torch.distributions.Categorical(logits=logits)
               log_probs = dist.log_prob(actions[idx])
               entropy = dist.entropy().mean()
               ratio = torch.exp(log_probs - old_log_probs[idx])
               surr1 = ratio * advantages[idx]
               surr2 = torch.clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantages[idx]
               policy_loss = -torch.min(surr1, surr2).mean()
               value_loss = ((returns[idx] - values) ** 2).mean()
               loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
               self.optimizer.zero_grad()
               loss.backward()
               nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
               self.optimizer.step()


   def collect_batch(self, obs, batch_size):
       observations, actions, rewards, dones, log_probs, values, episode_rewards = [], [], [], [], [], [],[]
       episode_reward = 0
       for _ in range(batch_size):
           action, log_prob, value = self.get_action(obs)
           next_obs, reward, terminated, truncated, _ = self.env.step(action)
           done = terminated or truncated   
           observations.append(obs)
           actions.append(action)
           rewards.append(reward)
           dones.append(done)
           log_probs.append(log_prob)
           values.append(value)
           obs = next_obs
           episode_reward += reward
           if done:
               episode_rewards.append(episode_reward)
               episode_reward = 0
               obs, _ = self.env.reset()
       with torch.no_grad():
           _, next_value = self.network(torch.FloatTensor(obs).unsqueeze(0).to(device))
           next_value = next_value.item()
       return (np.array(observations), actions, rewards, dones, log_probs, values,
               next_value, obs, episode_rewards)
  
   def train(self):
       obs, _ = self.env.reset()
       all_episode_rewards = []
       for _ in range(1, TOTAL_UPDATES + 1):
           (observations, actions, rewards, dones, log_probs,
            values, next_value, obs, episode_rewards) = self.collect_batch(obs, BATCH_SIZE) 
           all_episode_rewards.extend(episode_rewards)
           returns, advantages = self.compute_gae(rewards, values, dones, next_value)
           clipEpsilon = 0.9/(advantages.std(unbiased=False) + 1e-8)
           clipEpsilon = torch.clamp(clipEpsilon, 0.05, 0.3)


           self.update(observations, actions, log_probs, returns, advantages, clipEpsilon) 
       return all_episode_rewards






def plot_results(episode_rewards):
   plt.figure(figsize=(10, 5))
   plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')


   if len(episode_rewards) >= 50:
       ma = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
       plt.plot(range(49, 49 + len(ma)), ma, label='Moving Average (50)')


   plt.xlabel('Episode')
   plt.ylabel('Reward')
   plt.title('Clip PPO Training on CartPole-v1')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()


   plt.savefig("clip_ppo_cartpole_training.png")
   plt.close()




if __name__ == "__main__":
   env = gym.make("CartPole-v1")
   agent = PPO(env)
   episode_rewards = agent.train()
   #reward_std = rolling_std(episode_rewards)
   env.close()
   plot_results(episode_rewards)
   #plot_batch_reward_std(episode_rewards)