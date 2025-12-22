
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

# Check description: "Deep value iteration methods align with the objective... 
# This includes Q-learning [161] and its deep-learning version—DQN [162]" (Lines 104, 122)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# PART 1: CONTINUOUS ENVIRONMENT
# ==========================================
# Making this "Practically Relevant" by using Continuous State Space (Position, Velocity)
# Goal: Stabilize a particle at Position = 0.
class ContinuousWorld:
    def __init__(self):
        # State: [Position, Velocity] (Continuous)
        self.state_size = 2
        # Action: 0 (Push Left), 1 (Do Nothing), 2 (Push Right) (Discrete)
        self.action_size = 3
        
        self.state = np.array([0.0, 0.0])
        self.max_steps = 200
        self.current_step = 0
        
    def reset(self):
        # Start at random position away from center
        pos = random.uniform(-2.0, 2.0)
        vel = 0.0
        self.state = np.array([pos, vel])
        self.current_step = 0
        return self.state

    def step(self, action):
        pos, vel = self.state
        
        # Physics Dynamics
        force = 0.0
        if action == 0: force = -1.0 # Left
        elif action == 2: force = 1.0 # Right
        
        vel += force * 0.1 # F=ma (dt=0.1)
        pos += vel * 0.1
        
        # Clip to keeping boundaries reasonable
        pos = np.clip(pos, -5.0, 5.0)
        vel = np.clip(vel, -2.0, 2.0)
        
        self.state = np.array([pos, vel])
        self.current_step += 1
        
        # Reward Function
        # We want: Distance to be 0
        reward = -np.abs(pos) - 0.1 * np.abs(vel) # Penalty for distance and speed (Energy min)
        
        # Done condition
        done = self.current_step >= self.max_steps
        
        return self.state, reward, done

# ==========================================
# PART 2: THE NETWORK (Q-Function)
# ==========================================
# Ref: "In policy gradient methods... theta can be the weights and biases of a deep neural network" (Lines 88-89)
# Here we use it for Value Function (DQN)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Simple feed-forward network
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size) # Outputs Q-value for each action
        
    def forward(self, x):
        # x is already a tensor on the correct device if passed correctly, 
        # but for safety in the policy method we usually cast it.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# ==========================================
# PART 3: REPLAY BUFFER
# ==========================================
# Essential for Deep RL to break correlations in training data
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# PART 4: DQN AGENT
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.lr = 0.001
        
        # Networks
        self.q_net = QNetwork(state_size, action_size).to(device)
            
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()

    def policy(self, state):
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device) # Add dim for gather
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Compute Q(s, a)
        # self.q_net(states) gives [Batch, ActionSize]
        # .gather(1, actions) selects the Q-value for the specific action taken
        current_q = self.q_net(states).gather(1, actions)
        
        # Compute Target: r + gamma * max Q(s')
        # We use .max(1)[0] to get max along action dim
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Loss (MSE)
        # "maximize the cumulative expected reward G" (Line 36) -> Minimize prediction error
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# PART 5: TRAINING LOOP
# ==========================================
def run_dqn():
    env = ContinuousWorld()
    agent = DQNAgent(env.state_size, env.action_size)
    
    num_episodes = 1000
    
    print("Starting DQN Training (PyTorch)...")
    print("Goal: Stabilize particle at 0.0.")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn from batch
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    print("\nTraining Complete.")
    
    # Test Run
    print("\nRunning Verification Episode (Greedy policy)...")
    state = env.reset()
    state[0] = 2.0 # Force start at 2.0
    state[1] = 0.0
    env.state = state # HACK: force env state
    
    print(f"Start State: {state}")
    for t in range(20):
        action = agent.policy(state) # Should likely be 0 (Push Left) if state is positive
        next_state, r, d = env.step(action)
        action_str = ["LEFT", "NONE", "RIGHT"][action]
        print(f"Time {t}: Pos={state[0]:.2f}, Vel={state[1]:.2f} -> Action: {action_str}")
        state = next_state
        if abs(state[0]) < 0.1 and abs(state[1]) < 0.1:
            print("Converged to Goal!")
            break

if __name__ == "__main__":
    run_dqn()
