
import numpy as np
import random
import time

# ==========================================
# PART 1: THE ENVIRONMENT
# ==========================================
# Ref: "In RL, the environment is initialized in a certain state s0...
# the environment refers to the physical system the agent interacts with" (Lines 14-16, 26)
class SimpleGridWorld:
    """
    A simple 1D Grid World environment.
    State space: 0, 1, 2, 3, 4 (Goal)
    Action space: 0 (Left), 1 (Right)
    """
    def __init__(self):
        self.state_space_size = 5
        self.action_space_size = 2 # Left, Right
        self.state = 0
        self.goal_state = 4
        
    def reset(self):
        """
        Ref: "Once the episode comes to an end, the environment is reset to its initial state s0" (Line 44-45)
        """
        self.state = 0  # Initialize in state s0 = 0
        return self.state

    def step(self, action):
        """
        Ref: "Depending on the action a_t selected, the environment reacts and changes state: s_t -> s_{t+1}...
        and is given a reward r_{t+1}" (Lines 28-31)
        """
        # Dynamics: s_t -> s_{t+1}
        if action == 1: # Right
            next_state = min(self.state + 1, self.state_space_size - 1)
        else: # Left
            next_state = max(self.state - 1, 0)
        
        # Reward r_{t+1}
        # "quantifies the quality of the action taken with respect to the objective" (Line 31)
        # Goal: reach state 4.
        if next_state == self.goal_state:
            reward = 10.0
            done = True
        else:
            reward = -1.0 # Small penalty to encourage reaching goal quickly
            done = False
            
        self.state = next_state
        return next_state, reward, done

# ==========================================
# PART 2: THE AGENT
# ==========================================
# Ref: "The main objective is to learn a strategy (a.k.a. policy)... 
# to select the next action" (Lines 18-19, 34)
class QLearningAgent:
    """
    Implements a Value-function method (Q-learning).
    Ref: "Value-function methods allow us to encode information about the policy...
    Q-learning [161]" (Lines 100-104)
    """
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.9, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate # Learning rate (how much we accept new information)
        self.gamma = gamma      # Discount factor (importance of future rewards)
        self.epsilon = epsilon  # Exploration probability
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Q-Function: Q(s, a)
        # Ref: "concept of action-value (or Q-) functions Q(s, a); they assign an expected return value to each state-action pair" (Lines 77-78)
        # Initialized to zeros (arbitrary initial guess)
        self.q_table = np.zeros((state_size, action_size))

    def policy(self, state):
        """
        Ref: "To explore new actions... the policy in RL is typically non-deterministic" (Lines 58-60)
        This implements Epsilon-Greedy policy:
        - Explore: choose random action (uniform distribution)
        - Exploit: choose action with max Q-value
        Ref: "exploration-exploitation dilemma" (Line 69)
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose random action from A
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: argmax Q(s, a)
            # Ref: "extract a policy... taking the action that maximizes the action-value function" (Lines 80-81)
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-function based on experience.
        Ref: "algorithms iteratively improve the policy... optimize Q-function itself" (Lines 70, 102)
        """
        # Q-learning update rule (Bellman equation)
        # Q(s,a) = Q(s,a) + lr * [r + gamma * max Q(s', a') - Q(s,a)]
        
        # Estimate of optimal future value
        best_future_q = np.max(self.q_table[next_state])
        
        # Current Q value
        current_q = self.q_table[state, action]
        
        # Update
        new_q = current_q + self.lr * (reward + self.gamma * best_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        # Reduce exploration over time as we learn
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# PART 3: INTERACTION LOOP (TRAINING)
# ==========================================
def run_training():
    # Setup
    env = SimpleGridWorld()
    agent = QLearningAgent(env.state_space_size, env.action_space_size)
    
    # "The learning process in RL is often divided into episodes" (Line 42-43)
    num_episodes = 50
    
    print(f"Starting Training for {num_episodes} Episodes...")
    print(f"Goal: Move from State 0 to State 4 (Right is action 1)")
    
    for episode in range(num_episodes):
        # "environment is reset to its initial state s0" (Line 45)
        state = env.reset()
        total_reward = 0
        done = False
        
        # Steps within an episode
        while not done:
            # 1. Agent observes state s_t and chooses action a_t
            # "Based on observations... select the next action" (Lines 19, 34)
            action = agent.policy(state)
            
            # 2. Environment reacts
            # "Reacts and changes state s_t -> s_{t+1}... given reward r_{t+1}" (Lines 29-31)
            next_state, reward, done = env.step(action)
            
            # 3. Agent learns
            # "The agent keeps building on the experience gained" (Line 46)
            agent.learn(state, action, reward, next_state)
            
            # Update current state to next state for next step
            state = next_state
            total_reward += reward
            
        # End of episode
        agent.decay_epsilon()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    print("\nTraining Complete.")
    
    # ==========================================
    # PART 4: INSPECT THE RESULTS
    # ==========================================
    print("\nLearned Q-Table (Action-Values):")
    print("State | Left (0) | Right (1)")
    print("----------------------------")
    for s in range(env.state_space_size):
        qs = agent.q_table[s]
        best_a = "RIGHT" if np.argmax(qs) == 1 else "LEFT"
        print(f"  {s}   |  {qs[0]:.2f}   |  {qs[1]:.2f}  -> Best Action: {best_a}")

    print("\nRef: 'Optimal policies, by definition, maximize the total expected return G' (Line 71-72)")
    print("Notice how the Q-values are highest for 'Right' actions that lead to the goal (State 4).")

if __name__ == "__main__":
    run_training()
