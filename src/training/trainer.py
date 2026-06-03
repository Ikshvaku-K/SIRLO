
import torch
import torch.optim as optim
import numpy as np
from src.training.marl_env import MARLEnvironment
from src.agents.vit_agent import ViTAgent

class MARLTrainer:
    def __init__(self, num_satellites=5, learning_rate=1e-3, gamma=0.99):
        """
        Initialize the MARL Trainer.
        """
        self.env = MARLEnvironment(num_satellites=num_satellites)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Shared Agent Policy
        self.agent = ViTAgent(input_resolution=64, patch_size=8, in_channels=2, num_actions=3).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.gamma = gamma

    def train_episode(self):
        """
        Run a single training episode.
        """
        state = self.env.reset()
        world_state = torch.tensor(state['world_state']).unsqueeze(0).to(self.device) # (1, C, H, W)
        satellite_states = torch.tensor(state['satellite_states']).to(self.device) # (N, SatDim)
        
        log_probs = []
        rewards = []
        
        # Episode length
        for _ in range(100): # 100 steps per episode
            # Select actions for all satellites
            # Since we have a shared policy but the world state is same for all (for now)
            # We can process each satellite with the same world state context
            
            # Batch World State: Repeat for N satellites
            batch_world = world_state.repeat(self.env.num_satellites, 1, 1, 1)
            
            # Forward pass
            logits = self.agent(batch_world, satellite_states)
            probs = torch.softmax(logits, dim=-1)
            
            # Sample actions
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            log_probs.append(dist.log_prob(actions))
            
            # Step Environment
            action_list = actions.cpu().numpy().tolist()
            next_state_dict, step_rewards, dones, _ = self.env.step(action_list)
            
            rewards.append(step_rewards)
            
            # Prepare next state
            satellite_states = torch.tensor(next_state_dict['satellite_states']).to(self.device)
            # World state update handled by repeat in next loop, or could update here
            if next_state_dict['world_state'] is not None:
                 world_state = torch.tensor(next_state_dict['world_state']).unsqueeze(0).to(self.device)

            if all(dones):
                break
                
        # Calculate returns and update policy
        self.update_policy(log_probs, rewards)
        
        return sum([sum(r) for r in rewards])

    def update_policy(self, log_probs, rewards):
        """
        Update the policy using REINFORCE.
        """
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns separately for each satellite?
        # Or average across satellites? 
        # For simplicity, let's treat each satellite's experience as a separate trajectory for the shared policy
        
        # rewards shape: (Steps, NumSatellites)
        # log_probs shape: (Steps, NumSatellites)
        
        num_steps = len(rewards)
        num_sats = len(rewards[0])
        
        for s in range(num_sats):
            R = 0
            sat_returns = []
            for t in range(num_steps - 1, -1, -1):
                R = rewards[t][s] + self.gamma * R
                sat_returns.insert(0, R)
                
            sat_returns = torch.tensor(sat_returns).to(self.device)
            sat_returns = (sat_returns - sat_returns.mean()) / (sat_returns.std() + 1e-9)
            
            for t in range(num_steps):
                policy_loss.append(-log_probs[t][s] * sat_returns[t])
                
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    trainer = MARLTrainer()
    print("Starting training...")
    for i in range(10):
        reward = trainer.train_episode()
        print(f"Episode {i}: Total Reward: {reward:.2f}")
