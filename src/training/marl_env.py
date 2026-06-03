
import numpy as np
import datetime
from typing import List, Dict, Tuple
import torch

from src.env.world_state import WorldStateGenerator
from src.env.propagator import OrbitalPropagator
from src.env.satellite import Satellite

class MARLEnvironment:
    def __init__(self, num_satellites=10, start_time=None):
        """
        Initialize the MARL Environment.
        
        Args:
            num_satellites: Number of satellites in the constellation.
            start_time: Start time of the simulation.
        """
        self.num_satellites = num_satellites
        self.current_time = start_time if start_time else datetime.datetime.now()
        self.time_step_seconds = 60
        
        # Initialize components
        # (Assuming dummy region for now)
        import ee
        # self.world_gen = WorldStateGenerator(region=ee.Geometry.Rectangle([-60.0, -5.0, -50.0, 5.0]))
        # Use a mock generator for now if GEE authentication is missing context
        self.world_gen = None 
        
        # Initialize Satellites
        self.satellites = [Satellite(i) for i in range(num_satellites)]
        
        # Initialize Propagators (with dummy TLEs for different orbits)
        self.propagators = []
        for i in range(num_satellites):
            # Generate different TLEs (very roughly)
            mean_motion_adjust = 0.0001 * i
            line1 = "1 25544U 98067A   21155.51864988  .00001609  00000-0  37475-4 0  9997"
            line2 = f"2 25544  51.6435 {210.4367+i*10:.4f} 0003058  45.0934  80.6015 {15.48967914+mean_motion_adjust:.8f}"
            self.propagators.append(OrbitalPropagator(line1, line2, f"Sat-{i}"))

        self.state_tensor = None

    def reset(self):
        """
        Reset the environment.
        """
        self.current_time = datetime.datetime.now()
        # Fetch initial world state
        # In a real scenario, this would call self.world_gen.generate_state_tensor
        # Mocking for speed/stability without GEE auth
        self.state_tensor = np.zeros((2, 64, 64), dtype=np.float32) 
        
        satellite_states = [sat.get_state() for sat in self.satellites]
        
        return {
            'world_state': self.state_tensor,
            'satellite_states': satellite_states
        }

    def step(self, actions: List[int]):
        """
        Step the environment.
        
        Args:
           actions: List of actions for each satellite.
           
        Returns:
           next_state, rewards, dones, info
        """
        self.current_time += datetime.timedelta(seconds=self.time_step_seconds)
        
        # 1. Update World State (Mocked)
        # self.state_tensor = self.world_gen.generate_state_tensor(self.current_time)
        # Randomly update some pixels to simulate clouds/fires moving
        if np.random.rand() > 0.5:
             self.state_tensor = np.random.rand(2, 64, 64).astype(np.float32)

        total_rewards = []
        infos = []
        
        # 2. Update Satellites
        for i, sat in enumerate(self.satellites):
            # Check solar visibility for charging
            # Simplified: Assuming cyclic daylight
            is_sunlit = (self.current_time.hour > 6 and self.current_time.hour < 18)
            
            # Check target visibility for imaging
            # (Need a target manager, for now assume random targets)
            target_visible = True
            
            # Override action if target not visible (can't image)
            actual_action = actions[i]
            if actual_action == 1 and not target_visible:
                actual_action = 0 # Idle instead
                
            reward, info = sat.step(actual_action, is_sunlit)
            
            # Additional global rewards/penalties
            # Cooperation bonus? 
            
            total_rewards.append(reward)
            infos.append(info)
            
        next_states = {
            'world_state': self.state_tensor,
            'satellite_states': [sat.get_state() for sat in self.satellites]
        }
        
        dones = [info.get('status') == 'dead' for info in infos]
        
        return next_states, total_rewards, dones, infos

if __name__ == "__main__":
    env = MARLEnvironment(num_satellites=5)
    state = env.reset()
    print("Environment reset.")
    
    actions = [1] * 5 # All image
    next_state, rewards, dones, infos = env.step(actions)
    
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
