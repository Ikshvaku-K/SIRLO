
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTAgent(nn.Module):
    def __init__(self, input_resolution=64, patch_size=8, in_channels=2, num_actions=3, embed_dim=128, depth=4, num_heads=4, satellite_state_dim=2):
        """
        Vision Transformer Agent.
        
        Args:
            input_resolution: Height/Width of the input world state tensor.
            patch_size: Size of patches.
            in_channels: Number of input channels (Cloud, Fire).
            num_actions: Number of output actions.
            embed_dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            satellite_state_dim: Dimension of the satellite internal state vector.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_resolution // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Encounter
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Satellite State Embedding
        self.sat_state_embed = nn.Linear(satellite_state_dim, embed_dim)
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # Concatenate CLS token + Sat State
            nn.ReLU(),
            nn.Linear(embed_dim, num_actions)
        )
        
    def forward(self, world_state, satellite_state):
        """
        Forward pass.
        
        Args:
            world_state: Tensor of shape (Batch, Channels, Height, Width)
            satellite_state: Tensor of shape (Batch, SatelliteStateDim)
            
        Returns:
            Logits for actions (Batch, NumActions)
        """
        B = world_state.shape[0]
        
        # Patch Embeddings
        x = self.projection(world_state) # (B, EmbedDim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, NumPatches, EmbedDim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Extract CLS token output
        cls_output = x[:, 0]
        
        # Embed satellite state
        sat_embed = self.sat_state_embed(satellite_state)
        
        # Combine
        combined = torch.cat((cls_output, sat_embed), dim=1)
        
        # Action logits
        logits = self.action_head(combined)
        
        return logits

if __name__ == "__main__":
    # Test model
    model = ViTAgent(input_resolution=64, patch_size=8, in_channels=2, num_actions=3)
    
    # Dummy inputs
    dummy_world = torch.randn(1, 2, 64, 64)
    dummy_sat = torch.randn(1, 2)
    
    output = model(dummy_world, dummy_sat)
    print(f"Output shape: {output.shape}")
    probs = F.softmax(output, dim=1)
    print(f"Probabilities: {probs}")
