import os
import numpy as np
import torch

def create_eval_func(model):
    """
    Create an eval_func compatible with reference.py
    
    Args:
        model: The neural network model with a forward_for_mcts method.
        device: The device to run the model on ('cpu' or 'cuda').
    
    Returns:
        eval_func(obs, batched) -> (prior, value)
    """
    
    board_len = int(np.prod(model.logic.BOARD_SHAPE))

    # keep model on cpu for game generation
    # import copy
    # cpu_model = copy.deepcopy(model).to('cpu').eval()
    # torch.set_num_threads(2)
    device = next(model.parameters()).device

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        # Fallback if model has no parameters
        model_dtype = torch.float32

    def eval_func(obs: np.ndarray, batched: bool):
        
        with torch.no_grad():
                
            if batched:
                # obs shape: (batch, board_len + 1)
                boards = torch.tensor(obs[:, :board_len], dtype=model_dtype).to(device)
                current_player = torch.tensor(obs[:, board_len].astype(np.int64), dtype=torch.long).to(device)
            else:
                # obs shape: (board_len + 1,)
                boards = torch.tensor(obs[:board_len], dtype=model_dtype).unsqueeze(0).to(device)
                current_player = torch.tensor([int(obs[board_len])], dtype=torch.long).to(device)
            
            batch = {
                'boards': boards,
                'current_player': current_player,
                'puzzle_identifiers': torch.zeros(boards.shape[0], dtype=torch.long).to(device)
            }
            
            outputs = model.forward_for_mcts(batch)

            priors = outputs['policy'].float().cpu().numpy()
            values = outputs['value'].float().cpu().numpy()
                
            if batched:
                return priors, [float(v) for v in values]  # (batch, num_actions), (batch,)
            else:
                return priors[0], float(values[0])  # (num_actions,), float
    
    return eval_func

def create_eval_func_safe(model, device):
    """Create eval function without deepcopy (for models that don't support it)."""
    def eval_func(obs, batched=False):
        model.eval()
        with torch.no_grad():
            if batched:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            batch_size = obs_tensor.shape[0]
            boards = obs_tensor[:, :42].reshape(batch_size, 6, 7)
            
            p1_count = (boards == 1).sum(dim=(1, 2))
            p2_count = (boards == 2).sum(dim=(1, 2))
            players = torch.where(p1_count <= p2_count, 
                                  torch.ones(batch_size, device=device),
                                  torch.full((batch_size,), 2, device=device)).long()
            
            batch = {
                "boards": boards.flatten(1).float(),
                "current_player": players,
                "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
            }
            
            outputs = model.forward_for_mcts(batch)
            
            priors = outputs['policy'].cpu().numpy()
            values = outputs['value'].cpu().numpy().flatten()
            
            if batched:
                return priors, [float(v) for v in values]
            else:
                return priors[0], float(values[0])
    
    return eval_func

import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.policy_head = nn.Linear(1, logic.NUM_ACTIONS) 
        self.value_head = nn.Linear(1, 1)
    
    def forward_for_mcts(self, batch):
        b = batch['boards'].shape[0]
        # Fast random outputs
        return {
            "policy": torch.softmax(torch.randn(b, self.logic.NUM_ACTIONS, device=batch['boards'].device), dim=-1),
            "value": torch.tanh(torch.randn(b, 1, device=batch['boards'].device)).squeeze(-1)
        }
    def eval(self): pass
    def train(self): pass
        
def get_device():
    """Utility to get the device for benchmarking."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
