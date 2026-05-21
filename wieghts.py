import torch

ckpt = torch.load('gpt_checkpoint_step_6500.pt', map_location='cpu')
torch.save(ckpt['model_state_dict'], 'gpt_weights_6500.pt')
