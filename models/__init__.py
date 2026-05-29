import os
import torch

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def available_models():
    return [
        f.replace('.py', '')
        for f in os.listdir(_MODELS_DIR)
        if f.endswith('.py') and not f.startswith('_')
    ]


def build_model(model_name: str):
    if model_name == 'gpt_rot':
        from models.gpt_rot import GPT2Rotary
        return GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    elif model_name == 'gpt_abs':
        from models.gpt_abs import GPT2Absolute
        return GPT2Absolute(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    elif model_name == 'gpt_rel':
        from models.gpt_rel import GPT2Relative
        return GPT2Relative(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    elif model_name == 'gpt_pos':
        from models.gpt_pos import GPT2Position
        return GPT2Position(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    elif model_name == 'gpt_uni':
        from models.gpt_uni import GPT2Uniform
        return GPT2Uniform(vocab_size=5, n_layer=12, n_embd=1024)
    raise ValueError(f"Unknown model '{model_name}'. Available: {available_models()}")


def load_model_weights(model, model_weights_path: str, device: str = 'cpu'):
    state = torch.load(model_weights_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model
