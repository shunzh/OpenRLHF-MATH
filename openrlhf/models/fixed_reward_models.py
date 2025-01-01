import torch

# Dictionary to store registered reward models
REWARD_MODELS = {}

def register_reward_model(name):
    """
    Decorator to register a reward model function.
    
    Args:
        name (str): The name of the reward model to register
        
    Returns:
        callable: The decorator function
    """
    def decorator(func):
        REWARD_MODELS[name] = func
        return func
    return decorator

def get_reward_model(name):
    """
    Get a registered reward model by name.
    
    Args:
        name (str): The name of the reward model to retrieve
        
    Returns:
        callable: The registered reward model function
        
    Raises:
        KeyError: If the reward model name is not registered
    """
    if name not in REWARD_MODELS:
        raise KeyError(f"Reward model '{name}' not found. Available models: {list(REWARD_MODELS.keys())}")
    return REWARD_MODELS[name]

@register_reward_model("MATH")
def math_reward_model(queries, answers):
    """
    Get the content inside \boxed{} in queries, and see if they match the answers.
    """
    rewards = []
    for query, answer in zip(queries, answers):
        query_content = query.split("\\boxed{")[1].split("}")[0]
        # TODO Handle floating point numbers
        if query_content == answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)
