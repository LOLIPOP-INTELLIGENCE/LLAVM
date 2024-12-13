from transformers import AutoModel, AutoConfig
from pprint import pprint

def inspect_embedding_size(model_name):
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
    
    # Method 1: Direct from config
    print(f"Embedding size from config: {config.hidden_size}")
    
    # Method 2: Inspect embedding layer
    for name, module in model.named_modules():
        if 'embedding' in name.lower():
            print(f"\nFound embedding layer: {name}")
            print(f"Module type: {type(module)}")
            print(f"Module parameters:")
            for param_name, param in module.named_parameters():
                print(f"- {param_name}: {param.shape}")

    # Method 3: Get embedding weight shape directly
    if hasattr(model, 'get_input_embeddings'):
        emb = model.get_input_embeddings()
        print(f"\nEmbedding weight shape: {emb.weight.shape}")
        # The second dimension is the embedding size
        print(f"Embedding size: {emb.weight.shape[1]}")
