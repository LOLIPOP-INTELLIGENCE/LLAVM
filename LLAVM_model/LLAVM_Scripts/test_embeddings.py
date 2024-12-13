import torch
import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

def compare_embeddings(text_embeddings, audio_embeddings, method='dtw'):
    """
    Compare two sequences of embeddings with different lengths.
    
    Parameters:
    text_embeddings: torch.Tensor of shape (B, L1, D) or (L1, D) where L1 is text sequence length
    audio_embeddings: torch.Tensor of shape (B, L2, D) or (L2, D) where L2 is audio sequence length
    method: str, one of ['dtw', 'ctc', 'average', 'window']
    
    Returns:
    float: similarity score between 0 and 1, where 1 is most similar
    dict: additional metrics and information
    """
    # Convert to numpy for processing and handle dimensions
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu()
        # Remove batch dimension if present
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.squeeze(0)
        text_embeddings = text_embeddings.numpy()
        
    if isinstance(audio_embeddings, torch.Tensor):
        audio_embeddings = audio_embeddings.detach().cpu()
        # Remove batch dimension if present
        if audio_embeddings.dim() == 3:
            audio_embeddings = audio_embeddings.squeeze(0)
        audio_embeddings = audio_embeddings.numpy()
    
    metrics = {}
    
    if method == 'dtw':
        # Dynamic Time Warping
        distance, path = fastdtw(text_embeddings, audio_embeddings, dist=cosine)
        metrics['dtw_distance'] = distance
        metrics['dtw_path'] = path
        similarity = 1 / (1 + distance)  # Convert distance to similarity score
        
    elif method == 'average':
        # Compare average embeddings
        text_avg = np.mean(text_embeddings, axis=0)
        audio_avg = np.mean(audio_embeddings, axis=0)
        similarity = 1 - cosine(text_avg, audio_avg)
        
    elif method == 'window':
        # Sliding window comparison
        window_size = min(text_embeddings.shape[0], audio_embeddings.shape[0])
        similarities = []
        
        for i in range(max(1, text_embeddings.shape[0] - window_size + 1)):
            text_window = text_embeddings[i:i+window_size]
            for j in range(max(1, audio_embeddings.shape[0] - window_size + 1)):
                audio_window = audio_embeddings[j:j+window_size]
                if text_window.shape[0] == audio_window.shape[0]:
                    sim = 1 - np.mean([cosine(t, a) for t, a in zip(text_window, audio_window)])
                    similarities.append(sim)
        
        similarity = max(similarities) if similarities else 0
        metrics['window_similarities'] = similarities
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    metrics['similarity_score'] = similarity
    return similarity, metrics

def analyze_embedding_alignment(text_embeddings, audio_embeddings):
    """
    Perform comprehensive analysis of embedding alignment between text and audio.
    
    Parameters:
    text_embeddings: torch.Tensor of shape (L1, D)
    audio_embeddings: torch.Tensor of shape (L2, D)
    
    Returns:
    dict: Dictionary containing various analysis metrics
    """
    analysis = {}
    
    # Compare using different methods
    for method in ['dtw', 'average', 'window']:
        similarity, metrics = compare_embeddings(text_embeddings, audio_embeddings, method=method)
        analysis[f'{method}_similarity'] = similarity
        analysis[f'{method}_metrics'] = metrics
    
    # Basic statistics
    analysis['text_length'] = len(text_embeddings)
    analysis['audio_length'] = len(audio_embeddings)
    analysis['length_ratio'] = len(text_embeddings) / len(audio_embeddings)
    
    # Embedding space statistics
    analysis['text_embedding_norm'] = np.linalg.norm(text_embeddings.cpu().detach(), axis=1).mean()
    analysis['audio_embedding_norm'] = np.linalg.norm(audio_embeddings.cpu().detach(), axis=1).mean()
    
    return analysis

# Example usage:
from Text2SemanticEmbedding import text_to_tokens_to_embeddings
from Audio2SemanticEmbedding import audio_to_semantic_embeddings

def compare_text_and_audio_embeddings(text, audio_path):
    # Get embeddings (assuming the previous functions are available)
    _, text_embeddings = text_to_tokens_to_embeddings(text)
    _, audio_embeddings = audio_to_semantic_embeddings(audio_path)
    
    # Perform analysis
    analysis = analyze_embedding_alignment(text_embeddings, audio_embeddings)
    
    return analysis

# Example usage:
if __name__ == "__main__":
    text = "Hello world"
    audio_path = "input.wav"
    
    analysis = compare_text_and_audio_embeddings(text, audio_path)
    import pdb; pdb.set_trace()