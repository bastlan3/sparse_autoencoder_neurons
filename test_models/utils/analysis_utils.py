"""
Advanced analysis utilities for sparse dictionary learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


def compute_feature_diversity_score(category_activations: Dict[str, torch.Tensor], 
                                  feature_idx: int) -> float:
    """
    Compute how diverse a feature's activations are across categories.
    Higher score means more diverse (less specialized).
    
    Args:
        category_activations: Dict mapping category names to activation tensors
        feature_idx: Index of the feature to analyze
        
    Returns:
        float: Diversity score (0 = perfectly specialized, 1 = perfectly uniform)
    """
    activations = []
    for category_tensor in category_activations.values():
        activations.append(category_tensor[feature_idx].item())
    
    activations = np.array(activations)
    
    # Normalize activations to sum to 1 (like a probability distribution)
    if np.sum(activations) > 0:
        activations = activations / np.sum(activations)
        
        # Calculate entropy (diversity measure)
        entropy = -np.sum(activations * np.log(activations + 1e-8))
        max_entropy = np.log(len(activations))  # Maximum possible entropy
        
        # Normalize to 0-1 range
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
    else:
        diversity_score = 0
    
    return diversity_score


def find_category_signature_features(category_activations: Dict[str, torch.Tensor],
                                   min_activation: float = 0.01,
                                   top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
    """
    Find signature features for each category (features that are most characteristic).
    
    Args:
        category_activations: Dict mapping category names to activation tensors
        min_activation: Minimum activation threshold
        top_k: Number of top features to return per category
        
    Returns:
        Dict mapping category names to lists of (feature_idx, activation_strength) tuples
    """
    signature_features = {}
    
    for category, activations in category_activations.items():
        # Get features that exceed minimum activation
        strong_features = [(i, val.item()) for i, val in enumerate(activations) 
                         if val.item() > min_activation]
        
        # Sort by activation strength
        strong_features.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        signature_features[category] = strong_features[:top_k]
    
    return signature_features


def create_feature_similarity_matrix(category_activations: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    Create a similarity matrix between categories based on their activation patterns.
    
    Args:
        category_activations: Dict mapping category names to activation tensors
        
    Returns:
        np.ndarray: Similarity matrix [n_categories, n_categories]
    """
    categories = list(category_activations.keys())
    n_categories = len(categories)
    
    # Stack all activation vectors
    activation_matrix = torch.stack([category_activations[cat] for cat in categories])
    
    # Compute cosine similarity between all pairs
    similarity_matrix = torch.nn.functional.cosine_similarity(
        activation_matrix.unsqueeze(1), 
        activation_matrix.unsqueeze(0), 
        dim=2
    )
    
    return similarity_matrix.numpy(), categories


def visualize_category_similarity(similarity_matrix: np.ndarray, 
                                categories: List[str], 
                                experiment_name: str):
    """
    Visualize similarity between categories based on activation patterns.
    
    Args:
        similarity_matrix: Category similarity matrix
        categories: List of category names
        experiment_name: Name for saving files
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=categories, 
                yticklabels=categories,
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True)
    
    plt.title(f'Category Similarity Matrix - {experiment_name}')
    plt.xlabel('Category')
    plt.ylabel('Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save plot
    os.makedirs('plots/category_analysis', exist_ok=True)
    filename = f'plots/category_analysis/{experiment_name}_category_similarity.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved category similarity plot: {filename}')


def generate_analysis_report(category_activations: Dict[str, torch.Tensor],
                           category_counts: Dict[str, int],
                           experiment_name: str) -> str:
    """
    Generate a comprehensive text report of the analysis.
    
    Args:
        category_activations: Dict mapping category names to activation tensors
        category_counts: Dict mapping category names to sample counts
        experiment_name: Name for the experiment
        
    Returns:
        str: Path to the saved report file
    """
    # Calculate various statistics
    signature_features = find_category_signature_features(category_activations)
    similarity_matrix, categories = create_feature_similarity_matrix(category_activations)
    
    # Calculate feature diversity scores
    n_features = len(next(iter(category_activations.values())))
    diversity_scores = []
    for feature_idx in range(n_features):
        diversity = compute_feature_diversity_score(category_activations, feature_idx)
        diversity_scores.append((feature_idx, diversity))
    
    # Sort by diversity (most specialized first)
    diversity_scores.sort(key=lambda x: x[1])
    
    # Generate report
    os.makedirs('plots/category_analysis', exist_ok=True)
    report_path = f'plots/category_analysis/{experiment_name}_detailed_report.txt'
    
    with open(report_path, 'w') as f:
        f.write(f"Detailed Analysis Report: {experiment_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Dataset summary
        f.write("Dataset Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total categories: {len(categories)}\n")
        f.write(f"Total features: {n_features}\n")
        f.write(f"Total samples analyzed: {sum(category_counts.values())}\n\n")
        
        # Per-category sample counts
        f.write("Samples per category:\n")
        for category in sorted(category_counts.keys()):
            f.write(f"  {category}: {category_counts[category]} samples\n")
        f.write("\n")
        
        # Most specialized features
        f.write("Most Specialized Features (low diversity):\n")
        f.write("-" * 40 + "\n")
        for i, (feature_idx, diversity) in enumerate(diversity_scores[:20]):
            # Find which category this feature is most active for
            max_category = max(category_activations.keys(), 
                             key=lambda cat: category_activations[cat][feature_idx].item())
            max_activation = category_activations[max_category][feature_idx].item()
            
            f.write(f"Feature {feature_idx:4d}: diversity={diversity:.3f}, "
                   f"strongest in '{max_category}' (activation={max_activation:.4f})\n")
        f.write("\n")
        
        # Most diverse features
        f.write("Most Diverse Features (high diversity):\n")
        f.write("-" * 40 + "\n")
        for i, (feature_idx, diversity) in enumerate(diversity_scores[-20:]):
            f.write(f"Feature {feature_idx:4d}: diversity={diversity:.3f}\n")
        f.write("\n")
        
        # Signature features per category
        f.write("Signature Features per Category:\n")
        f.write("-" * 35 + "\n")
        for category in sorted(signature_features.keys()):
            f.write(f"\n{category}:\n")
            for feature_idx, activation in signature_features[category]:
                f.write(f"  Feature {feature_idx:4d}: {activation:.4f}\n")
        f.write("\n")
        
        # Category similarity analysis
        f.write("Most Similar Category Pairs:\n")
        f.write("-" * 30 + "\n")
        similarity_pairs = []
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                similarity = similarity_matrix[i, j]
                similarity_pairs.append((categories[i], categories[j], similarity))
        
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        for cat1, cat2, sim in similarity_pairs[:10]:
            f.write(f"  {cat1} <-> {cat2}: {sim:.3f}\n")
    
    print(f'Saved detailed analysis report: {report_path}')
    return report_path
