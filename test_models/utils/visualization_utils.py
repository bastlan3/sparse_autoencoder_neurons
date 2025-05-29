"""
Visualization utilities for sparse dictionary learning.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def visualize_dictionary_features(dictionary, epoch, experiment_name, n_elements=16):
    """
    Visualize dictionary features as 2D images.
    
    Args:
        dictionary (SparseDictionary): The sparse dictionary model
        epoch (int): Current epoch number
        experiment_name (str): Name of the experiment
        n_elements (int): Number of dictionary elements to visualize
    """
    # Get dictionary weights [input_dim, dict_size]
    weights = dictionary.decoder.weight.detach().cpu()
    input_dim, dict_size = weights.shape
    
    # Determine how many elements to actually plot
    n_to_plot = min(n_elements, dict_size)
    
    # For visualization, we need to reshape to square images
    # Assume input_dim can be reshaped to a square (e.g., 64 -> 8x8, 256 -> 16x16)
    img_size = int(math.sqrt(input_dim))
    if img_size * img_size != input_dim:
        # If not perfect square, pad or crop
        img_size = int(math.ceil(math.sqrt(input_dim)))
        
    # Select top elements by L2 norm
    norms = torch.norm(weights, dim=0)
    _, top_indices = torch.topk(norms, n_to_plot)
    
    # Create subplot grid
    plot_cols = int(math.ceil(math.sqrt(n_to_plot)))
    plot_rows = int(math.ceil(n_to_plot / plot_cols))
    
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(15, 10))
    if plot_rows == 1 and plot_cols == 1:
        axes = [axes]
    elif plot_rows == 1 or plot_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot dictionary elements
    for i, idx in enumerate(top_indices):
        feature = weights[:, idx].numpy()
        
        # Reshape to square image (pad if necessary)
        if len(feature) < img_size * img_size:
            padded = np.zeros(img_size * img_size)
            padded[:len(feature)] = feature
            feature = padded
        else:
            feature = feature[:img_size * img_size]
            
        feature_img = feature.reshape(img_size, img_size)
        
        # Normalize for display
        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min() + 1e-8)
        
        axes[i].imshow(feature_img, cmap='RdBu_r', interpolation='nearest')
        axes[i].set_title(f'Dict {idx.item()}\nNorm: {norms[idx]:.3f}')
        axes[i].axis('off')
    
    # Remove unused subplots
    for i in range(n_to_plot, len(axes)):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle(f'Dictionary Features - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plot_filename = f'plots/{experiment_name}_dict_epoch_{epoch}.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    
    print(f'Dictionary visualization saved: {plot_filename}')


def analyze_activations_per_category(sparse_dict, resnet_extractor, dataloader, device, 
                                   use_subset=False, max_samples_per_category=100):
    """
    Analyze which dictionary features are activated for each category.
    
    Args:
        sparse_dict: Trained sparse dictionary model
        resnet_extractor: ResNet feature extractor
        dataloader: DataLoader with category information
        device: Computing device
        use_subset: Whether to use feature subset
        max_samples_per_category: Maximum samples to analyze per category
        
    Returns:
        dict: Category name -> activation statistics
    """
    sparse_dict.eval()
    category_activations = {}
    category_counts = {}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Extract features
            features = resnet_extractor(images)
            if use_subset:
                indices = torch.randperm(features.shape[1])[:64]
                features = features[:, indices]
            
            # Get activations
            activations = sparse_dict.encode(features)  # [batch_size, dict_size]
            
            # Process each sample in the batch
            for i, label in enumerate(labels):
                # Get category name from dataset
                if hasattr(dataloader.dataset, 'classes'):
                    category = dataloader.dataset.classes[label.item()]
                else:
                    category = f"class_{label.item()}"
                
                # Initialize category if not seen
                if category not in category_activations:
                    category_activations[category] = torch.zeros(activations.shape[1])
                    category_counts[category] = 0
                
                # Stop if we have enough samples for this category
                if category_counts[category] >= max_samples_per_category:
                    continue
                
                # Accumulate activations for this category
                sample_activations = activations[i].cpu()
                category_activations[category] += sample_activations
                category_counts[category] += 1
    
    # Average activations per category
    for category in category_activations:
        if category_counts[category] > 0:
            category_activations[category] /= category_counts[category]
    
    return category_activations, category_counts


def visualize_category_activations(category_activations, category_counts, experiment_name, 
                                 top_k=20, save_individual=True):
    """
    Visualize which dictionary features are most active for each category.
    
    Args:
        category_activations: Dict of category -> average activations
        category_counts: Dict of category -> sample count
        experiment_name: Name for saving files
        top_k: Number of top features to show per category
        save_individual: Whether to save individual category plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs('plots/category_analysis', exist_ok=True)
    
    # Create summary plot showing top features per category
    n_categories = len(category_activations)
    fig, axes = plt.subplots(n_categories, 1, figsize=(15, 3*n_categories))
    if n_categories == 1:
        axes = [axes]
    
    category_names = list(category_activations.keys())
    
    for idx, category in enumerate(category_names):
        activations = category_activations[category]
        count = category_counts[category]
        
        # Get top-k most active features
        top_values, top_indices = torch.topk(activations, min(top_k, len(activations)))
        
        # Plot bar chart
        x_pos = np.arange(len(top_values))
        axes[idx].bar(x_pos, top_values.numpy(), alpha=0.7)
        axes[idx].set_title(f'{category} (n={count}) - Top {len(top_values)} Active Features')
        axes[idx].set_xlabel('Dictionary Feature Index')
        axes[idx].set_ylabel('Average Activation')
        
        # Add feature indices as x-tick labels
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels([str(idx.item()) for idx in top_indices], rotation=45)
        
        # Save individual category plot if requested
        if save_individual:
            fig_individual, ax_individual = plt.subplots(1, 1, figsize=(12, 6))
            
            # Plot all activations with top-k highlighted
            all_activations = activations.numpy()
            ax_individual.bar(range(len(all_activations)), all_activations, alpha=0.3, color='gray')
            
            # Highlight top-k features
            for i, (val, idx) in enumerate(zip(top_values, top_indices)):
                ax_individual.bar(idx.item(), val.item(), alpha=0.8, color='red')
                ax_individual.text(idx.item(), val.item(), f'{idx.item()}', 
                                 ha='center', va='bottom', fontsize=8)
            
            ax_individual.set_title(f'{category} - All Dictionary Feature Activations (n={count})')
            ax_individual.set_xlabel('Dictionary Feature Index')
            ax_individual.set_ylabel('Average Activation')
            
            # Save individual plot
            individual_filename = f'plots/category_analysis/{experiment_name}_{category}_activations.png'
            plt.savefig(individual_filename, dpi=150, bbox_inches='tight')
            plt.close(fig_individual)
            print(f'Saved individual category plot: {individual_filename}')
    
    # Save summary plot
    plt.tight_layout()
    summary_filename = f'plots/category_analysis/{experiment_name}_category_summary.png'
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved category summary plot: {summary_filename}')


def create_activation_heatmap(category_activations, experiment_name, threshold=0.01):
    """
    Create a heatmap showing which features are active for which categories.
    
    Args:
        category_activations: Dict of category -> activations
        experiment_name: Name for saving
        threshold: Minimum activation value to consider "active"
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to matrix format
    categories = list(category_activations.keys())
    n_features = len(next(iter(category_activations.values())))
    
    # Create activation matrix [categories, features]
    activation_matrix = np.zeros((len(categories), n_features))
    for i, category in enumerate(categories):
        activation_matrix[i] = category_activations[category].numpy()
    
    # Create binary activation matrix (above threshold)
    binary_matrix = (activation_matrix > threshold).astype(float)
    
    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Raw activations heatmap
    im1 = ax1.imshow(activation_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Category Activation Heatmap - Raw Values')
    ax1.set_xlabel('Dictionary Feature Index')
    ax1.set_ylabel('Category')
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories)
    plt.colorbar(im1, ax=ax1, label='Average Activation')
    
    # Binary activations heatmap
    im2 = ax2.imshow(binary_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    ax2.set_title(f'Category Activation Heatmap - Binary (threshold={threshold})')
    ax2.set_xlabel('Dictionary Feature Index')
    ax2.set_ylabel('Category')
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories)
    plt.colorbar(im2, ax=ax2, label='Active (>threshold)')
    
    plt.tight_layout()
    
    # Save heatmap
    os.makedirs('plots/category_analysis', exist_ok=True)
    heatmap_filename = f'plots/category_analysis/{experiment_name}_activation_heatmap.png'
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved activation heatmap: {heatmap_filename}')
    
    return activation_matrix, binary_matrix


def analyze_feature_specialization(category_activations, experiment_name, min_activation=0.01):
    """
    Analyze which features are specialized for specific categories.
    
    Args:
        category_activations: Dict of category -> activations
        experiment_name: Name for saving
        min_activation: Minimum activation to consider
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    categories = list(category_activations.keys())
    n_features = len(next(iter(category_activations.values())))
    
    # Create activation matrix
    activation_matrix = np.zeros((len(categories), n_features))
    for i, category in enumerate(categories):
        activation_matrix[i] = category_activations[category].numpy()
    
    # Find specialized features (high activation for one category, low for others)
    specialization_scores = []
    feature_assignments = []
    
    for feature_idx in range(n_features):
        feature_activations = activation_matrix[:, feature_idx]
        
        if np.max(feature_activations) > min_activation:
            # Calculate specialization score (max - mean of others)
            max_idx = np.argmax(feature_activations)
            max_activation = feature_activations[max_idx]
            other_activations = np.concatenate([feature_activations[:max_idx], 
                                              feature_activations[max_idx+1:]])
            
            if len(other_activations) > 0:
                specialization = max_activation - np.mean(other_activations)
                specialization_scores.append(specialization)
                feature_assignments.append((feature_idx, categories[max_idx], max_activation))
            else:
                specialization_scores.append(max_activation)
                feature_assignments.append((feature_idx, categories[max_idx], max_activation))
        else:
            specialization_scores.append(0)
            feature_assignments.append((feature_idx, 'none', 0))
    
    # Sort by specialization score
    sorted_features = sorted(zip(specialization_scores, feature_assignments), reverse=True)
    
    # Create specialization report
    os.makedirs('plots/category_analysis', exist_ok=True)
    report_filename = f'plots/category_analysis/{experiment_name}_specialization_report.txt'
    
    with open(report_filename, 'w') as f:
        f.write(f"Feature Specialization Analysis for {experiment_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Top 50 Most Specialized Features:\n")
        f.write("-"*40 + "\n")
        
        for i, (spec_score, (feature_idx, assigned_category, max_activation)) in enumerate(sorted_features[:50]):
            f.write(f"Feature {feature_idx:4d}: {assigned_category:15s} "
                   f"(activation: {max_activation:.4f}, specialization: {spec_score:.4f})\n")
        
        # Category-wise summary
        f.write(f"\n\nCategory-wise Feature Count:\n")
        f.write("-"*30 + "\n")
        
        category_feature_counts = {}
        for _, (_, assigned_category, _) in sorted_features:
            if assigned_category != 'none':
                category_feature_counts[assigned_category] = category_feature_counts.get(assigned_category, 0) + 1
        
        for category, count in sorted(category_feature_counts.items()):
            f.write(f"{category:20s}: {count:3d} specialized features\n")
    
    print(f'Saved specialization report: {report_filename}')
    
    return sorted_features, category_feature_counts


def analyze_feature_activation_distributions(sparse_dict, resnet_extractor, dataloader, device,
                                           experiment_name, use_subset=False, n_features_to_analyze=50):
    """
    Analyze activation distributions for features, similar to Anthropic's monosemanticity analysis.
    Shows sparsity patterns and activation histograms.
    
    Args:
        sparse_dict: Trained sparse dictionary
        resnet_extractor: ResNet feature extractor
        dataloader: DataLoader for analysis
        device: Computing device
        experiment_name: Name for saving
        use_subset: Whether to use feature subset
        n_features_to_analyze: Number of top features to analyze in detail
    """
    sparse_dict.eval()
    all_activations = []
    
    print("Collecting activations for distribution analysis...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx > 20:  # Limit to prevent memory issues
                break
                
            images = images.to(device)
            features = resnet_extractor(images)
            
            if use_subset:
                indices = torch.randperm(features.shape[1])[:64]
                features = features[:, indices]
            
            activations = sparse_dict.encode(features)
            all_activations.append(activations.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)  # [total_samples, dict_size]
    
    print(f"Analyzing activation distributions for {all_activations.shape[0]} samples...")
    
    # Calculate feature statistics
    feature_means = all_activations.mean(dim=0)
    feature_stds = all_activations.std(dim=0)
    feature_sparsity = (all_activations > 0).float().mean(dim=0)  # Fraction of non-zero activations
    feature_max = all_activations.max(dim=0)[0]
    
    # Select top features by various criteria
    top_by_mean = torch.topk(feature_means, min(n_features_to_analyze, len(feature_means)))[1]
    top_by_sparsity = torch.topk(feature_sparsity, min(n_features_to_analyze, len(feature_sparsity)))[1]
    top_by_max = torch.topk(feature_max, min(n_features_to_analyze, len(feature_max)))[1]
    
    # Create comprehensive activation distribution plots
    os.makedirs('plots/feature_analysis', exist_ok=True)
    
    # 1. Overall sparsity distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(feature_sparsity.numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Activation Frequency (fraction of samples)')
    plt.ylabel('Number of Features')
    plt.title('Feature Sparsity Distribution')
    plt.axvline(feature_sparsity.mean(), color='red', linestyle='--', label=f'Mean: {feature_sparsity.mean():.3f}')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(feature_means.numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Activation')
    plt.ylabel('Number of Features')
    plt.title('Feature Mean Activation Distribution')
    plt.axvline(feature_means.mean(), color='red', linestyle='--', label=f'Mean: {feature_means.mean():.3f}')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.scatter(feature_sparsity.numpy(), feature_means.numpy(), alpha=0.6, s=10)
    plt.xlabel('Activation Frequency')
    plt.ylabel('Mean Activation')
    plt.title('Sparsity vs Mean Activation')
    
    plt.subplot(2, 2, 4)
    plt.hist(feature_max.numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Maximum Activation')
    plt.ylabel('Number of Features')
    plt.title('Feature Maximum Activation Distribution')
    
    plt.tight_layout()
    plt.savefig(f'plots/feature_analysis/{experiment_name}_activation_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed analysis of top features
    _plot_detailed_feature_analysis(all_activations, top_by_mean, top_by_sparsity, 
                                   experiment_name, 'top_features')
    
    print(f"Feature activation distribution analysis saved to plots/feature_analysis/")
    
    return {
        'all_activations': all_activations,
        'feature_means': feature_means,
        'feature_sparsity': feature_sparsity,
        'feature_max': feature_max,
        'top_by_mean': top_by_mean,
        'top_by_sparsity': top_by_sparsity
    }


def _plot_detailed_feature_analysis(all_activations, top_indices, alt_indices, experiment_name, suffix):
    """Helper function to plot detailed feature analysis."""
    n_features = min(16, len(top_indices))
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(n_features):
        feature_idx = top_indices[i]
        activations = all_activations[:, feature_idx].numpy()
        
        # Plot histogram of activations for this feature
        axes[i].hist(activations, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Feature {feature_idx}\nSparsity: {(activations > 0).mean():.3f}')
        axes[i].set_xlabel('Activation Value')
        axes[i].set_ylabel('Count')
        
        # Add statistics
        mean_val = activations.mean()
        max_val = activations.max()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        axes[i].axvline(max_val, color='orange', linestyle='--', alpha=0.7, label=f'Max: {max_val:.3f}')
        axes[i].legend(fontsize=8)
    
    # Remove unused subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f'plots/feature_analysis/{experiment_name}_{suffix}_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()


def find_top_activating_samples(sparse_dict, resnet_extractor, dataloader, device,
                               experiment_name, use_subset=False, top_k_features=20, top_k_samples=10):
    """
    Find samples that most strongly activate each feature, similar to Anthropic's approach.
    This helps understand what visual patterns each feature detects.
    
    Args:
        sparse_dict: Trained sparse dictionary
        resnet_extractor: ResNet feature extractor
        dataloader: DataLoader with images
        device: Computing device
        experiment_name: Name for saving
        use_subset: Whether to use feature subset
        top_k_features: Number of top features to analyze
        top_k_samples: Number of top activating samples to save per feature
    """
    sparse_dict.eval()
    
    # Collect all activations and keep track of which samples they came from
    all_activations = []
    all_images = []
    sample_indices = []
    
    print(f"Collecting samples and activations from layer '{resnet_extractor.layer_name}'...")
    with torch.no_grad():
        global_sample_idx = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx > 50:  # Limit to prevent memory issues
                break
                
            images = images.to(device)
            features = resnet_extractor(images)
            
            # Print feature shape for debugging (only once)
            if batch_idx == 0:
                print(f"Feature shape from {resnet_extractor.layer_name}: {features.shape}")
            
            if use_subset:
                indices = torch.randperm(features.shape[1])[:64]
                features = features[:, indices]
            
            activations = sparse_dict.encode(features)
            
            # Store activations and images
            all_activations.append(activations.cpu())
            all_images.append(images.cpu())
            
            # Keep track of sample indices
            batch_size = images.shape[0]
            sample_indices.extend(list(range(global_sample_idx, global_sample_idx + batch_size)))
            global_sample_idx += batch_size
    
    all_activations = torch.cat(all_activations, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    print(f"Finding top activating samples for {all_activations.shape[0]} total samples...")
    
    # Find most active features overall
    feature_max_activations = all_activations.max(dim=0)[0]
    top_features = torch.topk(feature_max_activations, top_k_features)[1]
    
    os.makedirs('plots/top_activating_samples', exist_ok=True)
    
    # For each top feature, find samples that activate it most
    for feature_rank, feature_idx in enumerate(top_features):
        feature_activations = all_activations[:, feature_idx]
        
        # Find top activating samples for this feature
        top_sample_values, top_sample_indices = torch.topk(feature_activations, top_k_samples)
        
        # Create visualization showing top activating images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(min(top_k_samples, len(top_sample_indices))):
            sample_idx = top_sample_indices[i]
            activation_value = top_sample_values[i]
            image = all_images[sample_idx]
            
            # Denormalize image for display (assuming ImageNet normalization)
            image_display = image.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_display = std * image_display + mean
            image_display = np.clip(image_display, 0, 1)
            
            axes[i].imshow(image_display)
            axes[i].set_title(f'Activation: {activation_value:.3f}')
            axes[i].axis('off')
        
        # Remove unused subplots
        for i in range(min(top_k_samples, len(top_sample_indices)), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Feature {feature_idx.item()} - Top Activating Images', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        filename = f'plots/top_activating_samples/{experiment_name}_feature_{feature_idx.item()}_top_samples.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if feature_rank < 5:  # Only print for first few
            print(f'Saved top activating samples for feature {feature_idx.item()}: {filename}')
    
    print(f"Top activating samples analysis complete! Check plots/top_activating_samples/")


def analyze_feature_interactions(sparse_dict, resnet_extractor, dataloader, device,
                               experiment_name, use_subset=False, n_features=50):
    """
    Analyze how features interact with each other (co-activation patterns).
    Similar to Anthropic's analysis of feature interactions and composition.
    
    Args:
        sparse_dict: Trained sparse dictionary
        resnet_extractor: ResNet feature extractor
        dataloader: DataLoader
        device: Computing device
        experiment_name: Name for saving
        use_subset: Whether to use feature subset
        n_features: Number of top features to analyze interactions for
    """
    sparse_dict.eval()
    all_activations = []
    
    print("Collecting activations for interaction analysis...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx > 30:  # Limit to prevent memory issues
                break
                
            images = images.to(device)
            features = resnet_extractor(images)
            
            if use_subset:
                indices = torch.randperm(features.shape[1])[:64]
                features = features[:, indices]
            
            activations = sparse_dict.encode(features)
            all_activations.append(activations.cpu())
    
    all_activations = torch.cat(all_activations, dim=0)
    
    # Select top features by maximum activation
    feature_max = all_activations.max(dim=0)[0]
    top_features = torch.topk(feature_max, min(n_features, len(feature_max)))[1]
    
    # Compute co-activation matrix for top features
    top_activations = all_activations[:, top_features]  # [samples, top_features]
    
    # Binary activation matrix (active/inactive)
    binary_activations = (top_activations > 0).float()
    
    # Compute co-activation probability matrix
    co_activation_matrix = torch.mm(binary_activations.t(), binary_activations) / len(binary_activations)
    
    # Compute correlation matrix of activation strengths
    correlation_matrix = torch.corrcoef(top_activations.t())
    
    # Create visualizations
    os.makedirs('plots/feature_interactions', exist_ok=True)
    
    # Plot co-activation matrix
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(co_activation_matrix.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Co-activation Probability')
    plt.title('Feature Co-activation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    
    # Plot correlation matrix
    plt.subplot(2, 2, 2)
    plt.imshow(correlation_matrix.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Activation Correlation')
    plt.title('Feature Activation Correlation')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    
    # Plot feature activation frequency
    activation_freq = binary_activations.mean(dim=0)
    plt.subplot(2, 2, 3)
    plt.bar(range(len(activation_freq)), activation_freq.numpy())
    plt.xlabel('Feature Index (Top Features)')
    plt.ylabel('Activation Frequency')
    plt.title('Feature Activation Frequencies')
    plt.xticks(range(0, len(activation_freq), max(1, len(activation_freq)//10)))
    
    # Plot number of simultaneously active features per sample
    n_active_per_sample = binary_activations.sum(dim=1)
    plt.subplot(2, 2, 4)
    plt.hist(n_active_per_sample.numpy(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Active Features')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Active Features per Sample')
    plt.axvline(n_active_per_sample.mean(), color='red', linestyle='--', 
               label=f'Mean: {n_active_per_sample.mean():.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/feature_interactions/{experiment_name}_feature_interactions.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find and report strongest feature pairs
    _analyze_feature_pairs(co_activation_matrix, correlation_matrix, top_features, experiment_name)
    
    print(f"Feature interaction analysis saved to plots/feature_interactions/")
    
    return {
        'co_activation_matrix': co_activation_matrix,
        'correlation_matrix': correlation_matrix,
        'top_features': top_features,
        'activation_frequencies': activation_freq
    }


def _analyze_feature_pairs(co_activation_matrix, correlation_matrix, top_features, experiment_name):
    """Analyze and report strongest feature pairs."""
    n_features = len(top_features)
    
    # Find strongest co-activating pairs (excluding diagonal)
    co_activation_no_diag = co_activation_matrix.clone()
    co_activation_no_diag.fill_diagonal_(0)
    
    # Get top co-activating pairs
    flat_indices = torch.topk(co_activation_no_diag.flatten(), 20)[1]
    pairs_co_activation = [(flat_indices[i] // n_features, flat_indices[i] % n_features, 
                          co_activation_no_diag.flatten()[flat_indices[i]]) 
                         for i in range(20)]
    
    # Find strongest correlating pairs
    correlation_no_diag = correlation_matrix.clone()
    correlation_no_diag.fill_diagonal_(0)
    flat_indices = torch.topk(torch.abs(correlation_no_diag).flatten(), 20)[1]
    pairs_correlation = [(flat_indices[i] // n_features, flat_indices[i] % n_features,
                        correlation_no_diag.flatten()[flat_indices[i]]) 
                       for i in range(20)]
    
    # Save report
    report_path = f'plots/feature_interactions/{experiment_name}_interaction_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Feature Interaction Analysis - {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top 20 Co-activating Feature Pairs:\n")
        f.write("-" * 40 + "\n")
        for i, j, prob in pairs_co_activation:
            f.write(f"Features {top_features[i].item():4d} & {top_features[j].item():4d}: "
                   f"co-activation = {prob:.4f}\n")
        
        f.write(f"\nTop 20 Correlated Feature Pairs:\n")
        f.write("-" * 40 + "\n")
        for i, j, corr in pairs_correlation:
            f.write(f"Features {top_features[i].item():4d} & {top_features[j].item():4d}: "
                   f"correlation = {corr:.4f}\n")
    
    print(f"Feature interaction report saved: {report_path}")


def create_monosemanticity_dashboard(sparse_dict, resnet_extractor, dataloader, device,
                                   experiment_name, use_subset=False):
    """
    Create a comprehensive dashboard similar to Anthropic's monosemanticity visualizations.
    Combines multiple analysis types into a single comprehensive view.
    """
    print("Creating comprehensive monosemanticity dashboard...")
    
    # Run all analyses
    print("1/4: Analyzing activation distributions...")
    dist_results = analyze_feature_activation_distributions(
        sparse_dict, resnet_extractor, dataloader, device, experiment_name, use_subset)
    
    print("2/4: Finding top activating samples...")
    find_top_activating_samples(
        sparse_dict, resnet_extractor, dataloader, device, experiment_name, use_subset)
    
    print("3/4: Analyzing feature interactions...")
    interaction_results = analyze_feature_interactions(
        sparse_dict, resnet_extractor, dataloader, device, experiment_name, use_subset)
    
    print("4/4: Running category analysis...")
    category_activations, category_counts = analyze_activations_per_category(
        sparse_dict, resnet_extractor, dataloader, device, use_subset)
    
    # Create summary dashboard
    _create_summary_dashboard(dist_results, interaction_results, category_activations, 
                            category_counts, experiment_name)
    
    print("Monosemanticity dashboard complete! Check all plots/ subdirectories for results.")


def _create_summary_dashboard(dist_results, interaction_results, category_activations, 
                            category_counts, experiment_name):
    """Create a summary dashboard combining key insights."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Sparsity distribution
    feature_sparsity = dist_results['feature_sparsity']
    axes[0, 0].hist(feature_sparsity.numpy(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Activation Frequency')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Sparsity Distribution')
    axes[0, 0].axvline(feature_sparsity.mean(), color='red', linestyle='--', 
                      label=f'Mean: {feature_sparsity.mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Activation strength vs sparsity
    feature_means = dist_results['feature_means']
    axes[0, 1].scatter(feature_sparsity.numpy(), feature_means.numpy(), alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Activation Frequency')
    axes[0, 1].set_ylabel('Mean Activation Strength')
    axes[0, 1].set_title('Sparsity vs Strength')
    
    # 3. Co-activation heatmap (subset)
    co_activation = interaction_results['co_activation_matrix'][:20, :20]  # Top 20x20
    im = axes[0, 2].imshow(co_activation.numpy(), cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Feature Co-activation (Top 20)')
    axes[0, 2].set_xlabel('Feature Index')
    axes[0, 2].set_ylabel('Feature Index')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Category activation summary
    if category_activations:
        n_categories = len(category_activations)
        category_names = list(category_activations.keys())[:10]  # Top 10 categories
        avg_activations = [category_activations[cat].mean().item() for cat in category_names]
        
        axes[1, 0].bar(range(len(category_names)), avg_activations)
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Average Activation')
        axes[1, 0].set_title('Category Average Activations')
        axes[1, 0].set_xticks(range(len(category_names)))
        axes[1, 0].set_xticklabels(category_names, rotation=45, ha='right')
    
    # 5. Feature activation frequencies
    activation_freq = interaction_results['activation_frequencies']
    axes[1, 1].bar(range(len(activation_freq)), activation_freq.numpy())
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Activation Frequency')
    axes[1, 1].set_title('Top Features Activation Frequency')
    
    # 6. Summary statistics with target sparsity info
    axes[1, 2].axis('off')
    stats_text = f"""
    Summary Statistics:
    
    Total Features: {len(feature_sparsity)}
    Mean Sparsity: {feature_sparsity.mean():.3f}
    Median Sparsity: {feature_sparsity.median():.3f}
    
    Target: 1-4 active features per sample
    
    Very Sparse Features (<0.1% active): {(feature_sparsity < 0.001).sum()}
    Sparse Features (0.1-1% active): {((feature_sparsity >= 0.001) & (feature_sparsity < 0.01)).sum()}
    Medium Features (1-5% active): {((feature_sparsity >= 0.01) & (feature_sparsity < 0.05)).sum()}
    Dense Features (>5% active): {(feature_sparsity >= 0.05).sum()}
    
    Categories Analyzed: {len(category_activations)}
    Total Samples: {sum(category_counts.values()) if category_counts else 'N/A'}
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save dashboard
    os.makedirs('plots/dashboard', exist_ok=True)
    plt.savefig(f'plots/dashboard/{experiment_name}_monosemanticity_dashboard.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary dashboard saved: plots/dashboard/{experiment_name}_monosemanticity_dashboard.png")


def run_post_training_analysis(sparse_dict, resnet_extractor, dataloader, device, 
                             experiment_name, use_subset=False):
    """
    Run complete post-training analysis of category activations.
    
    Args:
        sparse_dict: Trained sparse dictionary
        resnet_extractor: ResNet feature extractor  
        dataloader: DataLoader with category information
        device: Computing device
        experiment_name: Name for saving files
        use_subset: Whether to use feature subset
    """
    print("Running comprehensive post-training analysis...")
    
    # Create the comprehensive monosemanticity dashboard
    create_monosemanticity_dashboard(sparse_dict, resnet_extractor, dataloader, device, 
                                   experiment_name, use_subset)
    
    # Also run the original category-specific analysis
    print("Running category-specific analysis...")
    category_activations, category_counts = analyze_activations_per_category(
        sparse_dict, resnet_extractor, dataloader, device, use_subset
    )
    
    print(f"Found {len(category_activations)} categories:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} samples")
    
    # Create visualizations
    print("Creating category activation visualizations...")
    visualize_category_activations(category_activations, category_counts, experiment_name)
    
    print("Creating activation heatmap...")
    create_activation_heatmap(category_activations, experiment_name)
    
    # Analyze feature specialization
    print("Analyzing feature specialization...")
    analyze_feature_specialization(category_activations, experiment_name)
    
    print("Complete post-training analysis finished! Check all plots/ subdirectories for results.")
    
    return category_activations, category_counts
