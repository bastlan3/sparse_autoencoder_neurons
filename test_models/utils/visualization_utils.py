import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def visualize_dictionary_features(dictionary, epoch, experiment_name, n_elements=20, figsize=(15,10)):
    """
    Visualizes the top N dictionary elements (decoder weights) from a sparse dictionary model
    and saves the plot to a file.

    The function calculates the importance of each dictionary element (L2 norm of its weights),
    selects the top N most important elements, and plots them. The plot is saved
    to a directory named 'plots'.

    Args:
        dictionary (torch.nn.Module): The sparse dictionary model. It is assumed to have
                                      a `decoder.weight` attribute containing the dictionary
                                      elements (e.g., from a `SparseAutoencoder`).
        epoch (int): The current training epoch number. Used for naming the saved plot.
        experiment_name (str): A name for the current experiment. Used for naming the
                               saved plot and in the plot title.
        n_elements (int, optional): The number of top dictionary elements to visualize.
                                    Defaults to 20.
        figsize (tuple, optional): The size of the matplotlib figure. Defaults to (15, 10).

    Side Effects:
        Creates a directory named 'plots' if it doesn't already exist.
        Saves a PNG image file of the plot to the 'plots' directory.
        The filename will be in the format: 'plots/{experiment_name}_dict_epoch_{epoch}.png'.
    """
    decoder_weights = dictionary.decoder.weight.detach().cpu().numpy()

    # Calculate importance of each dictionary element (L2 norm)
    # The dictionary elements are the columns of the decoder weight matrix
    importance = np.linalg.norm(decoder_weights, axis=0)
    
    # Ensure n_elements is not greater than the number of available dictionary elements
    num_available_elements = decoder_weights.shape[1]
    n_elements_to_plot = min(n_elements, num_available_elements)

    top_indices = np.argsort(importance)[-n_elements_to_plot:][::-1]

    # Get input dimensions (feature dimension being reconstructed by each dictionary element)
    input_dim = decoder_weights.shape[0]

    # Try to determine if we can reshape to 2D (e.g., for image patches or spectrograms)
    # This is a heuristic and might need adjustment based on the actual data shape.
    # Assuming square-ish or rectangular features.
    n_rows = int(np.sqrt(input_dim))
    can_reshape_to_2d = (n_rows * n_rows == input_dim) or (input_dim % n_rows == 0)
    if can_reshape_to_2d:
        n_cols_reshape = input_dim // n_rows
    else:
        # Fallback for 1D features
        n_rows = 1 
        n_cols_reshape = input_dim


    # Determine plot grid: try to make it somewhat square
    plot_cols = int(np.ceil(np.sqrt(n_elements_to_plot)))
    plot_rows = int(np.ceil(n_elements_to_plot / plot_cols))

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=figsize)
    axes = axes.flatten() # Flatten to easily iterate

    for i, idx in enumerate(top_indices):
        if i >= len(axes): # Should not happen if plot_rows/cols are correct
            break
            
        dict_element = decoder_weights[:, idx] # A single column (dictionary element)
        
        ax = axes[i]
        # Try to reshape to 2D if possible
        if can_reshape_to_2d and n_rows > 1: # Only reshape if it makes sense (not 1xN)
            try:
                dict_element_2d = dict_element.reshape(n_rows, n_cols_reshape)
                im = ax.imshow(dict_element_2d, cmap='viridis', aspect='auto')
                fig.colorbar(im, ax=ax) # Add colorbar for context
            except Exception as e:
                # Fallback to 1D plot if reshape fails for some reason
                ax.plot(dict_element)
                print(f"Could not reshape element {idx} to 2D: {e}")
        else:
            ax.plot(dict_element)
            
        ax.set_title(f"Element {idx}\n(Importance: {importance[idx]:.2f})")
        ax.axis('off') # Turn off axis numbers and ticks

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Dictionary Elements for {experiment_name} at Epoch {epoch}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_filename = f'plots/{experiment_name}_dict_epoch_{epoch}.png'
    plt.savefig(plot_filename)
    print(f"Saved dictionary visualization to {plot_filename}")
    plt.close(fig) # Close the figure to free memory
