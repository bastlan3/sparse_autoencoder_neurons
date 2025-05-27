import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms # Already present in original script
from pathlib import Path # Changed from pathlib to Path for direct use
from torch.utils.data import Dataset # For the base class

# Custom Dataset to load from multiple session datasets produced by preprocess(i), with lazy evaluation and caching
class MultiSessionConcatDataset(Dataset): # Changed base class to Dataset
    def __init__(self, session_ids, preprocess_fn, images=False): # PEP8: images=False
        """
        Args:
            session_ids (list): List of session identifiers.
            preprocess_fn (callable): Function that returns a dataset for a session.
            images (bool): Whether to load images along with neural data.
        """
        self.session_ids = session_ids
        self.need_images = images
        self.preprocess_fn = preprocess_fn
        self.datasets = [None] * len(session_ids)  # Will be loaded lazily
        self.lengths = [None] * len(session_ids) # Stores length of each session's dataset
        self.cum_lengths = None # Cumulative lengths for global indexing
        self._cache_idx = None # Index of the currently cached dataset
        self._cache_dataset = None # Cached dataset (features)
        self.img_names = [None] * len(session_ids) # Stores image names/paths for each session

        # Path for precomputed lengths CSV, adjust if necessary relative to this file's location
        # Assuming this script is in 'refactored_code', and CSV is in 'neural_switch/data'
        # This relative path might need adjustment based on execution context.
        # For now, using the provided path, assuming it's accessible.
        self.precomputed_lengths_path = Path('./../neural_switch/data/precomputed_lengths.csv') # More robust path
        
        if self.precomputed_lengths_path.exists():
            self.precomputed_lengths_df = pd.read_csv(self.precomputed_lengths_path)
        else:
            self.precomputed_lengths_df = pd.DataFrame(columns=['session_id', 'length'])
        
        self._compute_lengths_and_total()

    def _compute_lengths_and_total(self):
        # Computes the lengths of individual datasets and the total length.
        # Filters out invalid sessions.
        cumulative_lengths = [0]
        valid_session_ids = []
        valid_lengths = []
        valid_datasets_temp_storage = [] # To store loaded datasets temporarily for length check
        valid_img_names_temp_storage = []

        for idx, sid in enumerate(self.session_ids):
            length = -1
            # Check if length is precomputed
            if not self.precomputed_lengths_df[self.precomputed_lengths_df['session_id'] == sid].empty:
                length = self.precomputed_lengths_df[self.precomputed_lengths_df['session_id'] == sid]['length'].iloc[0]
            
            current_dataset_features, current_img_names = self._load_and_process_session(idx, sid)

            if current_dataset_features is None or current_dataset_features.shape[0] == 0 or current_dataset_features.shape[-1] != 64:
                print(f"Skipping invalid or empty session {sid} (features shape: {current_dataset_features.shape if current_dataset_features is not None else 'None'}).")
                if length != -1 and length != 0 : # Precomputed length was wrong
                     self.precomputed_lengths_df = self.precomputed_lengths_df[self.precomputed_lengths_df['session_id'] != sid] # Remove incorrect entry
                continue

            # If length was not precomputed or was incorrect (e.g. 0 for a now valid session)
            if length == -1 or length != current_dataset_features.shape[0]:
                length = current_dataset_features.shape[0]
                # Update DataFrame
                if sid in self.precomputed_lengths_df['session_id'].values:
                    self.precomputed_lengths_df.loc[self.precomputed_lengths_df['session_id'] == sid, 'length'] = length
                else:
                    new_row = pd.DataFrame([{'session_id': sid, 'length': length}])
                    self.precomputed_lengths_df = pd.concat([self.precomputed_lengths_df, new_row], ignore_index=True)
                
                try: # Save updated lengths
                    self.precomputed_lengths_df.to_csv(self.precomputed_lengths_path, index=False)
                except IOError as e:
                    print(f"Warning: Could not save precomputed_lengths.csv: {e}")


            valid_session_ids.append(sid)
            valid_lengths.append(length)
            cumulative_lengths.append(cumulative_lengths[-1] + length)
            valid_datasets_temp_storage.append(current_dataset_features) # Store for assignment later
            valid_img_names_temp_storage.append(current_img_names)


        # Update instance variables with validated sessions and their data
        self.session_ids = valid_session_ids
        self.lengths = valid_lengths
        self.datasets = valid_datasets_temp_storage # Assign the loaded and validated datasets
        self.img_names = valid_img_names_temp_storage
        self.cum_lengths = cumulative_lengths
        self.total_length = cumulative_lengths[-1] if cumulative_lengths else 0
        
        if self.total_length == 0:
            print("Warning: Dataset is empty after processing all sessions.")

    def _load_and_process_session(self, original_idx, session_id):
        # Helper to load and process a single session's data.
        # This avoids repeated logic in _compute_lengths and _get_dataset.
        # The original_idx is the index in the initial self.session_ids list.
        # Note: This function might be called by _compute_lengths_and_total before self.datasets is finalized.
        
        # Path to neural data, adjust if necessary
        # Assuming it's relative to where the main script using this class runs from.
        neural_data_path = Path('./../neural_switch/data/neural_data') # More robust path

        try:
            features, img_paths = self.preprocess_fn(session_id, neural_data_path)
        except Exception as e:
            print(f"Error in preprocess_fn for session {session_id}: {e}")
            return None, None

        if features is None or features.shape[-1] != 64: # Assuming 64 is a critical dimension (e.g. num neurons)
            # print(f"Warning: Preprocessing returned None or incorrect feature dimension for session {session_id}.")
            return None, None
        
        # Reshape and average over the 3rd dimension (e.g., time bins or trials within a stimulus presentation)
        # Original: (n_stimuli, 30 timepoints, 20 trials/units, 64 neurons) -> (n_stimuli, 30 timepoints, 64 neurons)
        try:
            processed_features = features.reshape(features.shape[0], 30, 20, 64).mean(axis=2)
        except ValueError as e:
            print(f"Error reshaping features for session {session_id}. Shape was {features.shape}. Error: {e}")
            return None, None
            
        return processed_features, img_paths

    def _get_dataset_by_session_index(self, session_idx_in_valid_list):
        # This gets the dataset for a session that is already validated and its data stored.
        # It uses the index from the *validated* list of sessions.
        if self.datasets[session_idx_in_valid_list] is None:
            # This case should ideally not happen if _compute_lengths_and_total populates self.datasets correctly.
            # However, as a fallback or for truly lazy loading after initial length computation:
            sid = self.session_ids[session_idx_in_valid_list]
            print(f"Re-loading session {sid} in _get_dataset_by_session_index. This should be rare.")
            self.datasets[session_idx_in_valid_list], self.img_names[session_idx_in_valid_list] = self._load_and_process_session(None, sid) # original_idx not critical here

        return self.datasets[session_idx_in_valid_list], self.img_names[session_idx_in_valid_list]


    def __len__(self):
        return self.total_length

    def __getitem__(self, global_idx):
        if not self.total_length or global_idx < 0 or global_idx >= self.total_length:
            raise IndexError(f"Index {global_idx} out of bounds for dataset with total length {self.total_length}")

        # Find which session this global_idx falls into
        session_idx_in_valid_list = np.searchsorted(self.cum_lengths, global_idx, side='right') - 1
        
        # Calculate local_idx within that session
        local_idx = global_idx - self.cum_lengths[session_idx_in_valid_list]
        
        features_for_session, img_paths_for_session = self._get_dataset_by_session_index(session_idx_in_valid_list)

        if features_for_session is None or local_idx >= features_for_session.shape[0]:
             # This indicates an issue with length calculation or data loading consistency
            print(f"Error: Data for session {self.session_ids[session_idx_in_valid_list]} (idx {session_idx_in_valid_list}) seems inconsistent or local_idx {local_idx} is out of bounds for shape {features_for_session.shape if features_for_session is not None else 'None'}.")
            # Fallback: attempt to reload or skip. For now, raise error or return dummy.
            # This might happen if a session was deemed valid during _compute_lengths but fails here.
            # Or if _get_dataset_by_session_index fails to load.
            raise RuntimeError(f"Failed to get item for session {self.session_ids[session_idx_in_valid_list]}")


        current_features = features_for_session[local_idx]

        if self.need_images:
            img_path_relative = img_paths_for_session[local_idx]
            # Base path for images, adjust if necessary
            # Assuming this path is relative to where the main script runs.
            img_base_path = Path('./../data/manyOO/')
            
            # Try primary path, then fallback to 'TestItems' subdirectory
            img_full_path = img_base_path / img_path_relative
            if not img_full_path.is_file():
                img_full_path = img_base_path / 'TestItems' / img_path_relative

            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True), # antialias for modern torchvision
                transforms.Lambda(lambda x: x[:3, :, :]) # Ensure 3 channels
            ])
            try:
                with Image.open(img_full_path) as img:
                    img_tensor = img_transform(img.convert('RGB')) # Ensure RGB
            except FileNotFoundError:
                print(f"Error: Image file not found at {img_full_path}")
                img_tensor = torch.zeros((3, 256, 256)) # Return a dummy tensor
            except Exception as e:
                print(f"Could not load or transform image {img_full_path}: {e}")
                img_tensor = torch.zeros((3, 256, 256)) # Return a dummy tensor
            
            return current_features, img_tensor
        else:
            return current_features
