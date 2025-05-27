import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Assuming 'data' is a top-level directory and the script is run from project root
# or PYTHONPATH is set up.
from data.neural_data_loader import MultiSessionConcatDataset

# Define a global mock preprocess_fn for use in tests
def mock_preprocess_fn(session_id, neural_data_path):
    """
    Mock preprocessing function.
    neural_data_path is ignored in this mock.
    """
    if session_id == "sess1":
        # (n_stimuli, 30 timepoints, 20 trials/units, 64 neurons)
        return (np.random.rand(10, 30, 20, 64).astype(np.float32), 
                [f"sess1_img_{i}.png" for i in range(10)])
    elif session_id == "sess2":
        return (np.random.rand(5, 30, 20, 64).astype(np.float32), 
                [f"sess2_img_{i}.png" for i in range(5)])
    elif session_id == "empty_sess": # Session returns 0 samples
        return (np.zeros((0, 30, 20, 64)).astype(np.float32), [])
    elif session_id == "bad_features_sess": # Features with wrong final dimension
        return (np.random.rand(5, 30, 20, 32).astype(np.float32), 
                [f"bad_sess_img_{i}.png" for i in range(5)])
    elif session_id == "error_sess": # Simulates an error during preprocessing
        raise ValueError("Simulated preprocessing error")
    elif session_id == "none_features_sess": # Simulates preprocess_fn returning None for features
        return (None, [f"none_feat_img_{i}.png" for i in range(5)])
    else:
        # Default for any other session_id, effectively an empty/invalid session
        return (None, None)


class TestMultiSessionConcatDataset(unittest.TestCase):

    def setUp(self):
        self.session_ids_valid = ["sess1", "sess2"]
        self.mock_preprocess_fn_global = mock_preprocess_fn

        # Mock image for Image.open
        self.mock_pil_image = Image.new('RGB', (256, 256), color='blue')

    @patch('data.neural_data_loader.pd.DataFrame.to_csv')
    @patch('data.neural_data_loader.pd.read_csv')
    @patch('data.neural_data_loader.Path.exists')
    def test_basic_concatenation_and_lengths(self, mock_path_exists, mock_read_csv, mock_to_csv):
        """Test basic dataset concatenation, length, and cumulative lengths."""
        mock_path_exists.return_value = False # No precomputed_lengths.csv

        dataset = MultiSessionConcatDataset(
            session_ids=self.session_ids_valid,
            preprocess_fn=self.mock_preprocess_fn_global
        )

        self.assertEqual(len(dataset), 15) # 10 from sess1 + 5 from sess2
        self.assertEqual(dataset.total_length, 15)
        self.assertEqual(dataset.session_ids, ["sess1", "sess2"]) # Order maintained
        self.assertEqual(dataset.lengths, [10, 5])
        self.assertEqual(dataset.cum_lengths, [0, 10, 15])
        # Check that to_csv was called for sess1 and sess2 as lengths were computed
        self.assertEqual(mock_to_csv.call_count, 2)


    @patch('data.neural_data_loader.Image.open')
    @patch('data.neural_data_loader.Path.is_file')
    @patch('data.neural_data_loader.pd.DataFrame.to_csv')
    @patch('data.neural_data_loader.pd.read_csv')
    @patch('data.neural_data_loader.Path.exists')
    def test_getitem_no_images(self, mock_path_exists, mock_read_csv, mock_to_csv, mock_is_file, mock_image_open):
        """Test __getitem__ when images=False."""
        mock_path_exists.return_value = False
        mock_is_file.return_value = True # Assume all image paths are valid files
        mock_image_open.return_value = self.mock_pil_image

        dataset = MultiSessionConcatDataset(
            session_ids=self.session_ids_valid,
            preprocess_fn=self.mock_preprocess_fn_global,
            images=False # Default
        )
        
        # Item from sess1
        features_sess1 = dataset[0] 
        self.assertIsInstance(features_sess1, torch.Tensor)
         # Expected shape: (30 timepoints, 64 neurons) after .mean(axis=2) in _load_and_process_session
        self.assertEqual(features_sess1.shape, (30, 64))

        # Item from sess2 (index 10 is the first item of sess2)
        features_sess2 = dataset[10]
        self.assertIsInstance(features_sess2, torch.Tensor)
        self.assertEqual(features_sess2.shape, (30, 64))
        
        # Check that Image.open was not called
        mock_image_open.assert_not_called()

    @patch('data.neural_data_loader.Image.open')
    @patch('data.neural_data_loader.Path.is_file')
    @patch('data.neural_data_loader.pd.DataFrame.to_csv')
    @patch('data.neural_data_loader.pd.read_csv')
    @patch('data.neural_data_loader.Path.exists')
    def test_getitem_with_images(self, mock_path_exists, mock_read_csv, mock_to_csv, mock_is_file, mock_image_open):
        """Test __getitem__ when images=True."""
        mock_path_exists.return_value = False
        mock_is_file.return_value = True # Assume all image paths are valid
        mock_image_open.return_value = self.mock_pil_image

        dataset = MultiSessionConcatDataset(
            session_ids=self.session_ids_valid,
            preprocess_fn=self.mock_preprocess_fn_global,
            images=True
        )

        features, img_tensor = dataset[0] # First item from sess1
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (30, 64))
        self.assertIsInstance(img_tensor, torch.Tensor)
        # Default transform in __getitem__ for images: Resize(256,256), ToTensor, Lambda for 3 channels
        self.assertEqual(img_tensor.shape, (3, 256, 256)) 
        mock_image_open.assert_called_once() # Should be called once for this item
        
        # Check path construction for image (assuming FILE_BASE_PATH / '../data/manyOO/' / img_path_relative)
        # The actual img_path_relative is "sess1_img_0.png"
        # FILE_BASE_PATH for data/neural_data_loader.py is data/
        # So, data/../data/manyOO/sess1_img_0.png = data/manyOO/sess1_img_0.png
        expected_img_path = Path('data/manyOO/sess1_img_0.png')
        
        # mock_image_open.assert_called_with(expected_img_path) # This is tricky due to Path object comparison
        args, _ = mock_image_open.call_args
        called_path = args[0]
        self.assertIsInstance(called_path, Path)
        # Check parts of the path if direct comparison is problematic
        self.assertTrue(str(called_path).endswith('data/manyOO/sess1_img_0.png'))


    @patch('data.neural_data_loader.pd.DataFrame.to_csv')
    @patch('data.neural_data_loader.pd.read_csv')
    @patch('data.neural_data_loader.Path.exists')
    def test_session_skipping_and_error_handling(self, mock_path_exists, mock_read_csv, mock_to_csv):
        """Test that invalid sessions are skipped and errors handled."""
        mock_path_exists.return_value = False
        session_ids = ["sess1", "empty_sess", "bad_features_sess", "error_sess", "none_features_sess", "sess2"]
        
        dataset = MultiSessionConcatDataset(
            session_ids=session_ids,
            preprocess_fn=self.mock_preprocess_fn_global
        )

        self.assertEqual(len(dataset), 15) # Only sess1 (10) and sess2 (5) should be included
        self.assertEqual(dataset.session_ids, ["sess1", "sess2"])
        self.assertEqual(dataset.lengths, [10, 5])
        self.assertEqual(dataset.cum_lengths, [0, 10, 15])

    @patch('data.neural_data_loader.pd.DataFrame.to_csv')
    @patch('data.neural_data_loader.pd.read_csv')
    @patch('data.neural_data_loader.Path.exists')
    def test_precomputed_lengths_integration(self, mock_path_exists, mock_read_csv, mock_to_csv):
        """Test interaction with precomputed_lengths.csv."""
        # Scenario 1: precomputed_lengths.csv does not exist
        mock_path_exists.return_value = False
        dataset1 = MultiSessionConcatDataset(["sess1"], self.mock_preprocess_fn_global)
        self.assertEqual(len(dataset1), 10)
        mock_to_csv.assert_called_once() # Called to save newly computed length for sess1
        # The path used by to_csv should be dataset1.precomputed_lengths_path
        args_to_csv, _ = mock_to_csv.call_args
        self.assertEqual(args_to_csv[0], dataset1.precomputed_lengths_path)


        mock_to_csv.reset_mock()

        # Scenario 2: precomputed_lengths.csv exists and lengths are correct
        mock_path_exists.return_value = True
        mock_df = pd.DataFrame([{'session_id': 'sess1', 'length': 10}])
        mock_read_csv.return_value = mock_df
        
        dataset2 = MultiSessionConcatDataset(["sess1"], self.mock_preprocess_fn_global)
        self.assertEqual(len(dataset2), 10)
        mock_read_csv.assert_called_once_with(dataset2.precomputed_lengths_path)
        mock_to_csv.assert_not_called() # Not called because length matched

        mock_read_csv.reset_mock()
        mock_to_csv.reset_mock()

        # Scenario 3: precomputed_lengths.csv exists, but length for a session is incorrect
        mock_df_incorrect = pd.DataFrame([{'session_id': 'sess1', 'length': 8}, {'session_id': 'sess2', 'length': 5}])
        mock_read_csv.return_value = mock_df_incorrect
        
        dataset3 = MultiSessionConcatDataset(["sess1", "sess2"], self.mock_preprocess_fn_global)
        self.assertEqual(len(dataset3), 15) # sess1 is 10, sess2 is 5
        self.assertEqual(dataset3.lengths, [10, 5]) # Correct lengths are used
        # to_csv should be called to update the incorrect length of sess1
        # (and possibly for sess2 if its entry was missing, but here it's correct)
        
        # Check the updated DataFrame that would be saved
        # The df in dataset3.precomputed_lengths_df should be updated
        updated_df_sess1_length = dataset3.precomputed_lengths_df[
            dataset3.precomputed_lengths_df['session_id'] == 'sess1'
        ]['length'].iloc[0]
        self.assertEqual(updated_df_sess1_length, 10)
        
        mock_to_csv.assert_called() # Called at least for sess1
        
        # Check if it was called for 'sess1' because its length changed
        # This is a bit more involved to check precisely which rows triggered save without more complex mocking
        # But we know it must have been called due to the mismatch for sess1.
        # A simple check is that it's called. If it was called once, it implies sess1 was updated.
        # If sess2 was also missing from a hypothetical csv and then added, call_count would be 2.
        # Here, sess1 (8->10) triggers save. sess2 (5->5) does not. So 1 call.
        self.assertEqual(mock_to_csv.call_count, 1)


if __name__ == '__main__':
    unittest.main()
