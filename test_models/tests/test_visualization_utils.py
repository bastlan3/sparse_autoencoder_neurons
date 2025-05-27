import unittest
from unittest.mock import patch, MagicMock
import torch
import os # For os.makedirs, though it's mocked

# Assuming the project structure allows this import path
from test_models.utils.visualization_utils import visualize_dictionary_features

class TestVisualizationUtils(unittest.TestCase):

    @patch('test_models.utils.visualization_utils.os.makedirs')
    @patch('test_models.utils.visualization_utils.plt') # Mock the entire matplotlib.pyplot module
    def test_visualize_dictionary_features(self, mock_plt, mock_makedirs):
        """
        Test that visualize_dictionary_features calls plotting functions,
        creates directories, saves the plot, and closes the figure.
        """
        # Create a mock dictionary object
        # It needs a .decoder.weight attribute that is a torch.Tensor
        mock_dictionary = MagicMock()
        # input_dim=16 (for 4x4 reshape), dict_size=5
        mock_dictionary.decoder.weight = torch.randn(16, 5) 

        epoch = 1
        experiment_name = 'test_exp'
        
        # Call the function to be tested
        visualize_dictionary_features(mock_dictionary, epoch, experiment_name, n_elements=4)

        # Assert that os.makedirs was called correctly
        mock_makedirs.assert_called_once_with('plots', exist_ok=True)

        # Assert that a figure and subplots were created
        mock_plt.subplots.assert_called_once()
        
        # Get the figure object that subplots would return
        # In the actual code, fig, axes = plt.subplots(...)
        # So, mock_plt.subplots.return_value gives us the (fig, axes) tuple.
        # We need to ensure the 'fig' part of this tuple is what's passed to savefig and close.
        # However, the way it's structured, plt.savefig and plt.close are module-level calls
        # in the original code, which means they use the "current" figure.
        # The refactored code correctly uses fig.savefig and fig.suptitle, but then plt.savefig and plt.close.
        # The provided code for visualize_dictionary_features uses plt.savefig and plt.close(fig).

        # Assert that savefig was called with the correct path
        expected_plot_filename = f'plots/{experiment_name}_dict_epoch_{epoch}.png'
        mock_plt.savefig.assert_called_once_with(expected_plot_filename)
        
        # Assert that the plot was closed
        # The function passes the 'fig' object to plt.close()
        # mock_plt.subplots() returns a tuple (fig, axes)
        # We need to ensure that the first element of this tuple (the fig) is passed to plt.close
        returned_fig_object = mock_plt.subplots.return_value[0] # This is the 'fig'
        mock_plt.close.assert_called_once_with(returned_fig_object)

        # Check some other calls for completeness (optional, but good practice)
        mock_plt.tight_layout.assert_called_once()
        
        # Check that suptitle was called on the figure object
        returned_fig_object.suptitle.assert_called_once()


    @patch('test_models.utils.visualization_utils.os.makedirs')
    @patch('test_models.utils.visualization_utils.plt')
    def test_visualize_dictionary_features_few_elements(self, mock_plt, mock_makedirs):
        """
        Test with n_elements less than default and also less than dictionary size.
        """
        mock_dictionary = MagicMock()
        mock_dictionary.decoder.weight = torch.randn(16, 5) # dict_size = 5

        visualize_dictionary_features(mock_dictionary, epoch=2, experiment_name='test_few', n_elements=3)

        mock_makedirs.assert_called_once_with('plots', exist_ok=True)
        mock_plt.savefig.assert_called_once_with('plots/test_few_dict_epoch_2.png')
        
        # Check that subplots was called to generate enough axes for 3 elements (e.g., 2x2 grid)
        # The logic is: plot_cols = ceil(sqrt(3)) = 2; plot_rows = ceil(3/2) = 2. So, 2x2=4 axes.
        mock_plt.subplots.assert_called_once_with(2, 2, figsize=(15,10))
        
        # Ensure that only 3 dictionary elements are processed and plotted
        # The loop runs 'for i, idx in enumerate(top_indices):'
        # top_indices will have length min(n_elements, num_available_elements) = min(3, 5) = 3.
        # So, set_title should be called 3 times.
        # axes.flatten()[i].set_title(...)
        axes_array_mock = mock_plt.subplots.return_value[1].flatten() # This is axes.flatten()
        self.assertEqual(axes_array_mock[0].set_title.call_count, 1)
        self.assertEqual(axes_array_mock[1].set_title.call_count, 1)
        self.assertEqual(axes_array_mock[2].set_title.call_count, 1)
        # Ensure the 4th subplot's title was not set, and it was deleted
        self.assertEqual(axes_array_mock[3].set_title.call_count, 0)
        mock_plt.subplots.return_value[0].delaxes.assert_called_once_with(axes_array_mock[3])


if __name__ == '__main__':
    unittest.main()
