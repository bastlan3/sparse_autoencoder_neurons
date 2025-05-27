import unittest
import torch
import torch.nn as nn

# To import DictionaryLearner, we need to ensure main_dictionary_learning.py
# can be imported. This might require it to be structured to allow class import,
# or temporarily add its path. For testing, it's common to adjust sys.path
# or assume the test runner handles it.
# Here, we'll try a direct import path assuming the structure allows it,
# or that main_dictionary_learning.py defines the class at the top level.

# If main_dictionary_learning.py is in root, and tests/ is a dir in root:
# One way is to add project root to sys.path for the test.
import sys
import os
# Assuming the test is run from the project root or tests/ directory
# Add project root to path to find main_dictionary_learning
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main_dictionary_learning import DictionaryLearner


class TestDictionaryLearner(unittest.TestCase):

    def test_initialization(self):
        """Test the initialization of the DictionaryLearner model."""
        input_dim = 64
        n_components = 128
        model = DictionaryLearner(input_dim, n_components)

        self.assertIsInstance(model, nn.Module)
        
        # Check encoder structure
        self.assertIsInstance(model.encoder, nn.Sequential)
        self.assertEqual(len(model.encoder), 2) # Linear + ReLU
        self.assertIsInstance(model.encoder[0], nn.Linear)
        self.assertIsInstance(model.encoder[1], nn.ReLU)
        
        # Check encoder weight shapes
        self.assertEqual(model.encoder[0].weight.shape, (n_components, input_dim))
        self.assertEqual(model.encoder[0].bias.shape, (n_components,))
        
        # Check decoder structure
        self.assertIsInstance(model.decoder, nn.Linear)
        # Check decoder weight shapes
        self.assertEqual(model.decoder.weight.shape, (input_dim, n_components))
        self.assertEqual(model.decoder.bias.shape, (input_dim,))

    def test_forward_pass(self):
        """Test the forward pass of the DictionaryLearner model."""
        input_dim = 64
        n_components = 128
        batch_size = 10
        model = DictionaryLearner(input_dim, n_components)

        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, input_dim)

        # Perform a forward pass
        reconstructed_output = model(dummy_input)

        # Check output shape
        self.assertEqual(reconstructed_output.shape, (batch_size, input_dim))

        # Check intermediate encoded shape and ReLU activation
        encoded_output = model.encoder(dummy_input)
        self.assertEqual(encoded_output.shape, (batch_size, n_components))
        # Check if ReLU was applied (all values >= 0)
        self.assertTrue(torch.all(encoded_output >= 0))
        
    def test_forward_pass_different_dims(self):
        """Test with different dimensions to ensure robustness."""
        input_dim = 256
        n_components = 512
        batch_size = 5
        model = DictionaryLearner(input_dim, n_components)
        dummy_input = torch.randn(batch_size, input_dim)
        reconstructed_output = model(dummy_input)
        self.assertEqual(reconstructed_output.shape, (batch_size, input_dim))
        encoded_output = model.encoder(dummy_input)
        self.assertEqual(encoded_output.shape, (batch_size, n_components))
        self.assertTrue(torch.all(encoded_output >= 0))


if __name__ == '__main__':
    unittest.main()
