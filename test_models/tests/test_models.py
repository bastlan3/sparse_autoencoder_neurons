import unittest
import torch

# Assuming the project structure allows this import path
from test_models.models.resnet_extractor import ResNetFeatureExtractor
from test_models.models.sparse_dictionary import SparseDictionary

class TestResNetFeatureExtractor(unittest.TestCase):

    def setUp(self):
        """Set up the device for tests."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for ResNetFeatureExtractor tests")

    def test_instantiation(self):
        """Test that ResNetFeatureExtractor instantiates correctly."""
        try:
            model = ResNetFeatureExtractor(model_name='resnet50', layer_name='avgpool', pretrained=False)
            model.to(self.device)
            self.assertIsNotNone(model, "Model should not be None after instantiation.")
        except Exception as e:
            self.fail(f"ResNetFeatureExtractor instantiation failed: {e}")

    def test_forward_pass_and_output_shape(self):
        """Test the forward pass and the output shape of ResNetFeatureExtractor."""
        model = ResNetFeatureExtractor(model_name='resnet50', layer_name='avgpool', pretrained=False)
        model.to(self.device)
        model.eval() # Set model to evaluation mode

        # Batch size of 2, 3 color channels, 224x224 image
        dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
        
        with torch.no_grad(): # No need to track gradients for inference
            features = model(dummy_input)
        
        # ResNet50 with 'avgpool' layer should output (batch_size, 2048)
        # The features are flattened after avgpool.
        expected_shape = (2, 2048) 
        self.assertEqual(features.shape, expected_shape,
                         f"Output shape mismatch. Expected {expected_shape}, got {features.shape}")

class TestSparseDictionary(unittest.TestCase):

    def setUp(self):
        """Set up the device for tests."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for SparseDictionary tests")
        self.input_dim = 64
        self.dict_size = 128

    def test_instantiation(self):
        """Test that SparseDictionary instantiates correctly."""
        try:
            model = SparseDictionary(input_dim=self.input_dim, 
                                     dict_size=self.dict_size, 
                                     sparsity_coef=0.01)
            model.to(self.device)
            self.assertIsNotNone(model, "Model should not be None after instantiation.")
        except Exception as e:
            self.fail(f"SparseDictionary instantiation failed: {e}")

    def test_forward_pass_and_output_shapes(self):
        """Test the forward pass and output shapes of SparseDictionary."""
        model = SparseDictionary(input_dim=self.input_dim, 
                                 dict_size=self.dict_size, 
                                 sparsity_coef=0.01)
        model.to(self.device)
        model.eval() # Set model to evaluation mode

        # Batch size of 2, feature dimension = input_dim
        dummy_features = torch.randn(2, self.input_dim).to(self.device)

        with torch.no_grad(): # No need to track gradients for inference
            reconstructed, activations = model(dummy_features)

        # Reconstructed features should have the same shape as input features
        expected_reconstructed_shape = (2, self.input_dim)
        self.assertEqual(reconstructed.shape, expected_reconstructed_shape,
                         f"Reconstructed shape mismatch. Expected {expected_reconstructed_shape}, got {reconstructed.shape}")

        # Activations should have shape (batch_size, dict_size)
        expected_activations_shape = (2, self.dict_size)
        self.assertEqual(activations.shape, expected_activations_shape,
                         f"Activations shape mismatch. Expected {expected_activations_shape}, got {activations.shape}")

if __name__ == '__main__':
    unittest.main()
