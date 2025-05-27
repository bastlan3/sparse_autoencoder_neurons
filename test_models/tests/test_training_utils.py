import unittest
import torch
import torch.nn.functional as F # For Huber loss calculation reference

# Assuming the project structure allows this import path
from test_models.utils.training_utils import (
    compute_reconstruction_loss,
    compute_sparsity_loss,
    compute_orthogonality_loss,
    compute_total_loss
)

class TestTrainingUtils(unittest.TestCase):

    def setUp(self):
        """Set up sample tensors for tests."""
        self.original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        self.reconstructed = torch.tensor([[1.1, 2.1], [3.1, 4.1]], dtype=torch.float32)
        
        # activations: batch_size=2, dict_size=3
        self.activations = torch.tensor([[-0.5, 0.0, 0.5], [0.1, -0.1, 0.0]], dtype=torch.float32)
        
        # dictionary_vectors: input_dim=5, dict_size=3
        self.dictionary_vectors = torch.randn(5, 3, dtype=torch.float32)

    def test_compute_reconstruction_loss(self):
        """Test various reconstruction loss types."""
        # MSE: ((0.1^2 + 0.1^2) + (0.1^2 + 0.1^2)) / 4 = (0.02 + 0.02) / 4 = 0.04 / 4 = 0.01
        mse_loss = compute_reconstruction_loss(self.original, self.reconstructed, loss_type='mse')
        self.assertAlmostEqual(mse_loss.item(), 0.01, places=5)

        # L1: ((0.1 + 0.1) + (0.1 + 0.1)) / 4 = 0.4 / 4 = 0.1
        l1_loss = compute_reconstruction_loss(self.original, self.reconstructed, loss_type='l1')
        self.assertAlmostEqual(l1_loss.item(), 0.1, places=5)

        # Huber (delta=1.0 by default in F.huber_loss)
        # For |x-y| <= delta, it's 0.5 * (x-y)^2. Here, all diffs are 0.1, so 0.5 * 0.1^2 = 0.005
        # Total Huber loss: (0.005 * 4) / 4 = 0.005
        huber_loss_actual = F.huber_loss(self.reconstructed, self.original, reduction='mean', delta=1.0)
        huber_loss_computed = compute_reconstruction_loss(self.original, self.reconstructed, loss_type='huber')
        self.assertAlmostEqual(huber_loss_computed.item(), huber_loss_actual.item(), places=5)
        self.assertAlmostEqual(huber_loss_computed.item(), 0.005, places=5)


        with self.assertRaisesRegex(ValueError, "Unsupported loss type: invalid_type"):
            compute_reconstruction_loss(self.original, self.reconstructed, loss_type='invalid_type')

    def test_compute_sparsity_loss(self):
        """Test various sparsity loss types."""
        sparsity_coef = 0.1
        
        # L1 sparsity: mean_abs_activations * coef
        # abs_activations = [[0.5, 0.0, 0.5], [0.1, 0.1, 0.0]]
        # sum_abs = 0.5 + 0.0 + 0.5 + 0.1 + 0.1 + 0.0 = 1.2
        # mean_abs = 1.2 / 6 = 0.2
        # loss = 0.2 * 0.1 = 0.02
        l1_sparsity = compute_sparsity_loss(self.activations, sparsity_coef=sparsity_coef, sparsity_type='l1')
        self.assertAlmostEqual(l1_sparsity.item(), 0.02, places=5)

        # L0_approx sparsity: just check if it runs and returns a positive value
        l0_approx_sparsity = compute_sparsity_loss(self.activations, sparsity_coef=sparsity_coef, sparsity_type='l0_approx')
        self.assertIsInstance(l0_approx_sparsity, torch.Tensor)
        self.assertGreater(l0_approx_sparsity.item(), 0)
        
        # Entropy sparsity: just check if it runs
        entropy_sparsity = compute_sparsity_loss(self.activations, sparsity_coef=sparsity_coef, sparsity_type='entropy')
        self.assertIsInstance(entropy_sparsity, torch.Tensor)
        # Entropy can be negative if we want low entropy (peaky distributions), and coef is positive.
        # The function implements -coef * mean(entropy), so if entropy is positive, loss is negative.
        # A uniform distribution has high entropy. A peaky one has low entropy.
        # F.softmax([[ -very_large, 0, +very_large ]]) -> [0, 0, 1] (low entropy)
        # F.softmax([[ 0.1, 0.1, 0.1 ]]) -> [1/3, 1/3, 1/3] (high entropy)
        # The current self.activations are not extremely peaky nor uniform.

        with self.assertRaisesRegex(ValueError, "Unsupported sparsity type: invalid_type"):
            compute_sparsity_loss(self.activations, sparsity_coef=sparsity_coef, sparsity_type='invalid_type')

    def test_compute_orthogonality_loss(self):
        """Test orthogonality loss computation."""
        ortho_coef = 0.01
        ortho_loss = compute_orthogonality_loss(self.dictionary_vectors, ortho_coef=ortho_coef)
        
        self.assertIsInstance(ortho_loss, torch.Tensor)
        self.assertEqual(ortho_loss.shape, torch.Size([])) # Should be a scalar

        # Test with perfectly orthogonal dictionary
        orthogonal_dict = torch.eye(5, 3) # Will not work as mm(3x5, 5x3) -> 3x3
        orthogonal_dict = torch.zeros(5,3)
        orthogonal_dict[0,0] = 1
        orthogonal_dict[1,1] = 1
        orthogonal_dict[2,2] = 1
        # Normalized, this should result in near zero loss if ortho_coef is small
        # or if the off-diagonal elements are zero
        # (A^T A - I)^2 where A has orthogonal columns.
        # A^T A will be diagonal (ideally I if columns are orthonormal)
        # So, A^T A - I will have zeros on diagonal, and small values off-diagonal if not perfectly orthogonal.
        
        # If dict_vectors are orthogonal, normalized_dict.t() @ normalized_dict should be Identity
        # gram_matrix = I, then (I - I)^2 = 0.
        # Let's test with an actual orthogonal (but not necessarily normal) matrix
        # For simplicity, let's use a small example.
        dict_simple_ortho = torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,0.0]], dtype=torch.float32) # 3x2
        ortho_loss_simple = compute_orthogonality_loss(dict_simple_ortho, ortho_coef=ortho_coef)
        # F.normalize(dict_simple_ortho, dim=0) -> [[1,0],[0,1],[0,0]]
        # gram = [[1,0],[0,1]]
        # identity = [[1,0],[0,1]]
        # gram - identity = [[0,0],[0,0]]
        # loss = ortho_coef * 0 = 0
        self.assertAlmostEqual(ortho_loss_simple.item(), 0.0, places=5)


    def test_compute_total_loss(self):
        """Test computation of total loss from components."""
        loss_config_basic = {
            'recon_loss_type': 'mse',
            'sparsity_coef': 0.1,
            'sparsity_type': 'l1',
            'use_orthogonality': False
        }
        
        losses_basic = compute_total_loss(self.original, self.reconstructed, self.activations, 
                                          self.dictionary_vectors, loss_config_basic)
        
        expected_recon_loss = compute_reconstruction_loss(self.original, self.reconstructed, 'mse')
        expected_sparsity_loss = compute_sparsity_loss(self.activations, 0.1, 'l1')
        expected_total_basic = expected_recon_loss + expected_sparsity_loss
        
        self.assertIn('reconstruction', losses_basic)
        self.assertIn('sparsity', losses_basic)
        self.assertNotIn('orthogonality', losses_basic)
        self.assertIn('total', losses_basic)
        
        self.assertAlmostEqual(losses_basic['reconstruction'].item(), expected_recon_loss.item(), places=5)
        self.assertAlmostEqual(losses_basic['sparsity'].item(), expected_sparsity_loss.item(), places=5)
        self.assertAlmostEqual(losses_basic['total'].item(), expected_total_basic.item(), places=5)

        loss_config_with_ortho = {
            'recon_loss_type': 'l1',
            'sparsity_coef': 0.05,
            'sparsity_type': 'l0_approx',
            'use_orthogonality': True,
            'ortho_coef': 0.01
        }
        
        losses_with_ortho = compute_total_loss(self.original, self.reconstructed, self.activations,
                                               self.dictionary_vectors, loss_config_with_ortho)
        
        expected_recon_l1 = compute_reconstruction_loss(self.original, self.reconstructed, 'l1')
        expected_sparsity_l0 = compute_sparsity_loss(self.activations, 0.05, 'l0_approx')
        expected_ortho = compute_orthogonality_loss(self.dictionary_vectors, 0.01)
        expected_total_with_ortho = expected_recon_l1 + expected_sparsity_l0 + expected_ortho
        
        self.assertIn('reconstruction', losses_with_ortho)
        self.assertIn('sparsity', losses_with_ortho)
        self.assertIn('orthogonality', losses_with_ortho)
        self.assertIn('total', losses_with_ortho)
        
        self.assertAlmostEqual(losses_with_ortho['reconstruction'].item(), expected_recon_l1.item(), places=5)
        self.assertAlmostEqual(losses_with_ortho['sparsity'].item(), expected_sparsity_l0.item(), places=5)
        self.assertAlmostEqual(losses_with_ortho['orthogonality'].item(), expected_ortho.item(), places=5)
        self.assertAlmostEqual(losses_with_ortho['total'].item(), expected_total_with_ortho.item(), places=5)

if __name__ == '__main__':
    unittest.main()
