"""
ResNet Feature Extractor for extracting intermediate layer representations.
Specifically designed to extract features from the penultimate layer.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional


class ResNetFeatureExtractor(nn.Module):
    """
    Feature extractor for ResNet models that can extract features from any intermediate layer.
    """
    
    def __init__(self, model_name='resnet50', layer_name='avgpool', pretrained=True, freeze=True):
        """
        Initialize ResNet feature extractor.
        
        Args:
            model_name (str): Name of ResNet model ('resnet18', 'resnet50', etc.)
            layer_name (str): Name of layer to extract features from
            pretrained (bool): Whether to use pretrained weights
            freeze (bool): Whether to freeze the ResNet parameters
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.layer_name = layer_name
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Load ResNet model
        self.resnet = self._load_resnet_model()
        
        # Setup feature extraction hooks
        self.features = {}
        self.hooks = []
        self._register_hooks()
        
        # Freeze parameters if specified
        if freeze:
            self._freeze_parameters()
    
    def _load_resnet_model(self):
        """Load the specified ResNet model."""
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if self.model_name not in model_dict:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model_dict[self.model_name](pretrained=self.pretrained)
        return model
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        def hook_fn(name):
            def hook(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    output = output[0]
                
                # Flatten spatial dimensions if needed (for conv layers)
                if len(output.shape) == 4:  # [batch, channels, height, width]
                    output = torch.flatten(output, start_dim=2)  # [batch, channels, h*w]
                    output = output.mean(dim=2)  # Global average pooling -> [batch, channels]
                elif len(output.shape) == 2:  # [batch, features]
                    pass  # Already in correct format
                else:
                    output = output.view(output.size(0), -1)  # Flatten to [batch, features]
                
                self.features[name] = output.detach()
            return hook
        
        # Register hook for the specified layer
        target_layer = self._get_layer_by_name(self.layer_name)
        if target_layer is not None:
            hook = target_layer.register_forward_hook(hook_fn(self.layer_name))
            self.hooks.append(hook)
        else:
            raise ValueError(f"Layer '{self.layer_name}' not found in {self.model_name}")
    
    def _get_layer_by_name(self, layer_name):
        """Get layer by name from the ResNet model."""
        # Common layer names for ResNet
        layer_mapping = {
            'conv1': self.resnet.conv1,
            'bn1': self.resnet.bn1,
            'relu': self.resnet.relu,
            'maxpool': self.resnet.maxpool,
            'layer1': self.resnet.layer1,
            'layer2': self.resnet.layer2,
            'layer3': self.resnet.layer3,
            'layer4': self.resnet.layer4,
            'avgpool': self.resnet.avgpool,
            'fc': self.resnet.fc
        }
        
        return layer_mapping.get(layer_name)
    
    def _freeze_parameters(self):
        """Freeze all parameters in the ResNet model."""
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass to extract features.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, feature_dim]
        """
        # Clear previous features
        self.features.clear()
        
        # Forward pass through ResNet
        _ = self.resnet(x)
        
        # Return features from the specified layer
        if self.layer_name in self.features:
            features = self.features[self.layer_name]
            
            # Ensure features are properly shaped
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            return features
        else:
            raise RuntimeError(f"Features not found for layer '{self.layer_name}'")
    
    def get_feature_dim(self):
        """Get the dimension of extracted features."""
        # Create dummy input to determine feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        if next(self.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            features = self.forward(dummy_input)
        
        return features.shape[1]
    
    def get_available_layers(self):
        """Get list of available layer names."""
        return ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 
                'layer3', 'layer4', 'avgpool', 'fc']
    
    def set_layer(self, layer_name):
        """Change the target layer for feature extraction."""
        # Remove existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Set new layer name and register hooks
        self.layer_name = layer_name
        self._register_hooks()
    
    def _register_hook(self):
        """Register forward hook to capture intermediate features."""
        def hook_fn(module, input, output):
            # Handle different layer outputs
            if isinstance(output, torch.Tensor):
                if len(output.shape) == 4:  # Convolutional layer output [B, C, H, W]
                    # Apply global average pooling to spatial dimensions
                    self.features = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
                else:  # Already flattened (e.g., from avgpool or fc layers)
                    self.features = output.squeeze() if output.dim() > 2 else output
            else:
                # Handle tuple outputs (some layers might return tuples)
                self.features = output[0] if isinstance(output, (tuple, list)) else output
        
        # Find the target layer and register hook
        target_module = self._get_layer_by_name(self.layer_name)
        if target_module is not None:
            target_module.register_forward_hook(hook_fn)
            print(f"Registered hook for layer: {self.layer_name}")
        else:
            raise ValueError(f"Layer '{self.layer_name}' not found in {self.model_name}")
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        for hook in self.hooks:
            hook.remove()


class MultiLayerResNetExtractor(nn.Module):
    """
    Extract features from multiple layers simultaneously.
    """
    
    def __init__(self, model_name='resnet50', layer_names=None, pretrained=True, freeze=True):
        """
        Initialize multi-layer extractor.
        
        Args:
            model_name (str): Name of ResNet model
            layer_names (list): List of layer names to extract features from
            pretrained (bool): Whether to use pretrained weights
            freeze (bool): Whether to freeze parameters
        """
        super(MultiLayerResNetExtractor, self).__init__()
        
        if layer_names is None:
            layer_names = ['layer3', 'layer4', 'avgpool']
        
        self.layer_names = layer_names
        self.extractors = nn.ModuleDict()
        
        # Create separate extractors for each layer
        for layer_name in layer_names:
            self.extractors[layer_name] = ResNetFeatureExtractor(
                model_name=model_name,
                layer_name=layer_name,
                pretrained=pretrained,
                freeze=freeze
            )
    
    def forward(self, x):
        """
        Extract features from multiple layers.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            dict: Dictionary mapping layer names to features
        """
        features = {}
        for layer_name, extractor in self.extractors.items():
            features[layer_name] = extractor(x)
        
        return features
    
    def get_feature_dims(self):
        """Get feature dimensions for all layers."""
        dims = {}
        for layer_name, extractor in self.extractors.items():
            dims[layer_name] = extractor.get_feature_dim()
        return dims


# Convenience function for getting penultimate layer features
def get_resnet50_penultimate_extractor(pretrained=True, freeze=True):
    """
    Get ResNet50 feature extractor for penultimate layer (before final FC).
    For ResNet50, this is the avgpool layer output (2048 dimensions).
    """
    return ResNetFeatureExtractor(
        model_name='resnet50',
        layer_name='avgpool',  # This gives us the penultimate layer
        pretrained=pretrained,
        freeze=freeze
    )