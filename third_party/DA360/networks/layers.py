import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from typing import List, Optional, Union, Dict, Any

class CirPad2d(nn.Module):
    def __init__(self, pad):
        super(CirPad2d, self).__init__()
        self.pad = pad

    def forward(self, x):

        w = x.shape[-1]

        pad_u = torch.flip(torch.roll(x[:, : , :self.pad, :], w // 2, -1), dims=[-2])
        pad_d = torch.flip(torch.roll(x[:, : , -self.pad:, :], w // 2, -1), dims=[-2])

        x = torch.cat([pad_u, x, pad_d], dim=-2)

        left = x[:, :, :, :self.pad]
        right = x[:, :, :, -self.pad:]
        x = torch.cat([right, x, left], dim=-1)

        return x

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, zero_padding=False, bias=True):
        super(Conv3x3, self).__init__()

        if zero_padding:
            self.pad = nn.ZeroPad2d(1)
        else:
            self.pad = CirPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class MultiLayerMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron with configurable depth, activation functions, and normalization.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dims (List[int]): List of hidden layer dimensions. Length determines network depth.
        output_dim (int): Dimension of output features
        activation (str): Activation function name 
        dropout (float): Dropout rate for regularization
        use_batch_norm (bool): Whether to use batch normalization
        use_layer_norm (bool): Whether to use layer normalization as alternative to batch norm
        init_method (str): Weight initialization method ('kaiming', 'xavier', 'normal')
        output_activation (str): Activation function for output layer (None for linear)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 init_method: str = 'kaiming',
                 output_activation: Optional[str] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.depth = len(hidden_dims)
        
        # Validate normalization choices
        if use_batch_norm and use_layer_norm:
            raise ValueError("Cannot use both batch norm and layer norm simultaneously")
        
        # Get activation functions
        self.activation_fn = self._get_activation(activation)
        self.output_activation_fn = self._get_activation(output_activation) if output_activation else None
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if (use_batch_norm or use_layer_norm) else None
        
        # Input layer
        prev_dim = input_dim
        
        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization layer
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights(init_method)

    def _get_activation(self, activation: Optional[str]) -> nn.Module:
        """
        Get activation function module.
        
        Args:
            activation (str): Name of activation function
            
        Returns:
            nn.Module: Activation function module
        """
        if activation is None:
            return nn.Identity()
        
        activation_dict = {
            'relu': nn.ReLU(inplace=True),
            'elu': nn.ELU(alpha=1.0, inplace=True),
            'softplus': nn.Softplus(),
        }
        
        return activation_dict.get(activation.lower(), nn.ReLU(inplace=True))

    def _initialize_weights(self, init_method: str):
        """
        Initialize network weights.
        
        Args:
            init_method (str): Weight initialization method
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                elif init_method == 'normal':
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        
        # Pass through hidden layers
        for i in range(self.depth):
            # Linear transformation
            x = self.layers[i](x)
            
            # Normalization
            if self.norms is not None:
                x = self.norms[i](x)
            
            # Activation and dropout
            x = self.activation_fn(x)
            x = self.dropout(x)
                    
        # Output layer
        x = self.output_layer(x)
        
        # Output activation (if specified)
        if self.output_activation_fn:
            x = self.output_activation_fn(x)
        
        # Squeeze for single-output regression
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving/recreating."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': str(self.activation_fn),
            'dropout': self.dropout.p,
            'use_batch_norm': self.norms is not None and isinstance(self.norms[0], nn.BatchNorm1d),
            'use_layer_norm': self.norms is not None and isinstance(self.norms[0], nn.LayerNorm),
            'depth': self.depth
        }




class ERPCircularConv2d(nn.Module):
    """
    ERP image convolution with circular padding
    Horizontal: circular padding (continuous 360°)
    Vertical: special padding considering pole adjacency
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, **kwargs):
        super().__init__()
        
        # Store original padding value
        self._original_padding = _pair(padding)
        
        # Create internal Conv2d with padding=0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        
        # Expose weight and bias for compatibility
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        
        # Store other attributes for compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = self.conv.stride
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        self.padding_mode = self.conv.padding_mode
    
    def forward(self, x):
        # Only apply custom padding if needed
        if not any(p != 0 for p in self._original_padding):
            return self.conv(x)
            
        pad_h, pad_w = self._original_padding
        b, c, h, w = x.shape
        
        # Create base tensor with zero padding
        y = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        
        # Special vertical padding for ERP images
        if pad_h > 0:
            # Top padding: use bottom region, roll horizontally by half width and flip
            top_fill = torch.flip(torch.roll(x[:, :, :pad_h, :], w // 2, -1), dims=[-2])
            y[:, :, :pad_h, pad_w:pad_w+w] = top_fill
            
            # Bottom padding: use top region, roll horizontally by half width and flip  
            bottom_fill = torch.flip(torch.roll(x[:, :, -pad_h:, :], w // 2, -1), dims=[-2])
            y[:, :, -pad_h:, pad_w:pad_w+w] = bottom_fill
        
        # Horizontal circular padding
        if pad_w > 0:
            # Left: use right region
            y[:, :, :, :pad_w] = y[:, :, :, -2*pad_w:-pad_w]
            # Right: use left region
            y[:, :, :, -pad_w:] = y[:, :, :, pad_w:2*pad_w]
        
        # Apply convolution with internal conv layer
        return self.conv(y)
    
    @property
    def padding(self):
        """Return original padding for compatibility"""
        if self._original_padding[0] == self._original_padding[1]:
            return self._original_padding[0]
        return self._original_padding
    
    def extra_repr(self):
        """String representation that shows actual padding"""
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
                f"groups={self.groups}, bias={self.bias is not None}, "
                f"padding_mode={self.padding_mode}")


def modify_conv_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            conv_args = {
                'in_channels': child.in_channels,
                'out_channels': child.out_channels,
                'kernel_size': child.kernel_size,
                'stride': child.stride,
                'padding': child.padding,
                'dilation': child.dilation,
                'groups': child.groups,
                'bias': child.bias is not None,
                'padding_mode': child.padding_mode
            }
            
            custom_conv = ERPCircularConv2d(**conv_args)
            custom_conv.weight.data = child.weight.data.clone()
            if child.bias is not None:
                custom_conv.bias.data = child.bias.data.clone()
            setattr(module, name, custom_conv)

