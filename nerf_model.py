import torch
from torch import nn
import torch.nn.functional as F

class NeRF2D(nn.Module):
    """
    A 2D version of the Neural Radiance Fields (NeRF) model.
    
    This model takes in 2D coordinates (x, y) and predicts the RGB color at those points.
    """

    def __init__(self, encoder, input_dim, hidden_dim=256, num_layers=3):
        """
        Initializes the 2D NeRF model with customizable parameters.
        
        Args:
        input_dim (int): The number of input dimensions for the input (after the PE).
        hidden_dim (int): The number of hidden dimensions in the network.
        num_layers (int): The number of layers in the network.
        """
        super(NeRF2D, self).__init__()
        self.encoder = encoder  
        # Linear layer to transform input points from 2 dimensions to hidden_dim
        self.linear_pre = nn.Linear(input_dim, hidden_dim)
        
        # List of linear layers for the main 2D NeRF network
        self.net = nn.ModuleList()
        for i in range(num_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Linear layer to predict RGB color from the network output
        self.RGB = nn.Linear(hidden_dim, 3)

    def forward(self, input):
        """
        Forward pass through the 2D NeRF model.
        
        Args:
        input (torch.Tensor): Input 2D coordinates with shape [batch_size, input_dim]
        
        Returns:
        rgb (torch.Tensor): Predicted RGB colors with shape [batch_size, 3]
        """
        if self.encoder is not None:
            input = self.encoder(input)

        # Transform input points through the initial linear layer
        h = F.relu(self.linear_pre(input))
        
        # Pass through the main 2D NeRF network
        for layer in self.net:
            h = F.relu(layer(h))
        
        # Predict RGB color from the network output
        rgb = F.sigmoid(self.RGB(h))  # rgb has shape [batch_size, 3]
        
        return rgb


class NeRF(nn.Module):
    """
    A PyTorch implementation of the Neural Radiance Fields (NeRF) model.
    
    This model takes in 3D points and views to predict the RGB color and density (alpha) at those points.
    """

    def __init__(self, xyz_encoder, dir_encoder, input_dim, hidden_dim=128, num_layers=7, view_dim=24, output_dim=3):
        """
        Initializes the NeRF model with customizable parameters.
        
        Args:
        input_dim (int): The number of input dimensions for the input (after the PE).
        hidden_dim (int): The number of hidden dimensions in the network.
        num_layers (int): The number of layers in the network.
        view_dim (int): The number of input dimensions for the views.
        output_dim (int): The number of output dimensions for the RGB prediction.
        """
        super(NeRF, self).__init__()
        self.xyz_encoder = xyz_encoder
        self.dir_encoder = dir_encoder
        # Linear layer to transform input points from input_dim dimensions to hidden_dim
        self.linear_pre = nn.Linear(input_dim, hidden_dim)
        
        # List of linear layers for the main NeRF network
        self.net = nn.ModuleList()
        for i in range(num_layers):
            if i == 4:
                # At the 5th layer, concatenate input points back in
                self.net.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.net.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Linear layer to predict density (alpha) from the network output
        self.alpha = nn.Linear(hidden_dim, 1)
        
        # Linear layer to predict features from the network output
        self.feature = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear layer to transform the concatenated feature and view vectors
        self.view = nn.Linear(hidden_dim + view_dim, hidden_dim)
        
        # Linear layer to predict RGB color from the view features
        self.RGB = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, input_view):
        """
        Forward pass through the NeRF model.
        
        Args:
        input (torch.Tensor): Input points with shape [batch_size, input_dim]
        input_view (torch.Tensor): Input views with shape [batch_size, view_dim]
        
        Returns:
        rgb (torch.Tensor): Predicted RGB colors with shape [batch_size, output_dim]
        alpha (torch.Tensor): Predicted density (alpha) with shape [batch_size, 1]
        """
        # Transform input points through the initial linear layer
        input = self.xyz_encoder(input)
        h = F.relu(self.linear_pre(input))

        # Pass through the main NeRF network
        for i, layer in enumerate(self.net):
            h = F.relu(layer(h))
            # At the 4th layer, concatenate input points back in
            if i == 3:
                h = torch.cat([input, h], dim=-1)
        
        # Predict density (sigma) from the network output
        sigma = self.alpha(h)  # sigma has shape [batch_size, 1]
        
        # Predict features from the network output
        feature = self.feature(h)  # feature has shape [batch_size, hidden_dim]
        
        # Concatenate feature and view vectors
        input_view = self.dir_encoder(input_view)
        h = torch.cat([feature, input_view], dim=-1)  # h has shape [batch_size, hidden_dim + view_dim]
        
        # Transform concatenated vector through the view network
        h = F.relu(self.view(h))  # h has shape [batch_size, hidden_dim]
        
        # Predict RGB color from the view features
        rgb = F.sigmoid(self.RGB(h))  # rgb has shape [batch_size, output_dim]
        
        return rgb, sigma