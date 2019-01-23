import torch
from torch import nn, optim
from torch.nn import functional as F 

class CVAE_model(nn.Module):
    '''
    variational autoencoder class;
    
    parameters:
      - image_size: tuple;
        height and width of images;
      - input_channels: int;
        number of channels in input images;
      - num_conv_layers: int;
        number of encoding/decoding convolutional layers;
      - feature_map_list: list of ints;
        number of output feature maps for each convolutional layer;
      - filter_shape_list: list of tuples;
        convolutional filter shape for each convolutional layer;
      - stride_list: list of tuples;
        convolutional stride for each convolutional layer;
      - num_dense_layers: int;
        number of encoding/decoding dense layers;
      - dense_neuron_list: list of ints;
        number of neurons for each dense layer;
      - dense_dropout_list: list of float;
        fraction of neurons to drop in each dense layer (between 0 and 1);
      - latent_dim: int;
        number of dimensions for latent embedding;
      - activation: string (default='relu');
        activation function to use for layers;
      - eps_mean: float (default = 0.0);
        mean to use for epsilon (target distribution for embedding);
      - eps_std: float (default = 1.0);
        standard dev to use for epsilon (target distribution for embedding); 
    '''
    
    def __init__(self, image_size, input_channel, 
                num_conv_layers, feature_map_list, filter_shape_list, stride_list, 
                num_dense_layers, dense_neuron_list, dense_dropout_list, latent_dim,
                activation_func='relu',
                eps_mean=0.0,eps_std=1.0): 
        super(CVAE_model, self).__init__() 
        if len(feature_map_list) < num_conv_layers: 
            raise Exception("The length of filter_map_list list must be more than number of convolutional layers. ") 
        if len(filter_shape_list) < num_conv_layers: 
            raise Exception("The length of filter_shape_list list must be more than number of convolutional layers. ") 
        if len(stride_list) < num_conv_layers: 
            raise Exception("The length of stride_list list must be more than number of convolutional layers. ") 
        if len(dense_neuron_list) < num_dense_layers: 
            raise Exception("The length of dense_neuron_list list must be more than number of dense layers. ") 
        if len(dense_dropout_list) < num_dense_layers: 
            raise Exception("The length of dense_dropout_list list must be more than number of dense layers. ") 
        
        self.image_size = image_size 
        self.input_channel = input_channel 
        self.input_shape = [input_channel, image_size, image_size]
        mat_size = image_size * image_size * input_channel
        self.mat_shape = [input_channel, image_size, image_size]
        # Encoding layers
        ## Encoding convolutional layers 
        self.encode_conv_layers = [] 
        self.encode_conv_shape = []
        for i in range(num_conv_layers): 
            if i == 0: 
                in_channel = self.input_channel
            else: 
                in_channel = feature_map_list[i-1]
            conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=feature_map_list[i], padding=filter_shape_list[i]//2, 
                                   kernel_size=filter_shape_list[i], stride=stride_list[i])
            self.encode_conv_layers.append(conv_layer) 
#             self.encode_conv_shape.append(self.mat_shape)
            mat_size /= stride_list[i] ** 2 
            self.mat_shape[0] = feature_map_list[i] 
            self.mat_shape[1] /= stride_list[i] 
            self.mat_shape[2] /= stride_list[i]             
            ## Padding shoule be half of the kernal size, which should always be an odd number 
        self.encode_conv_layers = nn.ModuleList(self.encode_conv_layers)
            
        ## Encoding dense layers 
        mat_size *= feature_map_list[-1] 
        self.conv_dense_dim = mat_size
        self.encode_dense_layers = [] 
        self.encode_dropout_layers = []
        for i in range(num_dense_layers): 
            if i == 0: 
                in_features = mat_size
            else: 
                in_features = dense_neuron_list[i-1]
            dense_layer = nn.Linear(in_features, dense_neuron_list[i]) 
            self.encode_dense_layers.append(dense_layer) 
            self.encode_dropout_layers.append(nn.Dropout2d(p=dense_dropout_list[i]))
        self.encode_dense_layers = nn.ModuleList(self.encode_dense_layers)
        self.encode_dropout_layers = nn.ModuleList(self.encode_dropout_layers)
        
        ## Encoding layer for mu and logvar 
        self.latent_mu = nn.Linear(dense_neuron_list[-1], latent_dim) 
        self.latent_logvar = nn.Linear(dense_neuron_list[-1], latent_dim) 
        
        # Decoding layers 
        ## Latent to dense 
        self.latent_decode = nn.Linear(latent_dim, dense_neuron_list[-1])
        ## Decoding dense layers 
        self.decode_dense_layers = [] 
        for i in range(num_dense_layers): 
            in_feature = dense_neuron_list[-1-i] 
            if i == num_dense_layers - 1: 
                out_features = mat_size 
            else: 
                out_features = dense_neuron_list[-1-i-1] 
            dense_layer = nn.Linear(in_feature, out_features) 
            self.decode_dense_layers.append(dense_layer) 
        self.decode_dense_layers = nn.ModuleList(self.decode_dense_layers)
            
        ## Decoding convolutional layers 
        self.decode_conv_layers = []
        for i in range(num_conv_layers): 
            in_channel = feature_map_list[-1-i]
            if i == num_conv_layers - 1: 
                out_channel = self.input_channel
            else: 
                out_channel = feature_map_list[-1-i-1] 
            conv_layer = nn.ConvTranspose2d(in_channels=feature_map_list[-1-i], out_channels=out_channel, padding=filter_shape_list[-1-i]//2, 
                                            kernel_size=filter_shape_list[-1-i], stride=stride_list[-1-i], 
                                            output_padding=stride_list[-1-i]+filter_shape_list[-1-i]//2*2-filter_shape_list[-1-i]) 
            self.decode_conv_layers.append(conv_layer)
        self.decode_conv_layers = nn.ModuleList(self.decode_conv_layers)
            
            
    def encode(self, x): 
        for conv_layer in self.encode_conv_layers: 
            x = F.relu(conv_layer(x))            
        x = x.view(-1, self.conv_dense_dim) 
        for dense_layer, dropput_layer in zip(self.encode_dense_layers, self.encode_dropout_layers): 
            x = F.relu(dense_layer(dropput_layer(x)))      
        mu = self.latent_mu(x) 
        logvar = self.latent_logvar(x)
        return mu, logvar 
    
    def reparameterize(self, mu, logvar): 
        # eps * exp(-0.5 * logvar) * mu 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z): 
        z = self.latent_decode(z)
        for dense_layer in self.decode_dense_layers: 
            z = F.relu(dense_layer(z)) 
        z = z.view([-1,] + self.mat_shape)
        for i, conv_layer in enumerate(self.decode_conv_layers): 
            if i == len(self.decode_conv_layers) - 1: 
                z = torch.sigmoid(self.decode_conv_layers[-1](z))
            else: 
                z = F.relu(conv_layer(z)) 
        return z 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar 