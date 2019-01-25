import torch
from torch import nn, optim
from torch.nn import functional as F 


class conv_variational_autoencoder(object): 
    '''
    variational autoencoder class 

    parameters:
      - image_size: tuple
        height and width of images
      - input_channels: int
        number of channels in input images
      - num_conv_layers: int 
        number of encoding/decoding convolutional layers
      - feature_map_list: list of ints
        number of output feature maps for each convolutional layer
      - filter_shape_list: list of tuples;
        convolutional filter shape for each convolutional layer
      - stride_list: list of tuples
        convolutional stride for each convolutional layer
      - num_dense_layers: int
        number of encoding/decoding dense layers
      - dense_neuron_list: list of ints
        number of neurons for each dense layer
      - dense_dropout_list: list of float
        fraction of neurons to drop in each dense layer (between 0 and 1)
      - latent_dim: int
        number of dimensions for latent embedding
      - activation: string (default='relu') 
        activation function to use for layers
      - eps_mean: float (default = 0.0) 
        mean to use for epsilon (target distribution for embedding);
      - eps_std: float (default = 1.0) 
        standard dev to use for epsilon (target distribution for embedding)
       
    methods:
      - train(data,batch_size,epochs=1,checkpoint=False,filepath=None)
        train network on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
      - return_embeddings(data)
        return the embeddings for given data
      - decode(embedding)
        return a generated output given a latent embedding 
      - predict(data) 
        return the predicted result of data
    '''
    def __init__(self, image_size, input_channel, 
                num_conv_layers, feature_map_list, filter_shape_list, stride_list, 
                num_dense_layers, dense_neuron_list, dense_dropout_list, latent_dim,
                activation_func='relu',
                eps_mean=0.0,eps_std=1.0):   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CVAE_model(image_size, input_channel, 
                          num_conv_layers, feature_map_list, filter_shape_list, stride_list, 
                          num_dense_layers, dense_neuron_list, dense_dropout_list, latent_dim, 
                          activation_func='relu', eps_mean=0.0,eps_std=1.0).to(self.device)
#         self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, alpha=0.9, eps=1e-08, weight_decay=1e-6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
#         self.optimizer = optim.Adadelta(self.model.parameters(), lr=1e-1)
        self.loss_func = nn.BCELoss(reduction='sum') 
        
    def encode(self, x): 
        return self.model.encode(x) 
    
    def reparameterize(self, mu, logvar): 
        return self.model.reparameterize(mu, logvar)
    
    def decode(self, z): 
        return self.model.decode(z) 
    
    def return_embeddings(self, x): 
        return self.model.reparameterize(*(self.model.encode(x))) 
    
    def predict(self, x): 
        return self.model(x)[0] 
    
    def loss(self, recon_x, x, mu, logvar):  
        BCE = self.loss_func(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
        return BCE + KLD 
    
    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
    #         print(recon_batch.shape, data.shape)
            loss = self.loss(recon_batch, data, mu, logvar) 
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
#             if batch_idx % log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader),
#                     loss.item() / len(data)))
        print '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)),
    
        
    def test(self, test_loader, epoch):
        self.model.train()
        test_loss = 0 
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device) 
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss(recon_batch, data, mu, logvar).item()
#                 if i == 0:
#                     n = min(data.size(0), 8)
#                     comparison = torch.cat([data[:n],
#                                           recon_batch.view(batch_size, 1, 28, 28)[:n]])
#                     save_image(comparison.cpu(),
#                              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        

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
        self.eps_mean = eps_mean
        self.eps_std = eps_std
        if len(feature_map_list) != num_conv_layers: 
            raise Exception("The length of filter_map_list list must equal the number of convolutional layers. ") 
        if len(filter_shape_list) != num_conv_layers: 
            raise Exception("The length of filter_shape_list list must equal the number of convolutional layers. ") 
        if len(stride_list) != num_conv_layers: 
            raise Exception("The length of stride_list list must equal the number of convolutional layers. ") 
        if len(dense_neuron_list) != num_dense_layers: 
            raise Exception("The length of dense_neuron_list list must equal the number of dense layers. ") 
        if len(dense_dropout_list) != num_dense_layers: 
            raise Exception("The length of dense_dropout_list list must equal the number of dense layers. ") 
        
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
        if feature_map_list: 
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
        # eps * exp(-0.5 * logvar) + mu 
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