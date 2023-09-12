#This code is mainly based on Pytorch Official Code.
#Syntax based structure modification has been done.
#Input data : MNIST dataset
class VAE_Model(nn.Module):
  def __init__(self):
    super(VAE_Model, self).__init__()

    h_dims = [32, 64, 128, 256, 512]
    inner_modules = []
    id = 1

    for hd in h_dims:
      inner_modules.append(
          torch.nn.Sequential(
              nn.Conv2d(in_channels=id, out_channels=hd,
                        kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(hd)
              )
          )
      id = hd

    self.encoder = nn.Sequential(*inner_modules)
    self.latent_dim = h_dims[2]
    self.mu = nn.Linear(h_dims[-1], self.latent_dim)
    self.var = nn.Linear(h_dims[-1],self.latent_dim)
    self.pre_dim = nn.Linear(self.latent_dim, h_dims[-1])

    #Transposed Convolution has to be done
    modules = []
    h_dims.reverse() #we have to go back through!
    for i in range(len(h_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(h_dims[i],
                                    h_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(h_dims[i + 1]))
        )
    self.decoder = nn.Sequential(*modules)



    self.post_layer = nn.Sequential(
                            nn.ConvTranspose2d(h_dims[-1],
                                               h_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(h_dims[-1], out_channels= 1,
                                      kernel_size= 7, stride=1, padding= 1),
                            nn.Tanh(),
                            nn.BatchNorm2d(1))

  def forward(self, x):
    #Encode
    #input_x = torch.Size([4, 1, 28, 28])
    x = self.encoder(x)
    x = torch.flatten(x, start_dim=1)
    x_mu = self.mu(x)
    x_var = self.var(x)
    z = self.z_reparameterization(x_mu, x_var) #[batch, dimension]

    #Decode
    output = self.pre_dim(z)
    output = output.reshape(-1, 512, 1, 1)
    output = self.decoder(output)
    output = self.post_layer(output)
    return output, x_mu, x_var

  #Offical code based
  def z_reparameterization(self, mu, var):
    std = torch.exp(0.5 * var)
    epsilon = torch.randn_like(std)
    z = mu + (std * epsilon)
    return z


