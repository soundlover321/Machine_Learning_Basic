#MNIST dataset based simple AutoEncoder test

#Importing essential packages
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as TT
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(999)
if device == 'cuda':
    torch.cuda.manual_seed_all(999)
print(device + " is available")






'''
Data Loading part
'''
import os, random, struct

def read(dataset = None, path = None):
  if dataset == "training":
      fname_img = os.path.join(path, 'train_images.idx3-ubyte')
      fname_lbl = os.path.join(path, 'train_labels.idx1-ubyte')
  elif dataset == "testing":
      fname_img = os.path.join(path, 'test_images.idx3-ubyte')
      fname_lbl = os.path.join(path, 'test_labels.idx1-ubyte')
  else:
    print('Error')
    exit(-1)

  # Load everything in some numpy arrays
  with open(fname_lbl, 'rb') as flbl:
      magic, num = struct.unpack(">II", flbl.read(8))
      lbl = np.fromfile(flbl, dtype=np.int8)

  with open(fname_img, 'rb') as fimg:
      magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
      img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


  get_img = lambda idx: (img[idx], lbl[idx])

  # Create an iterator which returns each image in turn
  for i in range(len(lbl)):
      yield get_img(i)

class MNIST_Datasets(torch.utils.data.Dataset):
    def __init__(self, flag):
      self.data_path = "Data_Path"
      self.flag = flag
      if flag == "training":
        self.data = list(read(dataset="training", path=self.data_path))
      else:
        self.data = list(read(dataset="testing", path=self.data_path))

    def __getitem__(self, index):

        image, label = self.data[index]
        image, label = torch.tensor(image), torch.tensor(label)

        return (image, label.long())

    # # of data
    def __len__(self):
        return len(self.data)





batch_size = ? #Setting number whatever you want!
Epoch = ?
train_d = MNIST_Datasets(flag="training")
test_d = MNIST_Datasets(flag="testing")
train_loader = DataLoader(dataset=train_d,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_d,
                         batch_size=1,
                         shuffle=False)




#Super simple AutoEncoder
class AutoEncoder(torch.nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()

    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(64),
        torch.nn.Linear(64, 12),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(12),
        torch.nn.Linear(12, 3)
    )

    self.decoder = torch.nn.Sequential(
        torch.nn.Linear(3, 12),
        torch.nn.Linear(12, 64),
        torch.nn.Linear(64, 128),
        torch.nn.Linear(128, 784),
        torch.nn.Sigmoid()
    )


  def forward(self, x):
    x = x.reshape(batch_size, -1)
    x = self.encoder(x)
    x = self.decoder(x)
    x = x.reshape(batch_size, 28, 28)
    x = x.unsqueeze(1)
    return x





autoencoder_model = AutoEncoder()
autoencoder_model.to(device)

#Loss function for classification
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=0.001)


#Training Process
print('Training Start~')
for e in range(Epoch):
  epoch_loss = 0
  print('Epoch : {:d}'.format(e))
  for i, (data, _) in enumerate(tqdm(train_loader)):
    #data = [batch, channel, width, height]
    data = data.to(device)
    optimizer.zero_grad()
    pred = autoencoder_model(data)
    assert pred.shape == data.shape
    loss = F.mse_loss(pred, data) * 45
    epoch_loss += loss
    loss.backward() #backpropagation
    optimizer.step()

  print('Training Loss : {:4.3f}'.format(epoch_loss/len(train_loader)))
print('Training process has been done!')



#Visualization & Evaluation Process
autoencoder_model.eval()
with torch.no_grad():
  recon_loss = 0
  for i, (data,_) in enumerate(tqdm(test_loader)):
    data = data.to(device)
    pred = autoencoder_model(data)
    data, pred = data.squeeze(1), pred.squeeze(1)
    data, pred = data.cpu().detach().numpy(), pred.cpu().detach().numpy()

    #Visualizing results (data(ground-truth), predicted result)

    if i % 100 == 0:
      plt.subplot(2, 2, 1)
      plt.imshow(data[0], cmap="gray")
      plt.gca().set_title("Ground-Truth")
      plt.subplot(2, 2, 2)
      plt.imshow(pred[0], cmap="gray")
      plt.gca().set_title("Generated")
      plt.show()

print('Test reconstruction loss = ', (recon_loss / len(test_loader)))
