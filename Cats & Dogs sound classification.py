#This code is based on dataset from Kaggle
#Please download Cats&Dogs audio dataset and run this code

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(999)
if device == 'cuda':
    torch.cuda.manual_seed_all(999)
print(device + " is available")

#Importing essential packages
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TT
import matplotlib.pyplot as plt
import os, math, random, glob, librosa, torchaudio
import torch.utils.data
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm
import torchaudio.transforms as T

'''
Data Loading part
'''

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        # mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)#Newly added

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))


    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist():
    final_train, final_test = [], []
    trains, tests = [], []
    train_p = "Train_file path"
    test_p = "Test_file path"
    for d in os.listdir(train_p):
      assert d.endswith(".wav") == True
      trains.append((train_p+d, librosa.get_duration(path=train_p+d)))

    for d in os.listdir(test_p):
      assert d.endswith(".wav") == True
      tests.append((test_p+d, librosa.get_duration(path=test_p+d)))

    #sort by size of wav file
    trains.sort(key=lambda x:x[1], reverse=True)
    maximum_length = trains[0][1]


    random.shuffle(trains)
    random.shuffle(tests)
    for f in trains:
      final_train.append(f[0])
    for f in tests:
      final_test.append(f[0])

    #Calculate data ratio
    dog_nums, cat_nums = 0,0
    for t in final_train:
      t = t.split("/")
      if "c" == t[5][0]:
        cat_nums += 1
      else:
        dog_nums += 1
    del trains, tests
    return final_train, final_test, cat_nums, dog_nums, maximum_length



class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax,
                 shuffle=True,
                 device=None, maximum_length=None):

        self.audio_files = training_files

        if shuffle:
            random.seed(1234)
            random.shuffle(self.audio_files)

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.device = device

    # # of data
    def __len__(self):
        return len(self.audio_files)



    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename)
        # print('filenmae = ', filename, audio)

        if sampling_rate != self.sampling_rate:
            #Resampling the wave file should be done!
            audio = torchaudio.resample(audio, sampling_rate, 16000)


        #Normalizing audio
        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        #audio frame length = sampling_rate * duration
        if audio.shape[-1] < int(sampling_rate*maximum_length):
          temp_a = torch.zeros(audio.shape[0], int(sampling_rate*maximum_length))
          temp_a[:, :audio.shape[-1]] = audio
          audio = temp_a
          del temp_a

        elif audio.shape[-1] > int(sampling_rate*maximum_length):
          audio = audio[:, :int(sampling_rate*maximum_length)]


        #Mel-spectrogram generation
        mel_spec = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size,
                                  self.win_size, self.fmin, self.fmax,
                                  center=False)


        #Manual labeling : Cat(1), Dog(0)
        filename = filename.split("/")
        filename = filename[5][0].lower()
        if "cat" == filename:
          label = torch.LongTensor([0]) 
        else:
          label = torch.LongTensor([1])

        return (mel_spec, audio.squeeze(1), label, self.audio_files[index])




#Network Declaration --> Super Simple(Just for testing)
#Input shape [batch, dim, length](mel-spectrogram)
class AudioClassification(torch.nn.Module):
  def __init__(self):
    super(AudioClassification, self).__init__()
    self.process_1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 10, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(10),#BatchNormalization
        torch.nn.Conv2d(10, 40, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(40),
        torch.nn.Dropout2d(p=0.8))

    self.process_2 = torch.nn.Sequential(
        torch.nn.Conv2d(40, 20, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(20),
        torch.nn.Conv2d(20, 10, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(10),
        torch.nn.Dropout2d(p=0.8)
    )
    self.input_bn = torch.nn.BatchNorm2d(1)
    self.first_fc = torch.nn.Linear(650, 300)
    self.final_fc2 = torch.nn.Linear(300, 1)

  def forward(self, input_data):
    output = self.input_bn(input_data)
    output = self.process_1(input_data)
    output = self.process_2(output)
    output = torch.flatten(output, start_dim=1)
    output = self.first_fc(output)
    output = self.final_fc2(output)
    return output





batch_size = ? #Setting number whatever you want!
Epoch = ?
train_files, test_files, cat_nums, dog_nums, maximum_length = get_dataset_filelist()

'''
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
'''
train_files = MelDataset(train_files, n_fft=1024, num_mels=80,
                 hop_size=256, win_size=1024, sampling_rate=16000,
                 fmin=0, fmax=8000, shuffle=True,
                 device=device, maximum_length=maximum_length)

test_files = MelDataset(test_files, n_fft=1024, num_mels=80,
                 hop_size=256, win_size=1024, sampling_rate=16000,
                 fmin=0, fmax=8000, shuffle=True,
                 device=device, maximum_length=maximum_length)


train_loader = DataLoader(dataset=train_files, num_workers=2,
                          batch_size=batch_size,
                          shuffle=False, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_files, num_workers=2,
                         batch_size=1,
                         shuffle=False, pin_memory=True, drop_last=True)



#Declaring network
model = AudioClassification()
model = model.to(device)

#Due to class imbalance
assert cat_nums == max(cat_nums, dog_nums)
class_weight = torch.tensor([cat_nums / dog_nums]).to(device)

#Weighted Bianry Crossentropy Loss
loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for e in range(Epoch):
  epoch_loss = 0
  print('Epoch : {:d}'.format(e))
  for _, (mel, _, labels, filename) in enumerate(tqdm(train_loader)):
    #Training begins
    mel, labels = mel.to(device), labels.to(device)

    optimizer.zero_grad()
    pred = model(mel)
    assert labels.shape == pred.shape
    loss = loss_func(pred,labels)
    epoch_loss += loss
    loss.backward() #backpropagation
    optimizer.step()

  print('Training Loss : {:4.3f}'.format(epoch_loss/len(train_loader)))

print('Training process has been done!')




#Evaluation Process (previous one)
correct_numbers, eval_numbers = 0, 0
sliced_mel_length = None
model.eval()
with torch.no_grad():
  for _, (mel, audio, labels, _) in enumerate(tqdm(test_loader)):
    eval_numbers += len(labels)
    mel, labels = mel.to(device), labels.to(device)
    sliced_mel_length = mel.shape[-1]
    pred = model(mel)
    preds = torch.zeros_like(pred)
    for i, p in enumerate(pred):
      if p > 0.5:
        #dog
        preds[i] = 1.0
      else:
        #cat
        preds[i] = 0.0
    correct_numbers += 1 if preds[0] == labels[0] else 0

  #Print evaluation result
  print('Total Accuracy of Evaluation : {:4.3f}'.format(100 * (correct_numbers / eval_numbers)))


