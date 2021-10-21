import os
import sys
import requests
import shutil
import tarfile

import numpy as np

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from utils import check_path


CIFAR10_PATH = '../data/CIFAR10'
CIFAR10C_PATH = '../data'

CIFAR10C_URL = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'

check_path(CIFAR10_PATH)
check_path(CIFAR10C_PATH)


# Download CIFAR10-C
target_path = os.path.join(CIFAR10C_PATH, 'CIFAR-10-C.tar')
print('Downloading CIFAR10-C...')
with open(target_path, 'wb') as f:
    with requests.get(CIFAR10C_URL, allow_redirects=True, stream=True) as resp:
        total_length = resp.headers.get('content-length')
        dl = 0
        total_length = int(total_length)
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                dl += len(chunk)
                f.write(chunk)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[{:s}{:s}{:s}] {:.2f}%".format('=' * done, '>', ' ' * (50-done), (dl/total_length)*100) )    
                sys.stdout.flush()

# Extract
print('Extracting CIFAR10-C...')
tarf = tarfile.open(target_path, 'r')
tarf.extractall(CIFAR10C_PATH)
tarf.close()
os.remove(target_path)
print('Done')

transforms_data=transforms.Compose([
    transforms.ToTensor(),
])

train = CIFAR10(CIFAR10_PATH, train=True, download=True, transform=transforms_data)
test = CIFAR10(CIFAR10_PATH, train=False, transform=transforms_data)




# Divide into validation and train splits
N = len(train)

np.random.seed(10)
rand_ix = np.random.permutation(N)

ix_train, ix_val = rand_ix[:45000], rand_ix[45000:]

train_data = data.Subset(train, ix_train)
val_data = data.Subset(train, ix_val)

train = data.DataLoader(train_data, batch_size=45000)
val = data.DataLoader(val_data, batch_size=5000)
test = data.DataLoader(test, batch_size=10000)

X_train, y_train = next(iter(train))
X_val, y_val = next(iter(val))
X_test, y_test = next(iter(test))

X_train, y_train = (255*X_train.numpy()).astype(np.uint8), y_train.numpy()
X_val, y_val = (255*X_val.numpy()).astype(np.uint8), y_val.numpy()
X_test, y_test = (255*X_test.numpy()).astype(np.uint8), y_test.numpy()

X_train = np.transpose(X_train, (0, 2,3,1))
X_val= np.transpose(X_val, (0, 2,3,1))
X_test = np.transpose(X_test, (0, 2,3,1))

for arrs, name in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)],
                      ['train', 'val', 'test']):
    np.save(os.path.join(CIFAR10_PATH, name + '_imas.npy'), arrs[0])
    np.save(os.path.join(CIFAR10_PATH, name + '_labels.npy'), arrs[1])


os.remove(os.path.join(CIFAR10_PATH, 'cifar-10-python.tar.gz'))
shutil.rmtree(os.path.join(CIFAR10_PATH, 'cifar-10-batches-py'))
