import tables
import numpy as np
from torch.utils.data import Dataset 

class DatasetCM(Dataset):    
    def __init__(self, h5_file, transform=None): 
        h5_data = tables.open_file(h5_file, 'r') 
        self.data = np.array([triu_to_full(contact_map) for contact_map in h5_data.root.contact_maps.read()])
        self.image_size = self.data.shape[-1]
        if self.image_size % 2 == 1: 
            self.image_size = self.image_size + 1 
            self.data = [np.pad(cm, ((0, 1), (0, 1)), mode='constant') for cm in self.data] 
        self.shape = (self.image_size, self.image_size, 1)
        self.transform = transform 
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
#         if self.image_size % 2 == 1: 
#             self.padded_image_size = self.image_size + 1
#             self.padded_shape = (self.padded_image_size, self.padded_image_size, 1)             
#             image = np.pad(triu_to_full(self.data[index, :]), ((0, 1), (0, 1)), mode='constant').astype(np.float).reshape(self.padded_shape) 
#         else: 
        image = self.data[index].astype(np.float).reshape(self.shape) 
#         np.pad(pad_test, ((0, 1), (0, 1)), mode='constant').shape
        if self.transform is not None:
            image = self.transform(image) 
        return image 

    
def triu_to_full(cm0): 
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0 
    cm_full.T[iu1] = cm0 
    np.fill_diagonal(cm_full, 1)
    
    return cm_full 