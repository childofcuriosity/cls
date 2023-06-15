from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, samples, transforms=None): # samples是list，元素是(img,label)
        self.transforms = transforms
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img,label = self.samples[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


