class CelebDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        img = self.transform(img)
        return img
dataset = CelebDataset(ds, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
