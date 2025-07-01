from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, textos):
        self.textos = textos

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        return texto
