# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
from skimage.color import rgb2lab

class ColorizationDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: pasta onde estão as imagens coloridas (jpg, png, etc.)
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        # Quantidade total de imagens no dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Para cada índice, pega:
        - input: canal L normalizado [-1,1]
        - target: canais ab normalizados [-1,1]
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Lê imagem com OpenCV e converte BGR → RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensiona para 256x256 (ou outro tamanho fixo)
        img = cv2.resize(img, (256, 256))

        # Converte para Lab (escala Lab típica: L [0,100], a [-128,127], b [-128,127])
        lab = rgb2lab(img).astype("float32")
        
        # Normaliza:
        L = lab[:, :, 0] / 50.0 - 1.0       # L normalizado para [-1,1]
        ab = lab[:, :, 1:] / 110.0          # ab normalizado para [-1,1]

        # Transforma em formato C x H x W (como PyTorch espera)
        L = np.expand_dims(L, axis=0)                  # 1 x H x W
        ab = np.transpose(ab, (2, 0, 1))              # 2 x H x W

        return {'L': L, 'ab': ab}
