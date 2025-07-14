import argparse
import torch
import cv2
import numpy as np

from siggraph17 import ColorizationNet
from quantize_ab import ab_to_class
from postprocess import decode_ab
from dataset import ColorizationDataset

# parser pra pegar argumentos do terminal (input e output)
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# carrega o modelo
model = ColorizationNet()
model.load_state_dict(torch.load('checkpoint_epoch_3.pth', map_location='cpu', weights_only=True))
model.eval()

# lê imagem de entrada
img = cv2.imread(args.input)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256,256))  # redimensiona pra bater com o modelo

# converte pra Lab e normaliza
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")
L = lab[:,:,0] / 50. - 1.0  # [-1,1]
L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float()

# passa pelo modelo pra prever as cores
with torch.no_grad():
 output = model(L_tensor)  # (1,313,H,W)
 pred_ab = decode_ab(output)  # transforma saída em ab contínuo

# monta imagem final (junta L com ab)
L_out = (L + 1.0) * 50.0  # volta pro range original
result_lab = np.zeros((256,256,3))
result_lab[:,:,0] = L_out[0]
result_lab[:,:,1:] = pred_ab[0].transpose(1,2,0)

# converte de Lab pra RGB e salva
result_rgb = cv2.cvtColor(result_lab.astype("uint8"), cv2.COLOR_LAB2RGB)
result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.output, result_bgr)
