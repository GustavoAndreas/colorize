import torch
import numpy as np

# carrega os 313 pontos ab do paper
pts_in_hull = np.load('pts_in_hull.npy')  # (313,2)

def decode_ab(output, temperature=0.38):
 # output: tensor (1,313,H,W), com as probabilidades por classe
 # vamos transformar isso num mapa ab contínuo (1,2,H,W)

 # aplica softmax com temperatura pra ajustar confiança
 output = torch.softmax(output / temperature, dim=1)

 # multiplica as probabilidades pelos valores ab reais
 output = output.permute(0,2,3,1).cpu().numpy()  # (1,H,W,313)
 ab_map = np.dot(output, pts_in_hull)  # (1,H,W,2)

 # reorganiza pra (1,2,H,W)
 ab_map = ab_map.transpose(0,3,1,2)
 return ab_map
