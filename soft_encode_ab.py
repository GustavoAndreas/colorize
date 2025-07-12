import numpy as np

# carrega os 313 pontos no espaço ab (pré-definidos pelo paper)
pts_in_hull = np.load('pts_in_hull.npy')  # (313,2)

def soft_encode_ab(ab_batch, sigma=5):
 # ab_batch: (B,2,H,W) com os valores ab normalizados [-1,1]
 # vamos transformar isso em um vetor suave de probabilidades (B,313,H,W)

 B, _, H, W = ab_batch.shape

 # reorganiza pra (B*H*W,2)
 ab_flat = ab_batch.transpose(0,2,3,1).reshape(-1,2)

 # calcula distâncias para cada ponto do dicionário
 dists = np.linalg.norm(ab_flat[:,None,:] - pts_in_hull[None,:,:], axis=2)  # (B*H*W,313)

 # pega os 5 vizinhos mais próximos
 top5 = np.argsort(dists, axis=1)[:, :5]

 # calcula pesos gaussianos (quanto mais perto, maior o peso)
 weights = np.exp(- (dists[np.arange(dists.shape[0])[:,None], top5] ** 2) / (2 * sigma ** 2))

 # normaliza os pesos pra somarem 1
 weights = weights / np.sum(weights, axis=1, keepdims=True)

 # cria matriz final (B*H*W,313) com zeros
 Z = np.zeros((ab_flat.shape[0], 313))
 Z[np.arange(dists.shape[0])[:,None], top5] = weights

 # reorganiza pra (B,313,H,W)
 return Z.reshape(B,H,W,313).transpose(0,3,1,2)
