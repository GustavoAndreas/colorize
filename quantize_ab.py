import numpy as np

# Carrega os 313 pontos de referência no espaço de cor ab
pts_in_hull = np.load('pts_in_hull.npy')  # isso vem do paper original

def ab_to_class(ab_batch):
 # ab_batch: array numpy com forma (B, 2, H, W)
 # vamos transformar isso em índices de classe (0-312)

 B, _, H, W = ab_batch.shape

 # reorganiza para (B*H*W, 2), ou seja, junta tudo numa lista longa
 ab_flat = ab_batch.transpose(0,2,3,1).reshape(-1,2)

 # calcula a distância de cada ponto ab para cada centro do dicionário
 dists = np.linalg.norm(ab_flat[:,None,:] - pts_in_hull[None,:,:], axis=2)  # (B*H*W, 313)

 # pega o índice (classe) do centro mais próximo
 idx = np.argmin(dists, axis=1)  # (B*H*W,)

 # volta para formato (B, H, W)
 return idx.reshape(B,H,W).astype(np.int64)
