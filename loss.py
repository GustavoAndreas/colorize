import torch
import torch.nn as nn

# define uma classe simples pra calcular a loss
class ColorizationLoss(nn.Module):
 def __init__(self, weight=None):
  super(ColorizationLoss, self).__init__()
  # usamos CrossEntropy porque estamos tratando como classificação (313 classes)
  self.criterion = nn.CrossEntropyLoss(weight=weight)

 def forward(self, output, target):
  # output: (B,313,H,W), target: (B,H,W) com índices de classe
  loss = self.criterion(output, target)
  return loss
