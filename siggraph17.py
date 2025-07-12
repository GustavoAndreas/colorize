import torch
import torch.nn as nn

# Aqui a gente vai definir o modelo principal: ColorizationNet
# Ele recebe imagens preto-e-branco (canal L) e tenta prever as cores (canais ab)

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()

        # Funçãozinha que ajuda a criar blocos de convolução
        # Um bloco tem: convolução -> ReLU -> convolução -> ReLU
        def conv_block(in_c, out_c, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
                nn.ReLU()
            )

        # Aqui montamos a rede como uma sequência de blocos
        # Começamos simples, depois vamos aumentando os canais (64, 128, 256, 512...)
        # E colocamos 'dilated convolutions' para capturar detalhes em várias escalas

        self.model = nn.Sequential(
            conv_block(1, 64),        # conv1: entrada com 1 canal (L), saída com 64 canais
            conv_block(64, 128),      # conv2: aumenta para 128 canais
            conv_block(128, 256),     # conv3: aumenta para 256 canais
            conv_block(256, 512, dilation=2),  # conv4: usa dilatação pra olhar áreas maiores
            conv_block(512, 512, dilation=2),  # conv5
            conv_block(512, 512, dilation=2),  # conv6
            conv_block(512, 256),     # conv7: começa a reduzir canais de volta
            conv_block(256, 128),     # conv8
            nn.Conv2d(128, 313, kernel_size=1)  # Última camada: 313 classes de cor (ab bins)
        )

    def forward(self, x):
        # A função forward diz como os dados passam pela rede
        # A gente só joga o input L dentro da sequência (self.model)
        out = self.model(x)
        return out  # A saída são mapas de probabilidade para cada classe ab (por pixel)
