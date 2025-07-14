import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import ColorizationDataset
from siggraph17 import ColorizationNet
from quantize_ab import ab_to_class
from loss import ColorizationLoss

if __name__ == "__main__":
    # define alguns parâmetros básicos
    batch_size = 8
    num_epochs = 3
    learning_rate = 3e-4

    # carrega o dataset (pega as imagens da pasta)
    train_dataset = ColorizationDataset('./data/train/')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # inicializa o modelo (a rede neural)
    model = ColorizationNet()
    device = torch.device('cuda')
    model = model.to(device)

    # define o otimizador (quem vai ajustar os pesos)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # define a função de perda (o quanto a rede erra)
    criterion = ColorizationLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nComeçando época {epoch+1}/{num_epochs}...")

        for i, batch in enumerate(train_loader):
            print(f"  Rodando batch {i+1}/{len(train_loader)}...")

            # pega o canal L e os canais ab do batch
            L = batch['L'].float().to(device)        # (B,1,H,W)
            ab = batch['ab'].float().to(device)     # (B,2,H,W)

            # transforma ab em índices de classe (0 a 312)
            target = ab_to_class(ab.cpu().numpy())
            target = torch.from_numpy(target).to(device)

            # zera os gradientes
            optimizer.zero_grad()

            # passa L pelo modelo e obtém as previsões (313 classes por pixel)
            output = model(L)

            # calcula a perda (o erro entre previsão e target)
            loss = criterion(output, target)

            # faz backpropagation (ajusta os pesos da rede)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # salva os pesos do modelo (checkpoint)
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
