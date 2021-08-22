import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------- General ----------
def saveModel(model, modelFileName, stateFileName):
    torch.save(model, modelFileName)
    torch.save(model.state_dict(), stateFileName)
    
def modelEvaluation(model, test_loader, device, is_MTL):
    model.eval()
    predictions = []
    if is_MTL == True:
        with torch.no_grad():
            for test_samples, test_main_labels, test_aux_labels in test_loader:
                test_samples = test_samples.to(device)
                test_main_labels = test_main_labels.to(device)
                test_outputs_main, test_outputs_aux = model(test_samples)
                test_outputs_main = torch.squeeze(test_outputs_main, dim=1)
                predictions.extend(test_outputs_main.cpu().detach().numpy())
    else: 
        with torch.no_grad():
            for test_samples, test_main_labels in test_loader:
                test_samples = test_samples.to(device)
                test_main_labels = test_main_labels.to(device)
                test_outputs_main = model(test_samples)
                test_outputs_main = torch.squeeze(test_outputs_main, dim=1)
                predictions.extend(test_outputs_main.cpu().detach().numpy())
    return predictions

class ArrayDataset(Dataset):
    def __init__(self, df, main_label_vec, aux_label_vec, is_MTL, transform = None):
        self.df = df
        self.main_label_vec = main_label_vec
        self.aux_label_vec = aux_label_vec
        self.is_MTL = is_MTL
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        feature = self.df[idx,:]
        main_label = self.main_label_vec[idx]
        aux_label = self.aux_label_vec[idx]

        data = torch.tensor(feature).float()
        main_label = torch.tensor(main_label).float()
        aux_label = torch.tensor(aux_label).float()
        
        if self.transform:
            data = self.transform(data)

        if self.is_MTL == True:
            return (data, main_label, aux_label)
        else:
            return (data, main_label)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ---------- MTL ----------
class MTLNet(nn.Module):
    def __init__(self, input_size):
        super(MTLNet, self).__init__()
        self.input_size = input_size

        # shared network
        self.sharedNetwork = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.mainTaskNet = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

        self.auxTaskNet = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):

        x_shared = self.sharedNetwork(x)
        output_main = self.mainTaskNet(x_shared)
        output_aux = self.auxTaskNet(x_shared)

        return output_main, output_aux

def modelMTLTrain(model, train_loader, device, optimizer, scheduler, weight, train_losses, Re):
    model.train()

    criterion_main = nn.L1Loss()
    criterion_aux = nn.L1Loss()  

    losses = []
    losses_main = []
    losses_aux = []

    if Re == True:
        lr = get_lr(optimizer)
        print(f'Re-train the model with learning rate = {lr}')

    for _, (samples, main_labels, aux_labels) in enumerate(train_loader):
        samples = samples.to(device)
        main_labels = main_labels.to(device)
        aux_labels = aux_labels.to(device)

        outputs_main, outputs_aux = model(samples)
        outputs_main = outputs_main.reshape(-1)
        outputs_aux = outputs_aux.reshape(-1)

        optimizer.zero_grad()

        loss_main = criterion_main(outputs_main, main_labels)
        loss_aux = criterion_aux(outputs_aux, aux_labels)

        losses.append(loss_main.item() + loss_aux.item())
        losses_main.append(loss_main.item())
        losses_aux.append(loss_aux.item())
        loss = loss_main + torch.mul(loss_aux, weight)
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    mean_loss_main = sum(losses_main) / len(losses_main)
    mean_loss_aux = sum(losses_aux) / len(losses_aux)

    train_losses.append(mean_loss)
    
    if Re == False:
        scheduler.step(mean_loss)
    
    return train_losses, mean_loss, mean_loss_main, mean_loss_aux

def modelMTLValidate(model, val_loader, device, val_losses, valid_losses):
    model.eval()

    criterion_main = nn.L1Loss()
    criterion_aux = nn.L1Loss()  

    with torch.no_grad():
        for val_samples, val_main_labels, val_aux_labels in val_loader:
            val_samples = val_samples.to(device)
            val_main_labels = val_main_labels.to(device)
            val_aux_labels = val_aux_labels.to(device)

            val_outputs_main, val_outputs_aux = model(val_samples)
            val_outputs_main = val_outputs_main.reshape(-1)
            val_outputs_aux = val_outputs_aux.reshape(-1)

            val_loss_main = criterion_main(val_outputs_main, val_main_labels)
            val_loss_aux = criterion_aux(val_outputs_aux, val_aux_labels)
            valid_losses.append(val_loss_main.item() + val_loss_aux.item())

    mean_val_losses = sum(valid_losses) / len(valid_losses)
    val_losses.append(mean_val_losses)

    return mean_val_losses