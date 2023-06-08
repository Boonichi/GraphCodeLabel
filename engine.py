import torch
from typing import Iterator

def train_one_epoch(model : torch.nn.Module, criterion = torch.nn.Module, optimizer = torch.optim.Optimizer, data_loader = Iterator, device = torch.device,
                    update_freq = None):
    model.train()

    optimizer.zero_grad()
    
    for data_iter, (samples, targets) in enumerate(data_loader):
        samples.to(device)
        targets.to(device)
        
        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        loss /= update_freq
        loss.backward()
        if (data_iter + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        optimizer.step()

    return None

@torch.no_grad()
def evaluation(model: torch.nn.Module, data_loader = Iterator):
    
    return None