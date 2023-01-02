import torch
import torch.nn as nn
import numpy as np
from utils import globalvar as gl


def test(model, loader):
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    total_loss, correct = 0, 0
    size = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            total_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
            size += inputs.size(0)
        avg_loss = total_loss / size
        avg_acc = 100.0 * float(correct) / size
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} F1 ({:.1f}%)'.format(avg_loss, correct, size, 100 * correct  / size))
    return avg_loss, avg_acc

def predict(model, loader):
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    labels = []
    with torch.no_grad():
        # DO NOT use the true label, just generate the pseudo labels
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            outputs, _ = model(inputs)
            preds = torch.max(outputs, 1)[1].tolist()
        
            labels += preds

    return labels



