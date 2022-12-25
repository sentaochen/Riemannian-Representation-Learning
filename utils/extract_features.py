import torch
import numpy as np
from utils import globalvar as gl

def extract_features(args, model, dataloader, domain):
    extract_path = gl.get_value('extract_path')
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    fea = torch.zeros(1, model._in_features + 1).to(DEVICE)
    with torch.no_grad(): # Disabling gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            x = model.get_features(inputs)
            labels = labels.view(labels.size(0), 1).float()
            x = torch.cat((x, labels), dim=1)
            fea = torch.cat((fea, x), dim=0)
    fea_numpy = fea.cpu().numpy()
    if isinstance(args.source, list):
        source = '['
        for s in args.source:
            source += s
        source += ']'
    else:
        source = args.source
    np.savetxt('{}/{}_{}_{}.csv'.format(extract_path, str.upper(args.net), source, domain), fea_numpy[1:], fmt='%.6f', delimiter=',')
    print('{}: {} - {} done!'.format(args.net,  args.source, domain))

