from tqdm import tqdm
import torch

def data_gradients(loader, model, dataset="CUB"):
    c_explicit = []
    y = []
    c_truth = []
    y_explicit = []
    y_implicit = []
    y_truth = []
    c_implicit = []
    for _, data in tqdm(enumerate(loader)):
        if dataset in ["CUB", "Derm7pt"]:
            inputs, labels, attr_labels = data
        elif dataset in ['celeba']:
            inputs, (labels, attr_labels) = data
        inputs_var = torch.autograd.Variable(inputs).to(model.device)
        attr_labels = [i.long() for i in attr_labels]
        if dataset in ["CUB", "Derm7pt"]:
            attr_labels = torch.stack(attr_labels).t().to(model.device)
        else:
            attr_labels = torch.stack(attr_labels).to(model.device)
        outputs = model.dcbm(inputs_var)
        attr_outputs = outputs[2]
        c_explicit.extend([attr_outputs.detach()])
        y.extend([outputs[1].detach() + outputs[3].detach()])
        y_explicit.extend([outputs[1].detach()])
        y_implicit.extend([outputs[3].detach()])
        c_implicit.extend([outputs[4].detach()])
        y_truth.extend([labels])
        c_truth.extend([attr_labels])
    c_explicit = torch.cat(c_explicit, dim=0)
    c_implicit = torch.cat(c_implicit, dim=0)
    y = torch.cat(y, dim=0)
    y_explicit = torch.cat(y_explicit, dim=0)
    y_implicit = torch.cat(y_implicit, dim=0)
    y_truth = torch.cat(y_truth, dim=0)
    c_truth = torch.cat(c_truth, dim=0)
    c_truth = c_truth * 2 - 1.0
    return c_explicit, c_implicit, c_truth, y_explicit, y_implicit, y_truth