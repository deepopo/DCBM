import torch
from utils.analysis import Logger, AverageMeter, accuracy, binary_accuracy, js_div

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    loss_meter_main, loss_meter_semantic, loss_meter_penalty, loss_meter_implicit, loss_meter_explicit, acc_attr_meter = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels, attr_labels = data
        if args.n_attributes > 1:
            attr_labels = [i.long() for i in attr_labels]
            attr_labels = torch.stack(attr_labels).t()  # .float() #N x 312
        else:
            if isinstance(attr_labels, list):
                attr_labels = attr_labels[0]
            attr_labels = attr_labels.unsqueeze(1)
        attr_labels_var = torch.autograd.Variable(attr_labels).float()
        attr_labels_var = attr_labels_var.to(args.device)

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.to(args.device)
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.to(args.device)
        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            optimizer.zero_grad()
            losses = []
            # cross_entropy
            loss_main = 1.0 * criterion(outputs[1] + outputs[3], labels_var) + 0.4 * criterion(aux_outputs[1] + aux_outputs[3], labels_var)
            losses.append(loss_main)
            # explicit mapping
            if args.use_cuda:
                loss_concept = [args.attr_loss_weight * (
                            1.0 * attr_criterion[i](outputs[2][:, i].type(torch.cuda.FloatTensor),
                                                    attr_labels_var[:, i]) \
                            + 0.4 * attr_criterion[i](aux_outputs[2][:, i], attr_labels_var[:, i])) for i in
                              range(len(attr_criterion))]
            else:
                loss_concept = [args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[2][:, i], attr_labels_var[:, i]) \
                                                       + 0.4 * attr_criterion[i](aux_outputs[2][:, i],
                                                                                 attr_labels_var[:, i])) for i in
                              range(len(attr_criterion))]
            loss_concept = sum(loss_concept)

            losses.append(loss_concept)
            # implicit embedding
            loss_meter_main.update(loss_main.item(), inputs.size(0))
            loss_meter_semantic.update(loss_concept.item(), inputs.size(0))
            if args.use_embs:

                loss_implicit = js_div(outputs[1], outputs[1] + outputs[3])
                loss_penalty = args.embs_weight * loss_implicit
                loss_explicit = js_div(outputs[3], outputs[1] + outputs[3])
                loss_meter_penalty.update(loss_penalty.item(), inputs.size(0))
                losses.append(loss_penalty)
                loss_meter_implicit.update(loss_implicit.item(), inputs.size(0))
                loss_meter_explicit.update(loss_explicit.item(), inputs.size(0))
        else:
            outputs = model(inputs_var)
            #             outputs[3].retain_grad()
            optimizer.zero_grad()
            losses = []
            # loss_main = 1.0 * criterion(outputs[0], labels_var)
            loss_main = 1.0 * criterion(outputs[1] + outputs[3], labels_var)
            losses.append(loss_main)
            # explicit mapping
            if args.use_cuda:
                loss_concept = [args.attr_loss_weight * (
                            1.0 * attr_criterion[i](outputs[2][:, i].type(torch.cuda.FloatTensor),
                                                    attr_labels_var[:, i])) for i in range(len(attr_criterion))]
            else:
                loss_concept = [args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[2][:, i], attr_labels_var[:, i]))
                              for i in range(len(attr_criterion))]
            loss_concept = sum(loss_concept)
            losses.append(loss_concept)


        sigmoid_outputs = torch.nn.Sigmoid()(outputs[2])
        acc_attr = binary_accuracy(sigmoid_outputs, attr_labels)
        acc_attr_meter.update(acc_attr.data.cpu().numpy(), inputs.size(0))
        acc = accuracy((outputs[1] + outputs[3]).to(args.device), labels, args, topk=(1,))  # only care about class prediction accuracy
        acc_meter.update(acc[0], inputs.size(0))
        if args.use_embs and is_training:
            total_loss = sum(losses[:-1]) / (1 + args.attr_loss_weight * args.n_attributes) + losses[-1]
        else:
            total_loss = sum(losses) / (1 + args.attr_loss_weight * args.n_attributes)
        loss_meter.update(total_loss.item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    if is_training:
        if args.use_embs:
            return loss_meter, acc_meter, acc_attr_meter, loss_meter_main, loss_meter_semantic, loss_meter_penalty, loss_meter_implicit, loss_meter_explicit
        else:
            return loss_meter, acc_meter, acc_attr_meter, loss_meter_main, loss_meter_semantic
    else:
        return loss_meter, acc_meter, acc_attr_meter
