import torch
import tqdm
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric



'''def acc_metric(outputs, labels):
    Computes batch accuracy

    _,preds = torch.max(outputs,dim=1)
    correct = (preds ==labels).sum().item()
    return correct / labels.size(0)'''

def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()
    acc_metric = MulticlassAccuracy(num_classes=13,average="micro")
    mean_metric = MeanMetric()
    prog_bar = tqdm.tqdm(loader,desc="training",leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    running_loss = 0.0
    running_acc = 0.0

    for data,target in prog_bar:
        # Send data and target to appropriate device.
        data, target = data.to(device), target.to(device)

        lr = optimizer.param_groups[0]['lr']
        # Reset parameters gradient to zero.
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()
        batch_loss = mean_metric(loss.item(), weight=data.shape[0])

        # Get probability score using softmax.
        prob = F.softmax(output, dim=1)

        # Get the index of the max probability.
        pred_idx = prob.detach().argmax(dim=1)

        # Batch accuracy.
        batch_acc = acc_metric(pred_idx.cpu(), target.cpu())

        running_acc+=batch_acc
        running_loss+=loss.item()

        prog_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_acc:.4f}"
        )

    epoch_loss = mean_metric.compute()
    epoch_acc = acc_metric.compute()

    prog_bar.close()
    return epoch_loss, epoch_acc


def validate_one_epoch(model,loader,criterion,device):
    model.eval()

    
    count_sample = 0
    step_loss = 0
    step_accuracy = 0


    prog_bar = tqdm.tqdm(loader,desc ="Validation",leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for data, target in prog_bar:
        data, target = data.to(device), target.to(device)

        # Get the model's predicted logits.
        with torch.no_grad():
            output = model(data)

        # Compute the CE-Loss.
        test_loss = F.cross_entropy(output, target).item()

        # Convert model's logits to probability scores.
        prob = F.softmax(output, dim=1)

        # Get the class id for the maximum score.
        pred_idx = prob.detach().argmax(dim=1)

        # Batch validation loss.
        step_loss+= test_loss * data.shape[0]

        # Batch validation accuracy.
        step_accuracy+= (pred_idx.cpu() == target.cpu()).sum()

        # Count samples.
        count_sample+= data.shape[0]

        # Update progress bar description.
        prog_bar.set_postfix(
                loss=f"{step_loss:.4f}",
                acc=f"{step_accuracy:.4f}"
            )
    test_loss = float(step_loss / len(loader.dataset))
    test_acc = float(step_accuracy/ len(loader.dataset))

    prog_bar.close()

    return test_loss, test_acc
    