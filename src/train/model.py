import os

import torch
import datetime

# ToDo: Find a better way to do this
project_path = os.environ.get("PROJECT_PATH")
import sys
sys.path.append(os.path.join(project_path, "src"))

from train import helpers
from torch.utils.tensorboard import SummaryWriter

summary_writer = None

OBJECTIVE = "overfit"
ARCHITECTURE = helpers.Architectures.DENSENET

if OBJECTIVE == "overfit":
    data_augmentation = True
    data_size = "small"
    batch_size = 4


data_path = os.environ.get("DATA_PATH")

dfs_holder, dfs_names = helpers.get_dataframes(
    os.path.join(project_path, "meta"), diseases="all", data=data_size
)

transform = helpers.get_transforms(data_augmentation)

train_loader, test_loader, val_loader = helpers.get_data_loaders(
    dfs_holder, dfs_names, data_path, batch_size=batch_size, num_workers=2, data_augmentation=data_augmentation
)

_, sample_label = next(iter(train_loader))
num_classes = sample_label.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val_accuracy(model, val_loader, epoch):
    model.eval()  # set model to evaluation mode
    ground_truth = torch.FloatTensor()
    predictions = torch.FloatTensor()
    with torch.no_grad():
        for t, (images, labels) in enumerate(val_loader):
            images = images.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            labels = labels.to(device=device, dtype=torch.float32)
            
            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = torch.cat((predictions, probabilities.cpu()), 0)
            ground_truth = torch.cat((ground_truth, labels.cpu()), 0)

        roc_score = helpers.compute_auc(ground_truth, predictions)
        helpers.pprint('Validation Accuracy %f' % roc_score)
        summary_writer.add_scalar('validation_accuracy', roc_score, epoch)
        return roc_score

def train(architecture, optimiser, num_epochs=10):
    helpers.pprint(
        'Training model with architecture:', architecture, 
        'optimiser:', optimiser, 
        'epochs:', num_epochs,
        'time', datetime.datetime.now()
    )

    model = helpers.get_model(architecture, num_classes)
    optimiser = helpers.get_optimiser(model, architecture, optimiser=optimiser)

    # experiment knobs
    # 1. reduction
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    model = model.to(device)
    for epoch in range(num_epochs):
        helpers.pprint('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Initlialize the ground truth and predictions at every epoch
        ground_truth = torch.FloatTensor()
        predictions = torch.FloatTensor()
            
        model.train()
        for t, (images, labels) in enumerate(train_loader):
            if data_augmentation:
                # Combine the batch size and number of crops
                bs, n_crops, c, h, w = images.size()
                images = images.view(-1, c, h, w)
            
            labels = labels.to(device, dtype=torch.float32)
            images = images.to(device, dtype=torch.float32)

            logits = model(images)

            if data_augmentation:
                # Take mean across the crops
                logits = logits.view(bs, n_crops, -1).mean(1)

            loss = criterion(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            probabilities = torch.sigmoid(logits)
            
            ground_truth = torch.cat((ground_truth, labels.cpu()), 0)
            predictions = torch.cat((predictions, probabilities.detach().cpu()), 0)

            summary_writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + t)

            if t % 10 == 0:
                try:
                    auc_score = helpers.compute_auc(ground_truth, predictions)
                    helpers.pprint('Iteration:', t, 'Loss:', loss.item(), 'score:', auc_score)
                    summary_writer.add_scalar('training_auc_score', auc_score, epoch * len(train_loader) + t)
                except ValueError:
                    # If the auc score is not computable, pass
                    helpers.pprint('Iteration:', t, 'Loss:', loss.item(), 'score:', 'Not computable')


        val_accuracy(model, val_loader, epoch)

if __name__ == "__main__":
    for architecture in (helpers.Architectures.DENSENET, helpers.Architectures.VGG, helpers.Architectures.RESNET):
        summary_writer = SummaryWriter(f'runs/{ARCHITECTURE}_adam_10epochs_elu')
        train(architecture=architecture, optimiser=helpers.Optimisers.ADAM, num_epochs=10)