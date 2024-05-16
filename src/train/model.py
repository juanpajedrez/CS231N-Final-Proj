import os

import torch

from train import helpers

OBJECTIVE = "overfit"

if OBJECTIVE == "overfit":
    data_augmentation = True
    data_size = "small"

project_path = os.environ.get("PROJECT_PATH")
data_path = os.environ.get("DATA_PATH")

dfs_holder, dfs_names = helpers.get_dataframes(
    os.path.join(project_path, "meta"), diseases="all", data=data_size
)

transform = helpers.get_transforms(data_augmentation)

train_loader, test_loader, val_loader = helpers.get_data_loaders(
    dfs_holder, dfs_names, data_path, batch_size=32, num_workers=2, transform=transform
)

_, sample_label = next(iter(train_loader))
num_classes = sample_label.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val_accuracy(model, val_loader):
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
        print('Validation Accuracy %f' % roc_score)
        return roc_score

def train(architecture='densenet', optimiser='adam', num_epochs=10):
    if architecture == 'densenet':
        model = helpers.densenet_121(num_classes=num_classes)

    optimiser = helpers.get_optimiser(model, architecture, optimiser_name='adam')

    # experiment knobs
    # 1. reduction
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Initlialize the ground truth and predictions at every epoch
        ground_truth = torch.FloatTensor()
        predictions = torch.FloatTensor()
            
        model.train()
        for t, (images, labels) in enumerate(train_loader):
            labels = labels.to(device, dtype=torch.float32)
            images = images.to(device, dtype=torch.float32)

            logits = model(images)

            loss = criterion(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            probabilities = torch.sigmoid(logits)
            
            ground_truth = torch.cat((ground_truth, labels.cpu()), 0)
            predictions = torch.cat((predictions, probabilities.cpu()), 0)

            if t % 50 == 0:
                auc_score = helpers.compute_auc(ground_truth, predictions)
                print('Iteration:', t, 'Loss:', loss.item(), 'score:', auc_score)

        val_accuracy(model, val_loader)

if __name__ == "__main__":
    train(architecture='densenet', optimiser='adam', num_epochs=10)