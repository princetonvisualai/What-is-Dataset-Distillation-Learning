import sys
sys.path.append(".")
import torch
from random import shuffle
import torch
from tqdm import tqdm
from data_utils import get_dataset
from model_utils import get_network
import numpy as np
import os


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists("./experiment_results/agreement_models/"):
        os.mkdir("./experiment_results/agreement_models/")
    for model_id in tqdm(range(520)):
        channel, im_size, train_n_classes, test_n_classes, dst_train, dst_test = get_dataset(
                    "CIFAR10",
                    "../data",
                    zca=False
                )
        assert train_n_classes == test_n_classes
        label_indices = [[] for _ in range(train_n_classes)]
        for i, (x,y) in enumerate(dst_train):
            label_indices[y].append(i)
        for list in label_indices:
            shuffle(list)
        indices = []
        subset = np.random.uniform(0.005, 0.05) # random sample between 0.5% to 5% of training data
        for list in label_indices:
            indices += list[:int(len(list)*subset)]
        dst_train = torch.utils.data.Subset(dst_train, indices)
        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=8)
        train_model = get_network("ConvNet", channel, train_n_classes, im_size).to(device)
        optim = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        train_model.train()
        for _ in range(100):
            for x,y in train_loader:
                x = x.to(device)
                y = y.to(device)
                pred = train_model(x)[-1]
                loss = criterion(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
        predictions = []
        num_correct = 0
        num_total = 0
        train_model.eval()
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = torch.argmax(train_model(x)[-1], dim=1)
            predictions += pred.tolist()
            num_correct += torch.sum(pred == y).item()
            num_total += len(y)
        torch.save({"predictions": predictions, "accuracy": num_correct/num_total}, f"./experiment_results/agreement_models/subset_{model_id}.pt")

    channel, im_size, train_n_classes, test_n_classes, dst_train, dst_test = get_dataset(
                "CIFAR10",
                "../data",
                zca=False
            )
    assert train_n_classes == test_n_classes
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=8)
    for model_id in tqdm(range(30)):
        train_model = get_network("ConvNet", channel, train_n_classes, im_size).to(device)
        optim = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        all_the_predictions = [] # predictions across every iteration
        for x,y in train_loader:
            train_model.train()
            x = x.to(device)
            y = y.to(device)
            pred = train_model(x)[-1]
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            predictions_at_iter = [] # predictions from this iteration
            train_model.eval()
            for x,y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = torch.argmax(train_model(x)[-1], dim=1)
                predictions_at_iter += pred.tolist()
                num_correct += torch.sum(pred == y).item()
                num_total += len(y)
            all_the_predictions.append(predictions_at_iter)
        torch.save({"predictions": all_the_predictions}, f"./experiment_results/agreement_models/earlystopping_{model_id}.pt")

