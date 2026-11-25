import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scripts.constantes import EPOCH, SEED, TEST_RATIO
from scripts.utils import CNN, CustomLoss


def evaluate_model(model, test_loader, loss_function):
    model.eval()

    test_loss = 0.0
    for inputs, labels in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    return test_loss / len(test_loader.dataset)


def main():
    torch.manual_seed(SEED)

    image_path = sorted(glob.glob("frames/data/*/before.png"))
    image_data = np.array([np.array(Image.open(image)) for image in image_path])

    label_path = sorted(glob.glob("frames/data/*/label.npy"))
    label_data = np.array([np.load(label) for label in label_path])

    split_index = int(len(image_data) * TEST_RATIO)

    X_test = torch.FloatTensor(image_data[0:split_index])
    X_train = torch.FloatTensor(image_data[split_index:])

    y_test = torch.FloatTensor(label_data[0:split_index])
    y_train = torch.FloatTensor(label_data[split_index:])

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=2, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = CustomLoss()

    train_losses = []
    test_losses = []
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        train_losses.append(epoch_loss / len(train_loader.dataset))
        test_losses.append(evaluate_model(model, test_loader, loss_function))

    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and test loss over time")
    plt.legend()
    plt.savefig("models/target_detector.png")

    torch.save(model.state_dict(), "models/target_detector.pt")


if __name__ == "__main__":
    main()
