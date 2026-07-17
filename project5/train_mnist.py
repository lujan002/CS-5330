# Luke Jansen
# 7/12/2026

import sys
import argparse
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import matplotlib.pyplot as plt

# CLASS DEFINITIIONS
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # Shape: 28 x 28
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5), # 10 5x5 filers 
            # Shape: 10 x 24 x 24
            nn.MaxPool2d(kernel_size = 2), # scales image down by factor of 2
            # Shape: 10 x 12 x 12
            nn.Conv2d(10, 20, 5), # 20 5x5 filters
            # Shape: 20 x 8 x 8
            nn.Dropout(0.5), # 50% dropout
            nn.MaxPool2d(2),
            # Shape: 20 x 4 x 4
            nn.ReLU(),
            nn.Flatten(), # converts 2+ dims to tensor
            # Shape: 1 x 320
            nn.Linear(in_features = 320, out_features = 50), # Linear layers are by default fully connected. 50 nodes
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.LogSoftmax(),
        )

    # METHODS
    # computes a forward pass for the network
    def forward(self, x):
        # Do not flatten before Conv2d — conv needs [N, C, H, W], e.g. [1, 1, 28, 28]
        # Flatten is already inside the Sequential, after the conv/pool layers
        logits = self.linear_relu_stack(x)
        return logits

# FUNCTIONS
def train_network(device, dataloader, model, loss_fn, optimizer, train_loss_history, examples_seen):
    size = len(dataloader.dataset)
    model.train() # Puts the model in train mode (turns on training-only behavior)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute predicion error 
        pred = model(X) # run images through network, pred is raw score (logits) for each class
        loss = loss_fn(pred, y) # measure how wrong predictions are vs true labels (stays as a tensor)

        # Backpropogation 
        loss.backward() # compute gradients of the loss w.r.t every weight
        optimizer.step() # update weights using gradients
        optimizer.zero_grad() # clear gradients for next batch

        examples_seen += len(X)
        if batch % 100 == 0:
            # Print stats every 100 batches
            loss_val = loss.item()
            print(f"loss: {loss_val:>7f} [{examples_seen:>5d}/{size:>5d}]")
            train_loss_history.append((loss_val, examples_seen))
    return examples_seen
            
def test_network(device, dataloader, model, loss_fn, test_loss_history, examples_seen):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Puts the model in evaluation mode (turns off training-only behavior)
    test_loss = 0 # Total loss over the test set
    correct = 0 # Count of total correct predictions
    # Do not do grad descent on test images
    with torch.no_grad(): 
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device) # move tensors to hardware
            pred = model(X) # run test images through network, pred is raw score (logits) for each class
            test_loss += loss_fn(pred, y).item() # item() converts 0D tensor to plain number
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # finds max score in pred and checks if it matches y
    test_loss /= num_batches # turns summed test loss into average loss per batch
    correct /= size # turns summed correct into average correct over whole test dataset
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_loss_history.append((test_loss, examples_seen))

# MAIN FUNCTION
def main(argv):
    # handle any command line inputs in argv
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-digits",
        action = "store_true",
        help = "Print the first 6 digits from the test set"
    )
    args = parser.parse_args(argv[1:]) # skip script name


    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")

    # main function code
    model = MyNetwork().to(device)
    print(model)

    training_data = datasets.MNIST(
        root = "data", # folder where the dataset is stored
        train = True, # use training split, not test split
        download = True, # if data not in root, download it
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        # convert to torch image tensor, convert to float and scale pixel values from 0-255 to 0-1
    )

    test_data = datasets.MNIST(
        root = "data",
        train = False, # use test split
        download = True, 
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    if args.show_digits:
        # plot first 6 digits of test dataset
        fig, axs = plt.subplots(2, 3)
        for i in range(6):
            img, label = test_data[i]
            ax = axs[i // 3, i % 3]
            ax.imshow(img.squeeze(), cmap="gray") # squeeze to convert from shape: (1, 28, 28) to shape: (28, 28)
            ax.set_title(label)
            ax.axis("off")
        plt.show()
        return


    batch_size = 64

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size = batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # X - input images [N, C, H, W]: (torch.Size([64, 1, 28, 28])
    # y - ground truth labels: (torch.Size([64]) torch.int64)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = 5
    train_loss_history = []
    test_loss_history = []
    examples_seen = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------------")
        examples_seen = train_network(
            device, train_dataloader, model, loss_fn, optimizer, train_loss_history, examples_seen
        )
        test_network(device, test_dataloader, model, loss_fn, test_loss_history, examples_seen)
    print("Done!")

    # histories store (loss, examples_seen) — plot loss vs examples, not the tuples themselves
    train_x = [examples for (_, examples) in train_loss_history]
    train_y = [loss for (loss, _) in train_loss_history]
    test_x = [examples for (_, examples) in test_loss_history]
    test_y = [loss for (loss, _) in test_loss_history]

    # Save model to a file (Step D)
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Plot loss (Step C)
    plt.plot(train_x, train_y, label="Train loss")
    plt.plot(test_x, test_y, "ro", label="Test loss")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)