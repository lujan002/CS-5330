# Luke Jansen
# 7/16/26
# Perform transfer learning on the model pre-trained on the MNIST dataset by 
# freezing all weights except the last layer, which is changed to have three 
# outputs and retrained


# imports
import argparse
import sys
import torch
import torchvision
from train_mnist import MyNetwork # import custom class
from torch import nn
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# greek data set transform (converts to 28x28, greyscale, invert intensities to match MNIST)
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
class CustomGreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # Grayscale
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # Resize to 28x28 WITHOUT cropping
        x = torchvision.transforms.functional.resize(x, (28, 28))
        # Apply quantization: set to 0 if pixel < 128, otherwise keep actual value
        x = torch.where(x < 128 / 255.0, torch.tensor(0.0, dtype=x.dtype, device=x.device), x)
 
        # Invert 
        return torchvision.transforms.functional.invert( x )

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

def test_network(device, dataloader, model):
    model.eval() # Puts the model in evaluation mode (turns off training-only behavior)
    # Do not do grad descent on test images
    img_count = 0
    images = []
    pred_labels = []
    true_labels = []
    outputs_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # move tensors to hardware
            pred = model(X) # shape [batch, n_classes]
            for i in range(pred.shape[0]):
                outputs = [f"{v:.2f}" for v in pred[i].tolist()]
                pred_label = pred[i].argmax().item()
                true_label = y[i].item()
                print(
                    f"Image {img_count}: outputs={outputs}, "
                    f"pred={pred_label}, label={true_label}"
                )
                # For plotting later
                images.append(X[i].cpu().squeeze())
                pred_labels.append(pred_label)
                true_labels.append(true_label)
                outputs_list.append(outputs)
                img_count += 1

    # Compute grid size: close to a square
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    axs = np.array(axs)  # In case it's returned 1D

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        ax.imshow(img, cmap="grey")
        ax.set_title(f"Pred: {pred_labels[idx]}\nTrue: {true_labels[idx]}")
        ax.axis("off")

    # Hide any remaining empty axes
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main(argv):
    # handle any command line inputs in argv
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    args = parser.parse_args(argv[1:]) # skip script name

    # NOTE: we do not create a test subset from the data, only use train subset
    # import model
    model = MyNetwork()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    
    # set device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")

    # set paths to data
    training_set_path = "data/greek/greek_train"
    test_set_path = "data/greek/greek_test"

    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,) ) ] ) ),
        batch_size = len(torchvision.datasets.ImageFolder(training_set_path)),
        shuffle = True )
    
    # DataLoader for custom greek data
    greek_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( test_set_path,
            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                CustomGreekTransform(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,) ) ] ) ),
        batch_size = len(torchvision.datasets.ImageFolder(test_set_path)),
        shuffle = True )
    
    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer with a new Linear layer with three nodes
    # new layer has requires_grad = True by default, good because it won't train otherwise
    model.linear_relu_stack[9] = nn.Linear(50,3)

    # must create optimizer after freezing and swapping
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # main train loop
    epochs = args.epochs # How many epochs should it take? Should I make this number variable
    # based on the training error threshold (loss?)? 
    train_loss_history = []
    # test_loss_history = []
    examples_seen = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------------")
        examples_seen = train_network(device, greek_train, model, loss_fn, optimizer, train_loss_history, examples_seen)
    
    test_network(device, greek_train, model) # prints subplots of custom images and their predictions
    test_network(device, greek_test, model) # prints subplots of custom images and their predictions

    train_x = [examples for (_, examples) in train_loss_history]
    train_y = [loss for (loss, _) in train_loss_history]
    # test_x = [examples for (_, examples) in test_loss_history]
    # test_y = [loss for (loss, _) in test_loss_history]

    # Plot loss of training and test set
    plt.plot(train_x, train_y, label="Train loss")
    # plt.plot(test_x, test_y, "ro", label="Test loss")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.legend()
    plt.show()

    print(f"model architecture: {model}")

if __name__ == "__main__":
    main(sys.argv)