# Luke Jansen
# 7/16/26
# Read the network from a file and run the model on the first 10 examples in the test set
# Prints out the 10 output values (2 decimal places), index of max output value, and actual correct label
# Plot first 9 digits with predictions in 3x3 grid
import os
from PIL import Image
import sys
import argparse

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch import nn
from train_mnist import MyNetwork # import custom class
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt


def test_network(device, dataloader, model):
    model.eval() # Puts the model in evaluation mode (turns off training-only behavior)
    # Do not do grad descent on test images
    img_count = 0
    fig, axs = plt.subplots(3, 3)
    with torch.no_grad(): 
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device) # move tensors to hardware
            pred = model(X) # shape [batch, 10] — one row per image
            for i in range(pred.shape[0]):
                outputs = [f"{v:.2f}" for v in pred[i].tolist()]
                pred_label = pred[i].argmax().item()
                true_label = y[i].item()
                print(
                    f"Image {img_count}: outputs={outputs}, "
                    f"pred={pred_label}, label={true_label}"
                )
                img_count += 1

                if i < 9:
                    img = X[i].cpu().squeeze() # .cpu() makes sure CUDA tensor is on CPU and not GPU 
                    ax = axs[i // 3, i % 3]
                    ax.imshow(img.squeeze(), cmap="grey")
                    ax.set_title(f"Prediction: {pred_label}")
                    ax.axis("off")
    
def main(argv):
    # handle any command line inputs in argv
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-digits",
        action = "store_true",
        help = "Use custom digits for the test set"
    )
    args = parser.parse_args(argv[1:]) # skip script name


    # import model 

    model = MyNetwork()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")

    batch_size = 10
    if args.custom_digits:
        # Use only the pre-cropped *_resize images (folder also has full-size crops)
        custom_digits_dir = "data/custom_digits"
        image_filenames = sorted([
            f for f in os.listdir(custom_digits_dir)
            if "_resize" in f and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        images = []
        labels = []
        for filename in image_filenames:
            # filename like custom_7_resize.jpeg → label 7
            digit = int(filename.split("_")[1])
            img_path = os.path.join(custom_digits_dir, filename)
            # Force exact MNIST size — *_resize files are close but not always 28x28
            img = Image.open(img_path).convert("L").resize((28, 28))
            # Invert: handwritten digits are black-on-white; MNIST is white-on-black
            img_tensor = 1.0 - (torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0)
            images.append(img_tensor)
            labels.append(digit)
        images = torch.stack(images)              # [N, 1, 28, 28]
        labels = torch.tensor(labels, dtype=torch.long)

        test_data = TensorDataset(images, labels)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
    else:
        test_data = datasets.MNIST(
            root = "data",
            train = False, # use test split
            download = True, 
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )

        # only use first 10 images in test set
        test_subset = Subset(test_data, range(10))  # first 10 examples
        test_dataloader = DataLoader(test_subset, batch_size=batch_size)

    test_network(device, test_dataloader, model)

    plt.show()

if __name__ == "__main__":
    main(sys.argv)