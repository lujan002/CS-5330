# Luke Jansen
# 7/17/26
# Evaluate network, analyze the first layer

# imports
import os
import sys
import argparse
import torch
from train_mnist import MyNetwork # import custom class
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2

import cv2 as cv
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# model tensor shape: [N, C, H, W] (num filers, num channels, width, height)

def main(argv):
    # import model 
    model = MyNetwork()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    # get the weights of first layer
    weight_tensor = model.linear_relu_stack[0].weight  # shape [10, 1, 5, 5]

    # set up plot
    fig1, axs1 = plt.subplots(3, 4)
    fig2, axs2 = plt.subplots(5, 4)

    # set up test data and get first image
    test_data = datasets.MNIST(
        root = "data",
        train = False, # use test split
        download = True, 
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )
    first_img, _ = test_data[0]

    with torch.no_grad():
        for i in range(weight_tensor.size(0)):
            filter = weight_tensor[i, 0]
            # that is cut off from the autograd graph and convert to numpy (imshow expects numpy)
            ax1 = axs1[i // 4, i % 4]
            ax1.imshow(filter, cmap="viridis")
            ax1.set_title(f"Filter {i}")
            ax1.axis("off")

            ax2 = axs2[i // 2, 2*i % 4]
            dst = cv.filter2D(first_img.squeeze().numpy(), -1, filter.squeeze().numpy()) # convolve filter over image
            ax2.imshow(filter, cmap="grey")
            ax3 = axs2[i // 2, (2*i+1) % 4]
            ax3.imshow(dst.squeeze(), cmap="grey")
            ax2.axis('off')
            ax3.axis('off')


    plt.show()

if __name__ == "__main__":
    main(sys.argv)
