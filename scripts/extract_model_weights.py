"""
Script to extract the network's state_dict from a checkpoint file
"""

from argparse import ArgumentParser

import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="weights.pt")
    parser.add_argument("--prefix", "-p", type=str, default="net")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint = checkpoint["state_dict"]

    newmodel = {}
    for k, v in checkpoint.items():
        if not k.startswith(args.prefix):
            continue

        k = k.replace(args.prefix + ".", "")
        newmodel[k] = v

    with open(args.output, "wb") as f:
        torch.save(newmodel, f)
