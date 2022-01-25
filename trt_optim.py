import argparse
import os

import torch

from torch2trt import torch2trt
from denoiser.pretrained import add_model_flags, get_model


def convert(args):
    model = get_model(args)
    model = model.cuda().eval().half()
    print(model)
    # dummy input
    data = torch.zeros((1, 86, 2)).cuda().half()

    print("Optimizing model...")
    model_trt = torch2trt(model, [data], fp16_mode=True)
    torch.save(model_trt.state_dict(), "trt-model.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_flags(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args)