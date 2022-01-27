import argparse

import torch

from denoiser.pretrained import add_model_flags, get_model


def convert(args):
    model = get_model(args)
    model = model.cuda()

    print(model)
    # dummy input
    dummy_input = torch.zeros((1, 1, 256)).cuda()
    # print(model(data))
    input_names = [ "frame" ]
    output_names = [ "output_frame" ]

    print("Optimizing model...")
    torch.onnx.export(model, dummy_input, "denoise.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

    import onnx

    print("Checking model is properly formated...")
    # Load the ONNX model
    model = onnx.load("denoise.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(model)


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_flags(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args)
