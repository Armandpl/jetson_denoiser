# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys
import time

import numpy as np
import sounddevice as sd
import soxr
import torch

from .demucs import DemucsStreamer
from .pretrained import add_model_flags, get_model
from .utils import bold

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats, output_unit=1e-03)


def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu")
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    parser.add_argument(
        "--device_sr", type=int, default=16000,
        help="Specify a device sample rate to resample to.")
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps

@profile
def main():
    args = get_parser().parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    model = get_model(args).to(args.device)
    model.eval()
    print("Model loaded.")
    streamer = DemucsStreamer(model, dry=args.dry, num_frames=args.num_frames)

    device_in = parse_audio_device(args.in_)
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)

    rs_in = soxr.ResampleStream(
        args.device_sr,
        model.sample_rate,
        channels_in,
        dtype='float32'
    )

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)

    rs_out = soxr.ResampleStream(
        model.sample_rate,
        args.device_sr,
        channels_out,
        dtype='float32'
    )

    # warmup gpu
    if args.device == "cuda":
        torch.zeros(1000).cuda()
        torch.zeros(1000).cuda()

    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = model.sample_rate / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)

        frame = indata
        frame = rs_in.resample_chunk(frame, last = False)
        frame = torch.from_numpy(frame).mean(dim=1)

        frame = frame.to(args.device)
        with torch.no_grad():
            out = streamer.feed(frame[None])[0]
        if not out.numel():
            outdata[:] = indata
            return

        if args.compressor:
            out = 0.99 * torch.tanh(out)
        out = out[:, None].repeat(1, channels_out)
        mx = out.abs().max().item()
        if mx > 1:
            print("Clipping!!")
        out.clamp_(-1, 1)
        out = out.cpu().numpy()
        out = rs_out.resample_chunk(out, last = False)
        if out.shape[0] < indata.shape[0]:  # buffer not filled yet
            outdata[:] = indata
        else:
            outdata[:] = out[:-1,:]

    to_read = int(streamer.stride*args.device_sr/model.sample_rate)
    with sd.Stream(device=(device_in, device_out),
                   samplerate=args.device_sr, blocksize=to_read,
                   dtype=np.float32, latency=0,
                   channels=channels_in, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()


if __name__ == "__main__":
    main()
