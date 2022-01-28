# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys
import threading

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
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=args.device_sr,
        channels=channels_in,  # try only one channel to reduce latency
        # blocksize=706, try this
        dtype=np.float32,
        # latency='low',
        # never_drop_input=True
    )

    rs_in = soxr.ResampleStream(
        args.device_sr,
        model.sample_rate,
        channels_in,
        dtype='float32'
    )

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=args.device_sr,
        channels=channels_out,
        dtype=np.float32)

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

    raw_audio = None
    output = None
    out_flag = False
    thread_running = True
    overflow = False
    underflow = False
    first = True
    block_size = 2822
    first_block_size = 3938


    def start_reading():
        global raw_audio
        global overflow
        global first
        while thread_running:
            print("read read")
            length = first_block_size if first else block_size
            print(length)
            raw_audio, overflow = stream_in.read(length)
            print(raw_audio)


    def start_writing():
        global output
        global out_flag
        global underflow
        while not out_flag:
            pass
        while thread_running:  # if new frame to write
            output_copy = output
            out_flag = False
            print(output_copy.shape)
            underflow = stream_out.write(output_copy)

    stream_in.start()
    stream_out.start()

    read_thread = threading.Thread(target=start_reading, daemon=True)
    read_thread.start()

    write_thread = threading.Thread(target=start_writing, daemon=True)
    write_thread.start()

    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = model.sample_rate / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    counter = 0

    frame = raw_audio
    while True:
        counter += 1
        if counter == 500:
            break
        try:
            print(frame)
            print(raw_audio)
            if True:# not frame == raw_audio:  # if new frame available
                if current_time > last_log_time + log_delta:
                    last_log_time = current_time
                    tpf = streamer.time_per_frame * 1000
                    rtf = tpf / stride_ms
                    print(f"time per frame: {tpf:.1f}ms, ", end='')
                    print(f"RTF: {rtf:.1f}")
                    streamer.reset_time_per_frame()

                length = streamer.total_length if first else streamer.stride
                first = False
                current_time += length / model.sample_rate
                # to_read = int(length*args.device_sr/model.sample_rate)
                # frame, overflow = stream_in.read(to_read)

                frame = raw_audio
                frame = rs_in.resample_chunk(frame, last=False)
                frame = torch.from_numpy(frame).mean(dim=1)

                frame = frame.to(args.device)
                with torch.no_grad():
                    out = streamer.feed(frame[None])[0]
                if not out.numel():
                    continue

                if args.compressor:
                    out = 0.99 * torch.tanh(out)
                out = out[:, None].repeat(1, channels_out)
                mx = out.abs().max().item()
                if mx > 1:
                    print("Clipping!!")
                out.clamp_(-1, 1)
                out = out.cpu().numpy()
                out = rs_out.resample_chunk(out, last=False)

                # underflow = stream_out.write(out)
                output = out
                out_flag = True

                if overflow or underflow:
                    if current_time >= last_error_time + cooldown_time:
                        last_error_time = current_time
                        tpf = 1000 * streamer.time_per_frame
                        print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
                              f"(should be less than {stride_ms:.1f}ms).")
        except KeyboardInterrupt:
            print("Stopping")
            break

    thread_running = False
    read_thread.join()
    write_thread.join()

    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
