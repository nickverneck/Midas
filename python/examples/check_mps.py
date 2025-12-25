"""Sanity check for PyTorch MPS/CUDA availability."""
import torch


def main():
    print("torch version:", torch.__version__)
    print("mps built:", torch.backends.mps.is_built())
    print("mps available:", torch.backends.mps.is_available())
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda device 0:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
