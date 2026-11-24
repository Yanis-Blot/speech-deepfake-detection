import faulthandler, sys
faulthandler.enable()
import torch
import torch.nn as nn

print("torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version(), flush=True)
print("GPU :", torch.cuda.get_device_name(0), flush=True)

x = torch.randn(8, 64000)

print("1) STFT CPU...", flush=True)
w = torch.hann_window(512)
torch.stft(x, 512, 256, window=w, return_complex=True)
print("   OK", flush=True)

print("2) matmul GPU...", flush=True)
a = torch.randn(1000, 1000, device="cuda")
(a @ a).sum().item()
print("   OK", flush=True)

print("3) STFT GPU (cuFFT)...", flush=True)
torch.stft(x.cuda(), 512, 256, window=w.cuda(), return_complex=True)
torch.cuda.synchronize()
print("   OK", flush=True)

print("4) Conv2d GPU (cuDNN), cudnn activé...", flush=True)
conv = nn.Conv2d(1, 16, 3, padding=1).cuda()
conv(torch.randn(8, 1, 257, 251, device="cuda")).sum().item()
torch.cuda.synchronize()
print("   OK", flush=True)

print("5) Conv2d GPU, cudnn désactivé...", flush=True)
torch.backends.cudnn.enabled = False
conv(torch.randn(8, 1, 257, 251, device="cuda")).sum().item()
torch.cuda.synchronize()
print("   OK", flush=True)