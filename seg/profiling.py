import time
from typing import Optional, Tuple

import torch


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model: torch.nn.Module, input_size: Tuple[int, int], device: str) -> Optional[int]:
    h, w = input_size
    dummy = torch.randn(1, 3, h, w, device=device)
    model.eval()
    with torch.no_grad():
        try:
            from thop import profile as thop_profile  # type: ignore

            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return int(flops)
        except Exception:
            pass
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore

            flops = FlopCountAnalysis(model, dummy).total()
            return int(flops)
        except Exception:
            return None


def benchmark_fps(
    model: torch.nn.Module,
    input_size: Tuple[int, int],
    device: str,
    use_amp: bool,
    warmup: int = 10,
    iters: int = 50,
) -> Optional[float]:
    if device != "cuda":
        return None
    h, w = input_size
    dummy = torch.randn(1, 3, h, w, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(dummy)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    if elapsed <= 0:
        return None
    return float(iters) / elapsed


def describe_profile(
    model: torch.nn.Module,
    input_size: Tuple[int, int],
    device: str,
    use_amp: bool,
) -> str:
    total, trainable = count_params(model)
    flops = estimate_flops(model, input_size, device)
    fps = benchmark_fps(model, input_size, device, use_amp=use_amp)
    lines = [
        f"Params: {_format_count(total)} (trainable {_format_count(trainable)})",
    ]
    if flops is None:
        lines.append("FLOPs: N/A (install thop or fvcore)")
    else:
        lines.append(f"FLOPs: {flops / 1e9:.2f} GFLOPs @ 1x3x{input_size[0]}x{input_size[1]}")
    if fps is None:
        lines.append("FPS: N/A (CUDA required)")
    else:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        lines.append(f"FPS: {fps:.2f} ({gpu_name}, batch=1)")
    return " | ".join(lines)
