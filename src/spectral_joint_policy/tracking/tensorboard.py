from __future__ import annotations

import numpy as np
import torch


def _stack(frames: list):
    try:
        video = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    except (ValueError, TypeError):
        return None
    return video if video.ndim == 4 else None


class TensorboardBackend:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, flush_secs=30)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            if value is None:
                continue
            self.writer.add_scalar(key, float(value), step)

    def log_video(self, frames: list, step: int, *, fps: int = 30) -> None:
        video = _stack(frames)
        if video is None:
            return
        try:
            import io

            from torchvision.io import write_video

            buf = io.BytesIO()
            write_video(buf, video.permute(0, 3, 1, 2), fps=fps, format="mp4")
            self.writer.add_video("eval/video", torch.frombuffer(buf.getvalue(), dtype=torch.uint8), fps, step)
        except Exception:
            return

    def finish(self, exit_code: int) -> None:
        self.writer.close()
