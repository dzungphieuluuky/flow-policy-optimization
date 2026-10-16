from __future__ import annotations


class TensorboardBackend:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, flush_secs=30)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            if value is None:
                continue
            self.writer.add_scalar(key, float(value), step)

    def finish(self, exit_code: int) -> None:
        self.writer.close()
