"""Экспортирует чекпоинт из S3 и проверяет ONNX перед публикацией."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime
import torch

from pad_hw.data.cifar import CLASSES
from pad_hw.experiment_config import load_config
from pad_hw.s3 import download_file, make_client
from pad_hw.settings import S3Settings
from pad_hw.train import build_model


def export(checkpoint: str) -> None:
    settings = S3Settings.from_env()
    client = make_client(settings)
    repository = Path("model_repository/cifar_cnn")
    version = repository / "1"
    version.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=version) as directory:
        temp = Path(directory)
        for name in ("config.yaml", "model.pt"):
            download_file(client, settings.bucket, f"{checkpoint}/{name}", temp / name)
        config = load_config(temp / "config.yaml")
        model = build_model(config["model"], len(CLASSES))
        model.load_state_dict(torch.load(temp / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(42)
        example = torch.randn(2, 3, 32, 32, generator=generator)
        exported = temp / "model.onnx"
        torch.onnx.export(
            model,
            (example,),
            str(exported),
            input_names=["input"],
            output_names=["logits"],
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=64)},),
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        session = onnxruntime.InferenceSession(
            str(exported), sess_options=options, providers=["CPUExecutionProvider"]
        )
        with torch.inference_mode():
            for batch in (1, 7, 64):
                inputs = torch.randn(batch, 3, 32, 32, generator=generator)
                (logits,) = session.run(["logits"], {"input": inputs.numpy()})
                np.testing.assert_allclose(logits, model(inputs).numpy(), rtol=1e-4, atol=1e-5)
        dataset = config["data"]["s3_path"].strip("/")
        download_file(
            client, settings.bucket, f"{dataset}/stats.json", Path("data/serving/stats.json")
        )
        exported.replace(version / "model.onnx")
    (repository / "config.pbtxt").write_bytes(Path("configs/triton/config.pbtxt").read_bytes())
    print(f"Exported {checkpoint} from {dataset}: {version / 'model.onnx'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    export(load_config(args.config)["checkpoint"].strip("/"))


if __name__ == "__main__":
    main()
