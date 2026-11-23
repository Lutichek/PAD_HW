"""HTTP-интерфейс модели CIFAR-10."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from pad_hw.data.cifar import CLASSES
from pad_hw.utils import read_json

MAX_IMAGE_BYTES = 2 * 1024 * 1024
Pixels = npt.NDArray[np.float32]


class Prediction(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float]
    model_name: str
    model_version: str


class Triton:
    def __init__(self, client: httpx.AsyncClient, stats_path: Path) -> None:
        self.client = client
        self.name = os.environ.get("TRITON_MODEL", "cifar_cnn")
        self.version = os.environ.get("TRITON_MODEL_VERSION", "1")
        self.path = f"/v2/models/{self.name}/versions/{self.version}"
        stats = read_json(stats_path)
        self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError(f"Invalid normalization statistics in {stats_path}")

    async def predict(self, pixels: Pixels) -> Prediction:
        tensor = ((pixels / 255 - self.mean) / self.std).astype("<f4")
        header = json.dumps(
            {
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 3, 32, 32],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor.nbytes},
                    }
                ],
                "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
            },
            separators=(",", ":"),
        ).encode()
        response = await self.client.post(
            f"{self.path}/infer",
            content=header + tensor.tobytes(order="C"),
            headers={
                "Inference-Header-Content-Length": str(len(header)),
                "Content-Type": "application/octet-stream",
            },
        )
        response.raise_for_status()
        result = response.json()
        logits = np.asarray(result["outputs"][0]["data"], dtype=np.float64)
        if logits.shape != (len(CLASSES),) or not np.isfinite(logits).all():
            raise ValueError("Invalid logits returned by Triton")
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        class_id = int(probabilities.argmax())
        return Prediction(
            class_id=class_id,
            class_name=CLASSES[class_id],
            probabilities=probabilities.tolist(),
            model_name=result["model_name"],
            model_version=result["model_version"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(
        base_url=os.environ.get("TRITON_URL", "http://localhost:8000"), timeout=5.0
    ) as client:
        app.state.triton = Triton(
            client, Path(os.environ.get("STATS_PATH", "data/serving/stats.json"))
        )
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    triton: Triton = request.app.state.triton
    try:
        (await triton.client.get("/v2/health/ready")).raise_for_status()
        (await triton.client.get(f"{triton.path}/ready")).raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(503, "Triton or model is not ready") from error
    return {"status": "ready"}


def decode_image(raw: bytes) -> Pixels:
    if not raw:
        raise HTTPException(400, "Empty image file")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds 2 MiB")
    try:
        with Image.open(BytesIO(raw), formats=["PNG", "JPEG"]) as image:
            if image.width * image.height > 16_000_000:
                raise HTTPException(413, "Image dimensions are too large")
            rgb = image.convert("RGB")
            if rgb.size != (32, 32):
                rgb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None]
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(415, "Expected a valid PNG or JPEG image") from error


@app.post("/predict")
async def predict(file: UploadFile, request: Request) -> Prediction:
    pixels = decode_image(await file.read(MAX_IMAGE_BYTES + 1))
    triton: Triton = request.app.state.triton
    try:
        return await triton.predict(pixels)
    except httpx.HTTPError as error:
        raise HTTPException(503, "Triton inference is unavailable") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise HTTPException(502, "Invalid response from Triton") from error
