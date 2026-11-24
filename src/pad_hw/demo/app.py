"""Раздаёт интерфейс и передаёт изображения в API из ДЗ №4."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).with_name("static")
MAX_IMAGE_BYTES = 2 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(base_url=os.environ["DEMO_API_URL"], timeout=8.0) as client:
        app.state.upstream = client
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/predict")
async def predict(file: UploadFile, request: Request) -> JSONResponse:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Файл больше 2 МБ. Выберите изображение меньшего размера.")
    client: httpx.AsyncClient = request.app.state.upstream
    try:
        async with asyncio.timeout(8):
            response = await client.post(
                "/predict",
                files={"file": (file.filename or "image", data, file.content_type or "image/png")},
            )
    except (TimeoutError, httpx.TimeoutException) as error:
        raise HTTPException(504, "Сервис не ответил вовремя. Попробуйте ещё раз.") from error
    except httpx.RequestError as error:
        raise HTTPException(503, "Сервис предсказаний недоступен. Попробуйте позже.") from error
    if response.is_client_error:
        raise HTTPException(response.status_code, "Выберите непустое изображение PNG или JPEG.")
    if response.is_server_error:
        raise HTTPException(503, "Сервис предсказаний недоступен. Попробуйте позже.")
    try:
        result = response.json()
    except ValueError as error:
        raise HTTPException(502, "Сервис вернул неожиданный ответ. Попробуйте позже.") from error
    return JSONResponse(result)
