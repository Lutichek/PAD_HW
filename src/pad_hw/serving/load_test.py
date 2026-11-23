"""Измеряет задержку и пропускную способность POST /predict."""

from __future__ import annotations

import argparse
import os
import platform
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np

from pad_hw.serving.dataset import load_test_files
from pad_hw.utils import write_json

Payload = tuple[str, bytes, str]


def measure(
    client: httpx.Client, payloads: list[Payload], workers: int
) -> tuple[list[dict[str, Any]], float]:
    def send(worker: int) -> list[dict[str, Any]]:
        results = []
        for index in range(worker, len(payloads), workers):
            start = perf_counter()
            try:
                response = client.post("/predict", files={"file": payloads[index]})
                code, error = response.status_code, None
            except httpx.RequestError as exc:
                code, error = None, str(exc)
            results.append(
                {
                    "index": index,
                    "worker": worker + 1,
                    "file": payloads[index][0],
                    "latency_ms": (perf_counter() - start) * 1000,
                    "status_code": code,
                    "error": error,
                }
            )
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        start = perf_counter()
        measurements = [row for rows in pool.map(send, range(workers)) for row in rows]
        elapsed = perf_counter() - start
    return measurements, elapsed


def counters(client: httpx.Client, model: str, version: str) -> dict[str, float]:
    response = client.get("/metrics")
    response.raise_for_status()
    values = {}
    for name, labels, value in re.findall(
        r"^(nv_inference_\w+)\{([^}]+)\}\s+([\deE+.-]+)", response.text, re.MULTILINE
    ):
        tags = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        if tags.get("model") == model and tags.get("version") == version:
            values[name] = float(value)
    return values


def describe_cpu() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    logical = os.cpu_count()
    if cpuinfo.is_file():
        blocks = [block for block in cpuinfo.read_text().split("\n\n") if block.strip()]
        fields = [
            dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            for block in blocks
        ]
        info = [{key.strip(): value.strip() for key, value in block.items()} for block in fields]
        cores = {
            (row["physical id"], row["core id"])
            for row in info
            if "core id" in row and "physical id" in row
        }
        name = info[0].get("model name", platform.machine())
        if cores:
            return f"{name}, {len(cores)} физических ядер, {logical} логических CPU"
        return f"{name}, {logical} логических CPU"
    return f"{platform.processor() or platform.machine()}, {logical} логических CPU"


def report(rows: list[dict[str, Any]], conditions: list[str], raw_path: Path) -> str:
    lines = [
        "# Нагрузочный тест",
        "",
        "## Условия замера",
        "",
        *(f"- {condition}" for condition in conditions),
        f"- Сырые измерения: `{raw_path.as_posix()}`.",
        "",
        "## Результаты",
        "",
        "| Workers | Запросов | Ошибок | RPS | Среднее, мс | Медиана, мс | p90, мс | p95, мс |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workers} | {requests} | {errors} | {rps:.2f} | {mean:.2f} |"
            " {median:.2f} | {p90:.2f} | {p95:.2f} |".format(**row)
        )
    first, last = rows[0], rows[-1]
    peak = max(rows, key=lambda row: row["rps"])
    near_peak = next(row for row in rows if row["rps"] >= 0.95 * peak["rps"])
    saturation = (
        f"Прирост практически прекращается с {near_peak['workers']} workers "
        "(достигнуто не менее 95 % максимального RPS)."
        if near_peak is not last
        else "Плато по критерию 95 % от максимума в измеренном диапазоне не наблюдалось."
    )
    lines.extend(
        [
            "",
            "## Выводы",
            "",
            (
                f"От {first['workers']} до {last['workers']} workers RPS изменился с "
                f"{first['rps']:.2f} до {last['rps']:.2f}, а p95 — с "
                f"{first['p95']:.2f} до {last['p95']:.2f} мс."
            ),
            f"Максимум составил {peak['rps']:.2f} RPS при {peak['workers']} workers. {saturation}",
            (
                f"По счётчикам Triton средний батч изменился с {first['batch']:.2f} до "
                f"{last['batch']:.2f}, а среднее время в очереди — с "
                f"{first['queue_ms']:.2f} до {last['queue_ms']:.2f} мс. "
                "Запросы конкурируют за один CPU-экземпляр модели и обработку в HTTP-обёртке."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def load_test(args: argparse.Namespace) -> int:
    samples, reference = load_test_files(args.config)
    rng = random.Random(args.seed)
    chosen = rng.choices([path for path, _ in samples], k=args.requests + args.warmup)
    payloads = [(path.name, path.read_bytes(), "image/png") for path in chosen]
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []

    with (
        httpx.Client(base_url=args.url, timeout=15) as client,
        httpx.Client(base_url=args.triton_url, timeout=5) as triton,
        httpx.Client(base_url=args.metrics_url, timeout=5) as metrics,
    ):
        client.get("/health").raise_for_status()
        probe = client.post("/predict", files={"file": payloads[0]})
        probe.raise_for_status()
        model, version = (probe.json()[key] for key in ("model_name", "model_version"))
        server = triton.get("/v2").raise_for_status().json()
        config = triton.get(f"/v2/models/{model}/versions/{version}/config")
        config.raise_for_status()
        settings = config.json()
        for workers in args.workers:
            measure(client, payloads[: args.warmup], workers)
            before = counters(metrics, model, version)
            measurements, elapsed = measure(client, payloads[args.warmup :], workers)
            after = counters(metrics, model, version)
            delta = {key: after[key] - before.get(key, 0) for key in after}
            latencies = [row["latency_ms"] for row in measurements if row["status_code"] == 200]
            median, p90, p95 = np.percentile(latencies, [50, 90, 95]) if latencies else [0.0] * 3
            inferred = delta.get("nv_inference_count", 0)
            executions = delta.get("nv_inference_exec_count", 0)
            rows.append(
                {
                    "workers": workers,
                    "requests": len(measurements),
                    "errors": len(measurements) - len(latencies),
                    "rps": len(latencies) / elapsed,
                    "mean": float(np.mean(latencies)) if latencies else 0.0,
                    "median": median,
                    "p90": p90,
                    "p95": p95,
                    "batch": inferred / executions if executions else 0.0,
                    "queue_ms": delta.get("nv_inference_queue_duration_us", 0) / inferred / 1000
                    if inferred
                    else 0.0,
                }
            )
            series.append(
                {"workers": workers, "elapsed_seconds": elapsed, "requests": measurements}
            )
            print(f"Workers: {workers}, RPS: {rows[-1]['rps']:.2f}", flush=True)

    count = sum(int(group.get("count", 1)) for group in settings["instance_group"])
    batch_size = settings["max_batch_size"]
    queue = settings["dynamic_batching"].get("max_queue_delay_microseconds", 0)
    conditions = [
        f"Процессор: {describe_cpu()}.",
        f"ОС клиента: {platform.platform()}. {args.topology}",
        f"Triton {server['version']}, модель `{model}` версии {version}, {count} CPU-экземпляр.",
        f"Dynamic batching: max_batch_size={batch_size}, задержка очереди до {queue} мкс.",
        (
            f"Источник: `{reference['data']}`, seed {args.seed}, прогрев {args.warmup} запросов "
            "на режим. Задержки в таблице рассчитаны по успешным ответам."
        ),
    ]
    write_json(args.raw, {"conditions": conditions, "series": series})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(rows, conditions, args.raw), encoding="utf-8")
    return int(any(row["errors"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8002")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw", type=Path, default=Path("outputs/load_test.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/load_test.md"))
    parser.add_argument(
        "--topology", default="Клиент и Docker-контейнеры сервиса на одной машине."
    )
    args = parser.parse_args()
    if min(args.workers) < 1 or args.requests < 1000 or args.warmup < 1:
        parser.error("Specify positive workers and warmup and at least 1000 measured requests")
    raise SystemExit(load_test(args))


if __name__ == "__main__":
    main()
