"""Bounded one-shot transport to an existing private worker, not a model server.

The administrator fixes interpreter, environment and device in the worker wrapper.
User requests cannot choose commands, paths or hosts. Every execution still goes
through the installed M5PHET Registry. stdout is a single JSON protocol response.
"""
import contextlib
import json
import hashlib
import os
import subprocess
import sys


def admit_gpu():
    if os.getenv("NEWS_SIGNAL_DEVICE", "cpu") == "cpu":
        return
    uuid = os.environ["NEWS_SIGNAL_GPU_UUID"]
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,temperature.gpu,memory.free", "--format=csv,noheader,nounits"
    ], text=True, timeout=10).splitlines()
    match = [row.split(",") for row in rows if row.split(",")[0].strip() == uuid]
    if len(match) != 1 or int(match[0][1]) > 75 or int(match[0][2]) < 3500:
        raise RuntimeError("GPU admission refused: UUID, temperature or free VRAM")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True, timeout=10)
    if any(row.split(",")[0].strip() == uuid for row in processes.splitlines()):
        raise RuntimeError("GPU admission refused: another compute process owns this device")
    with open("/proc/meminfo") as stream:
        available = next(int(line.split()[1]) for line in stream if line.startswith("MemAvailable:"))
    if available < 4 * 1024 * 1024:
        raise RuntimeError("Worker admission refused: less than 4 GiB available RAM")


def main():
    try:
        command = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            from m5phet.runtime import Registry, run
            registry = Registry()
            report = registry.load_entry_points()
            if command.get("action") == "describe":
                result = {"capabilities": registry.capabilities("laya_news"), "discovery": report}
            elif command.get("action") == "infer":
                request = command["request"]
                if request.get("provider_ref") != "laya_news" or request.get("operation") != "infer":
                    raise ValueError("Worker permits laya_news inference only")
                admit_gpu()
                result = run(request, registry)
                provider = registry.get("laya_news")
                result["worker_observation"] = {
                    "device_uuid": os.environ.get("NEWS_SIGNAL_GPU_UUID"),
                    "model": provider.identity_for(request.get("fitted_state_ref")),
                    "provider_source_sha256": hashlib.sha256(open(sys.modules[type(provider).__module__].__file__, "rb").read()).hexdigest(),
                }
            elif command.get("action") == "task":
                # the question envelope, on the same contract the interface uses: one shape on both ends of the wire
                from m5phet.questions import run_task
                task = command["task"]
                if not isinstance(task, dict) or task.get("area") != "classification":
                    raise ValueError("Worker permits classification envelopes only")
                admit_gpu()
                result = run_task(task, registry, data=command.get("data"))
                provider = registry.get("laya_news")
                result["worker_observation"] = {
                    "device_uuid": os.environ.get("NEWS_SIGNAL_GPU_UUID"),
                    "model": provider.identity_for(result.get("state_ref")) if result.get("state_ref") else None,
                    "provider_source_sha256": hashlib.sha256(open(sys.modules[type(provider).__module__].__file__, "rb").read()).hexdigest(),
                }
            else:
                raise ValueError("Unknown worker action")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print(json.dumps({"transport_error": f"{type(error).__name__}: {error}"}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
