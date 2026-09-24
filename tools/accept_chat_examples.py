"""Run configured examples through HTTP. Does not train or authorize a broker."""
import argparse
import json
import time
import uuid
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", action="append")
    args = parser.parse_args()
    results = []
    with httpx.Client(base_url=args.url, timeout=30) as client:
        catalog = client.get("/api/catalog").raise_for_status().json()
        for example in catalog["examples"]:
            if args.provider and example["config"]["provider"] not in args.provider:
                continue
            config = catalog["defaults"] | example["config"]
            chat = client.post("/api/chats", json={"title": example["title"][:120]}).raise_for_status().json()
            cid = chat["id"]
            client.patch(f"/api/chats/{cid}", json={"config": config}).raise_for_status()
            data = example["data"]
            filename = "context.txt" if isinstance(data, str) else "data.json"
            raw = data if isinstance(data, str) else json.dumps(data)
            upload = client.post(f"/api/chats/{cid}/files", files={"file": (filename, raw.encode())}).raise_for_status().json()
            client.post(f"/api/chats/{cid}/messages", json={"prompt": example["prompt"], "file_ids": [upload["id"]], "client_id": uuid.uuid4().hex}).raise_for_status()
            started = time.monotonic()
            while time.monotonic() - started < 190:
                messages = client.get(f"/api/chats/{cid}").raise_for_status().json()["messages"]
                final = messages[-1]
                if final["status"] != "RUNNING":
                    break
                time.sleep(.3)
            detail = final["detail"]
            result = {"provider": config["provider"], "chat_id": cid, "status": final["status"],
                      "content": final["content"], "result": detail.get("result"),
                      "elapsed_seconds": detail.get("elapsed_seconds"), "profile": detail.get("profile"),
                      "input_sha256": upload["sha256"]}
            results.append(result)
            print(json.dumps({"provider": result["provider"], "status": result["status"], "seconds": result["elapsed_seconds"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    assert results and all(r["status"] == "OK" for r in results), "Inspect persisted refusal(s), not just this exit status"


if __name__ == "__main__":
    main()
