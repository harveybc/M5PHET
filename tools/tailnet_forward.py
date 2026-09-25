#!/usr/bin/env python3
"""Forward one tailnet-facing TCP port to the loopback workbench, so a phone on the tailnet can open it.

    M5PHET_CHAT_TAILNET_IP=<this host's tailnet address> tools/tailnet_forward.py [--port 8765] [--target 127.0.0.1:8765]

The workbench keeps listening on 127.0.0.1 only; this relay listens on the tailnet address, which only devices of the
owner's tailnet can reach (WireGuard-encrypted on the wire). The workbench's own Host allow-list and access token
still apply behind it. It is the interim for `sudo tailscale serve --https=443 http://127.0.0.1:8765`, which needs root
and adds a certificate; nothing here contains an address -- the address is the operator's environment.
"""
import argparse
import asyncio
import os
import sys


async def _pump(reader, writer):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:                                                     # noqa: BLE001
            pass


async def _serve(bind, port, target_host, target_port):
    async def handle(client_reader, client_writer):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(_pump(client_reader, upstream_writer), _pump(upstream_reader, client_writer))

    server = await asyncio.start_server(handle, bind, port, reuse_address=True)
    async with server:
        await server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default=os.environ.get("M5PHET_CHAT_TAILNET_IP", ""))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target", default="127.0.0.1:8765")
    args = parser.parse_args(argv)
    if not args.bind:
        sys.exit("set M5PHET_CHAT_TAILNET_IP or pass --bind; refusing to listen on every interface")
    if args.bind in ("0.0.0.0", "::"):
        sys.exit("refusing to listen on every interface; bind the tailnet address only")
    host, _, port = args.target.rpartition(":")
    asyncio.run(_serve(args.bind, args.port, host, int(port)))


if __name__ == "__main__":
    main()
