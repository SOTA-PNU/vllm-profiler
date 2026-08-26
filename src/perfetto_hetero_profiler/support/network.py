"""Read-only network preflight helpers."""

from __future__ import annotations

import socket


def port_available(host: str, port: int) -> bool:
    """Return whether a TCP listener can bind the requested host and port."""

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as stream:
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            stream.bind((host, port))
        except OSError:
            return False
    return True
