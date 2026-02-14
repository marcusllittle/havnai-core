"""Server-Sent Events (SSE) broker for HavnAI real-time updates."""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Generator, List, Optional


class SSEBroker:
    """Thread-safe broker for Server-Sent Events.

    Manages client connections per channel and broadcasts events.
    Clients that fall behind (full queue) are automatically dropped.
    """

    def __init__(self, max_queue_size: int = 50):
        self._clients: Dict[str, List[queue.Queue]] = {}  # channel -> list of queues
        self._lock = threading.Lock()
        self._max_queue = max_queue_size

    def subscribe(self, channel: str) -> queue.Queue:
        """Subscribe to a channel.  Returns a Queue that will receive SSE messages."""
        q: queue.Queue = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._clients.setdefault(channel, []).append(q)
        return q

    def unsubscribe(self, channel: str, q: queue.Queue) -> None:
        """Remove a client queue from a channel."""
        with self._lock:
            clients = self._clients.get(channel, [])
            if q in clients:
                clients.remove(q)

    def publish(self, channel: str, event: str, data: dict) -> None:
        """Broadcast an event to all subscribers of a channel.

        Clients whose queues are full are silently dropped.
        """
        message = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self._lock:
            dead: List[queue.Queue] = []
            for q in self._clients.get(channel, []):
                try:
                    q.put_nowait(message)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._clients[channel].remove(q)
                except (ValueError, KeyError):
                    pass

    def stream(self, channel: str) -> Generator[str, None, None]:
        """Yield SSE messages for a given channel.

        Sends a keepalive comment every 30 seconds to prevent connection
        timeouts from proxies and load balancers.
        """
        q = self.subscribe(channel)
        try:
            # Send initial keepalive so the client knows the connection is live
            yield ": connected\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            self.unsubscribe(channel, q)

    @property
    def client_count(self) -> int:
        """Return total number of connected clients across all channels."""
        with self._lock:
            return sum(len(clients) for clients in self._clients.values())

    def channel_count(self, channel: str) -> int:
        """Return number of connected clients for a specific channel."""
        with self._lock:
            return len(self._clients.get(channel, []))
