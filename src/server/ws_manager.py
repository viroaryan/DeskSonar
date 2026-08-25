"""
DeskSonar WebSocket Connection & Telemetry Broadcast Manager
"""
import json
import asyncio
from typing import Dict, Any, List, Set, Optional
from fastapi import WebSocket


class ConnectionManager:
    """
    Manages active dashboard and phone companion WebSocket connections.
    """

    def __init__(self):
        self.dashboard_clients: Set[WebSocket] = set()
        self.phone_clients: Set[WebSocket] = set()
        self.latest_telemetry: Dict[str, Any] = {}

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_clients.add(websocket)

    def disconnect_dashboard(self, websocket: WebSocket):
        self.dashboard_clients.discard(websocket)

    async def connect_phone(self, websocket: WebSocket):
        await websocket.accept()
        self.phone_clients.add(websocket)

    def disconnect_phone(self, websocket: WebSocket):
        self.phone_clients.discard(websocket)

    async def broadcast_telemetry(self, telemetry_data: Dict[str, Any]):
        """
        Broadcasts high-rate radar frame metrics to all connected visualizer dashboards.
        """
        self.latest_telemetry = telemetry_data
        if not self.dashboard_clients:
            return

        dead_sockets = set()
        payload = json.dumps(telemetry_data)

        for ws in self.dashboard_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead_sockets.add(ws)

        for dead in dead_sockets:
            self.dashboard_clients.discard(dead)

    async def broadcast_gesture(self, gesture_data: Dict[str, Any]):
        """
        Sends recognized gesture event to all clients.
        """
        payload = json.dumps({"type": "gesture_event", "data": gesture_data})
        for ws in list(self.dashboard_clients) + list(self.phone_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                pass
