
import asyncio
import logging
import traceback

logger = logging.getLogger(__name__)


class EmailMonitor:

    def __init__(
        self,
        scan_callback,
        employee_id,
        interval_seconds=120,
    ):
        self.scan_callback = scan_callback
        self.employee_id = employee_id
        self.interval_seconds = interval_seconds

        self._stop_event = asyncio.Event()
        self._task = None

    async def start(self):

        if self._task is not None and not self._task.done():
            print("[EMAIL MONITOR] Already running.", flush=True)
            return

        self._stop_event.clear()

        print(
            f"[EMAIL MONITOR] Starting for {self.employee_id}",
            flush=True,
        )

        self._task = asyncio.create_task(
            self._monitor_loop()
        )

    async def _monitor_loop(self):

        while not self._stop_event.is_set():

            try:
                print(
                    "[EMAIL MONITOR] Starting Gmail scan...",
                    flush=True,
                )

                result = await asyncio.to_thread(
                    self.scan_callback,
                    self.employee_id,
                )

                print(
                    f"[EMAIL MONITOR] Scan completed: {result}",
                    flush=True,
                )

            except Exception:
                print(
                    "[EMAIL MONITOR] Scan failed:",
                    flush=True,
                )
                traceback.print_exc()

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )

            except asyncio.TimeoutError:
                continue

        print(
            "[EMAIL MONITOR] Monitoring loop stopped.",
            flush=True,
        )

    async def stop(self):

        if self._task is None:
            return

        print(
            "[EMAIL MONITOR] Stopping...",
            flush=True,
        )

        self._stop_event.set()

        await self._task

        self._task = None

        print(
            "[EMAIL MONITOR] Stopped successfully.",
            flush=True,
        )
