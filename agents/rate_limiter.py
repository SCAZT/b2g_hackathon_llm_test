"""
Rate Limiter for Phase 2 Chatbot
Adapted from Phase 1 dual_api_threadpool_api.py
Purpose: Control API request rate to avoid exceeding TPM limits
"""
import asyncio
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """Rate limiter queued request data structure"""
    request_id: str
    system_prompt: str
    user_message: str
    enqueue_time: float
    timeout_time: float
    result_future: asyncio.Future


class RateLimitedQueue:
    """
    FIFO Rate Limiter with queue buffering

    Purpose: Ensure API calls never exceed TPM limits by controlling RPM
    Strategy: Buffer requests in queue, release at controlled rate

    Args:
        rpm: Requests per minute limit
        max_queue_size: Maximum queue capacity
        timeout_seconds: Request timeout in queue
    """

    def __init__(self, rpm: int = 250, max_queue_size: int = 1000, timeout_seconds: int = 240):
        self.rpm = rpm
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds
        self.release_interval = 60.0 / rpm  # Seconds between releases

        # FIFO queue
        self.request_queue = deque()
        self.queue_lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_enqueued": 0,
            "total_processed": 0,
            "total_rejected": 0,
            "total_timeout": 0,
            "current_queue_size": 0,
            "max_queue_size_reached": 0
        }
        self.stats_lock = threading.Lock()

        # Control flags
        self.is_running = False
        self.processor_task = None

        logger.info(
            f"🕐 RateLimitedQueue initialized: {rpm} RPM "
            f"({self.release_interval:.2f}s interval), "
            f"{max_queue_size} capacity, {timeout_seconds}s timeout"
        )

    async def start_processor(self):
        """Start queue processor"""
        if self.is_running:
            return

        self.is_running = True
        self.processor_task = asyncio.create_task(self._queue_processor())
        logger.info("🚀 Rate Limiter queue processor started")

    async def stop_processor(self):
        """Stop queue processor and handle remaining requests"""
        logger.info("🛑 Rate Limiter graceful shutdown starting...")
        self.is_running = False

        if self.processor_task:
            await self.processor_task

        # Handle remaining requests
        remaining_count = 0
        with self.queue_lock:
            remaining_count = len(self.request_queue)

        if remaining_count > 0:
            logger.info(f"⏳ Processing {remaining_count} remaining queued requests...")
            # Fast process remaining (no rate limit during shutdown)
            while True:
                with self.queue_lock:
                    if not self.request_queue:
                        break
                    request = self.request_queue.popleft()

                # Mark as timeout
                if not request.result_future.done():
                    request.result_future.set_exception(TimeoutError("System shutdown"))
                    with self.stats_lock:
                        self.stats["total_timeout"] += 1

        logger.info("✅ Rate Limiter graceful shutdown completed")

    async def _queue_processor(self):
        """Queue processor - release 1 request every interval"""
        logger.info(
            f"🔄 Queue processor running "
            f"(releasing 1 request every {self.release_interval:.2f}s)"
        )

        while self.is_running:
            try:
                current_time = time.time()
                request_to_process = None

                # Get one request from queue
                with self.queue_lock:
                    # Check and remove timeout requests
                    while self.request_queue:
                        request = self.request_queue[0]  # Peek front
                        if current_time > request.timeout_time:
                            # Timeout request
                            expired_request = self.request_queue.popleft()
                            if not expired_request.result_future.done():
                                expired_request.result_future.set_exception(
                                    TimeoutError(
                                        f"Request timed out after {self.timeout_seconds}s in queue"
                                    )
                                )
                                with self.stats_lock:
                                    self.stats["total_timeout"] += 1
                                    self.stats["current_queue_size"] = len(self.request_queue)

                                wait_time = current_time - expired_request.enqueue_time
                                logger.warning(
                                    f"⏰ REQUEST_TIMEOUT {expired_request.request_id} "
                                    f"(waited {wait_time:.1f}s)"
                                )
                        else:
                            # Front request not timeout, process it
                            request_to_process = self.request_queue.popleft()
                            with self.stats_lock:
                                self.stats["current_queue_size"] = len(self.request_queue)
                            break

                # Process request
                if request_to_process and not request_to_process.result_future.done():
                    queue_release_time = current_time
                    wait_time = queue_release_time - request_to_process.enqueue_time
                    logger.info(
                        f"📤 QUEUE_RELEASE {request_to_process.request_id} "
                        f"(waited {wait_time:.1f}s, queue:{self.stats['current_queue_size']})"
                    )

                    # Mark request as ready to process, pass timing info
                    request_to_process.result_future.set_result({
                        "status": "ready_to_process",
                        "queue_release_time": queue_release_time,
                        "queue_wait_time": wait_time,
                        "enqueue_time": request_to_process.enqueue_time
                    })

                    with self.stats_lock:
                        self.stats["total_processed"] += 1

                # Wait for next release interval
                await asyncio.sleep(self.release_interval)

            except Exception as e:
                logger.error(f"❌ Queue processor error: {e}")
                await asyncio.sleep(1)  # Brief wait after error

    async def enqueue_request(
        self,
        request_id: str,
        system_prompt: str,
        user_message: str
    ) -> dict:
        """
        Enqueue request and return when ready to process

        Returns:
            dict with timing info when ready to process

        Raises:
            Exception: Queue full (429) or timeout (408)
        """
        enqueue_time = time.time()
        timeout_time = enqueue_time + self.timeout_seconds

        # Create Future to wait for processing
        result_future = asyncio.Future()

        # Check queue capacity
        with self.queue_lock:
            current_size = len(self.request_queue)
            if current_size >= self.max_queue_size:
                # Queue full, reject request
                with self.stats_lock:
                    self.stats["total_rejected"] += 1

                logger.warning(
                    f"🚫 QUEUE_REJECTED {request_id} "
                    f"(queue full: {current_size}/{self.max_queue_size})"
                )
                raise Exception(
                    f"Rate limit queue full ({current_size}/{self.max_queue_size}). "
                    f"Please try again later."
                )

            # Add to queue
            request = QueuedRequest(
                request_id=request_id,
                system_prompt=system_prompt,
                user_message=user_message,
                enqueue_time=enqueue_time,
                timeout_time=timeout_time,
                result_future=result_future
            )

            self.request_queue.append(request)
            queue_size_after = len(self.request_queue)

            with self.stats_lock:
                self.stats["total_enqueued"] += 1
                self.stats["current_queue_size"] = queue_size_after
                if queue_size_after > self.stats["max_queue_size_reached"]:
                    self.stats["max_queue_size_reached"] = queue_size_after

        logger.info(
            f"📥 QUEUE_ENQUEUE {request_id} "
            f"(queue: {queue_size_after}/{self.max_queue_size})"
        )

        # Wait for request to be released by processor
        try:
            queue_timing_info = await result_future
            return queue_timing_info
        except TimeoutError as e:
            logger.error(f"⏰ QUEUE_TIMEOUT {request_id}: {e}")
            raise Exception(f"Request timeout: {e}")

    def get_stats(self):
        """Get queue statistics"""
        with self.stats_lock:
            current_time = time.time()

            # Calculate average wait time (based on current queue)
            avg_wait_time = 0
            with self.queue_lock:
                if self.request_queue:
                    total_wait = sum(
                        current_time - req.enqueue_time
                        for req in self.request_queue
                    )
                    avg_wait_time = total_wait / len(self.request_queue)

            return {
                "rate_limiter_config": {
                    "rpm": self.rpm,
                    "release_interval": self.release_interval,
                    "max_queue_size": self.max_queue_size,
                    "timeout_seconds": self.timeout_seconds
                },
                "queue_stats": {
                    "current_size": self.stats["current_queue_size"],
                    "max_size_reached": self.stats["max_queue_size_reached"],
                    "utilization": (self.stats["current_queue_size"] / self.max_queue_size) * 100,
                    "avg_wait_time_seconds": avg_wait_time
                },
                "request_stats": {
                    "total_enqueued": self.stats["total_enqueued"],
                    "total_processed": self.stats["total_processed"],
                    "total_rejected": self.stats["total_rejected"],
                    "total_timeout": self.stats["total_timeout"],
                    "success_rate": (
                        self.stats["total_processed"] /
                        max(1, self.stats["total_enqueued"])
                    ) * 100,
                    "rejection_rate": (
                        self.stats["total_rejected"] /
                        max(1, self.stats["total_enqueued"] + self.stats["total_rejected"])
                    ) * 100
                },
                "processor_status": {
                    "is_running": self.is_running,
                    "theoretical_max_rpm": self.rpm
                }
            }
