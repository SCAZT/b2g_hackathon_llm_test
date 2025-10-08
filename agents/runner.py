"""
AI Service Manager for Phase 2 Chatbot
Three-API Architecture + Thread Pool + Dual Rate Limiter

Architecture:
- MAIN_API: Chat main (83.3% via 5:1 distribution)
- BACKUP_API: Chat backup (16.7% via 5:1 distribution)
- MEMORY_API: Memory dedicated (100% of memory calls)
- Thread Pool: 300 workers for concurrent processing
- Dual Rate Limiter: Chat (250 RPM) + Memory (400 RPM)
"""
import os
import asyncio
import time
import threading
import logging
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Tuple
from .rate_limiter import RateLimitedQueue

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class AIServiceManager:
    """
    Three-API Service Manager with Thread Pool and Dual Rate Limiter

    Key Features:
    - Zero TPM errors (Rate Limiter ensures never exceed limits)
    - High concurrency (Thread pool handles 300 concurrent requests)
    - Clean separation (Chat and Memory use dedicated APIs)
    - Fallback protection (Memory can fallback to BACKUP if needed)
    """

    def __init__(self):
        # =================== Three API Clients ===================
        self.main_api_key = os.getenv("MAIN_API_KEY")
        self.backup_api_key = os.getenv("BACKUP_API_KEY")
        self.memory_api_key = os.getenv("MEMORY_API_KEY")

        # Validate configuration
        if not self.main_api_key:
            raise ValueError("MAIN_API_KEY is required")
        if not self.backup_api_key:
            raise ValueError("BACKUP_API_KEY is required")
        if not self.memory_api_key:
            logger.warning("⚠️ MEMORY_API_KEY not configured, will fallback to BACKUP")

        # Create OpenAI clients
        self.main_client = OpenAI(api_key=self.main_api_key)
        self.backup_client = OpenAI(api_key=self.backup_api_key)
        self.memory_client = OpenAI(api_key=self.memory_api_key) if self.memory_api_key else None

        logger.info(f"✅ MAIN API: ...{self.main_api_key[-4:]}")
        logger.info(f"✅ BACKUP API: ...{self.backup_api_key[-4:]}")
        if self.memory_client:
            logger.info(f"✅ MEMORY API: ...{self.memory_api_key[-4:]}")

        # =================== Thread Pool ===================
        max_workers = int(os.getenv("THREAD_POOL_MAX_WORKERS", "300"))
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ChatbotOpenAI"
        )
        logger.info(f"🧵 Thread Pool: {max_workers} workers initialized")

        # =================== Dual Rate Limiter ===================
        # Chat Rate Limiter (250 RPM default)
        chat_rpm = int(os.getenv("CHAT_RPM_LIMIT", "250"))
        chat_queue_size = int(os.getenv("CHAT_QUEUE_SIZE", "1000"))
        chat_timeout = int(os.getenv("CHAT_TIMEOUT_SECONDS", "240"))

        self.chat_rate_limiter = RateLimitedQueue(
            rpm=chat_rpm,
            max_queue_size=chat_queue_size,
            timeout_seconds=chat_timeout
        )
        logger.info(f"📊 Chat Rate Limiter: {chat_rpm} RPM, {chat_queue_size} queue")

        # Memory Rate Limiter (400 RPM default)
        memory_rpm = int(os.getenv("MEMORY_RPM_LIMIT", "400"))
        memory_queue_size = int(os.getenv("MEMORY_QUEUE_SIZE", "500"))
        memory_timeout = int(os.getenv("MEMORY_TIMEOUT_SECONDS", "120"))

        self.memory_rate_limiter = RateLimitedQueue(
            rpm=memory_rpm,
            max_queue_size=memory_queue_size,
            timeout_seconds=memory_timeout
        )
        logger.info(f"🧠 Memory Rate Limiter: {memory_rpm} RPM, {memory_queue_size} queue")

        # =================== 5:1 Distribution Counter ===================
        self.chat_counter = 0
        self.counter_lock = threading.Lock()

        # =================== Statistics ===================
        self.stats = {
            "main_api": {"total_calls": 0, "success": 0, "failures": 0},
            "backup_api": {"total_calls": 0, "success": 0, "failures": 0},
            "memory_api": {"total_calls": 0, "success": 0, "failures": 0},
        }
        self.stats_lock = threading.Lock()

        # =================== Model Configuration ===================
        self.chat_model = "gpt-4o"
        self.memory_model = "gpt-4o-mini"  # Cheaper for memory extraction
        self.embedding_model = "text-embedding-3-small"

        # =================== Auto-start Flag ===================
        self._started = False
        self._start_lock = asyncio.Lock()

        logger.info("🚀 AIServiceManager initialized (Three-API + Thread Pool + Dual Rate Limiter)")

    async def _ensure_started(self):
        """Ensure Rate Limiter processors are started (lazy initialization)"""
        if not self._started:
            async with self._start_lock:
                if not self._started:  # Double-check after acquiring lock
                    await self.chat_rate_limiter.start_processor()
                    await self.memory_rate_limiter.start_processor()
                    self._started = True
                    logger.info("✅ Rate Limiter processors auto-started")

    async def start(self):
        """Manually start Rate Limiter processors"""
        await self._ensure_started()

    async def stop(self):
        """Stop Rate Limiter processors gracefully"""
        await self.chat_rate_limiter.stop_processor()
        await self.memory_rate_limiter.stop_processor()
        self.executor.shutdown(wait=True)
        logger.info("✅ AIServiceManager shutdown complete")

    def _get_chat_client(self) -> Tuple[str, OpenAI]:
        """
        Get chat client using 5:1 distribution
        Returns: (api_type, client)
        """
        with self.counter_lock:
            self.chat_counter += 1
            position = self.chat_counter % 6

        # Every 6th request uses backup, first 5 use main
        if position == 0:
            return "backup", self.backup_client
        else:
            return "main", self.main_client

    def _get_memory_client(self) -> Tuple[str, OpenAI]:
        """
        Get memory client (dedicated MEMORY_API or fallback to BACKUP)
        Returns: (api_type, client)
        """
        if self.memory_client:
            return "memory", self.memory_client
        else:
            logger.warning("Memory API unavailable, using BACKUP")
            return "backup_fallback", self.backup_client

    def _sync_openai_call(
        self,
        client: OpenAI,
        messages: List[Dict],
        model: str,
        call_id: str,
        api_type: str
    ) -> Dict:
        """
        Synchronous OpenAI API call (runs in thread pool)

        Args:
            client: OpenAI client instance
            messages: Chat messages
            model: Model name
            call_id: Unique call identifier
            api_type: "main", "backup", or "memory"

        Returns:
            dict with content, tokens, timing info
        """
        start_time = time.time()

        try:
            logger.info(f"🔧 {api_type.upper()}_CALL {call_id} starting...")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )

            processing_time = time.time() - start_time
            content = response.choices[0].message.content.strip()

            total_tokens = response.usage.total_tokens if response.usage else 0
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            logger.info(
                f"✅ {api_type.upper()}_CALL {call_id} completed: "
                f"{total_tokens} tokens ({input_tokens}in+{output_tokens}out), "
                f"{processing_time:.2f}s"
            )

            # Update statistics
            with self.stats_lock:
                self.stats[f"{api_type}_api"]["total_calls"] += 1
                self.stats[f"{api_type}_api"]["success"] += 1

            return {
                "content": content,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "processing_time": processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ {api_type.upper()}_CALL {call_id} failed: {e} ({processing_time:.2f}s)")

            # Update statistics
            with self.stats_lock:
                self.stats[f"{api_type}_api"]["total_calls"] += 1
                self.stats[f"{api_type}_api"]["failures"] += 1

            raise e

    async def run_agent(
        self,
        agent,
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """
        Execute agent with chat Rate Limiter + thread pool + 5:1 distribution

        Flow:
        1. Enqueue in chat Rate Limiter (250 RPM control)
        2. Wait for queue release
        3. Execute in thread pool with 5:1 API selection
        4. Return response

        Args:
            agent: Agent instance with instructions
            prompt: User prompt/message
            model: OpenAI model (default: gpt-4o)

        Returns:
            str: Agent's response
        """
        try:
            # Ensure Rate Limiter processors are started
            await self._ensure_started()

            # Use default chat model if not specified
            if model is None:
                model = self.chat_model

            # Generate request ID
            request_id = f"chat_req_{int(time.time() * 1000)}"

            # Step 1: Enqueue in Rate Limiter
            queue_timing_info = await self.chat_rate_limiter.enqueue_request(
                request_id=request_id,
                system_prompt=agent.instructions or "",
                user_message=prompt
            )

            # Step 2: Get API client using 5:1 distribution
            api_type, client = self._get_chat_client()

            # Step 3: Prepare messages
            messages = [
                {"role": "system", "content": agent.instructions or ""},
                {"role": "user", "content": prompt}
            ]

            # Step 4: Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_openai_call,
                client,
                messages,
                model,
                request_id,
                api_type
            )

            return result["content"]

        except Exception as e:
            logger.error(f"❌ run_agent failed: {e}")
            return f"[OpenAI Error] {str(e)}"

    async def run_agent_stream(self, agent, prompt: str, model: Optional[str] = None):
        """
        Execute agent with streaming response
        Note: Streaming still goes through Rate Limiter but uses async API

        Args:
            agent: Agent instance
            prompt: User prompt
            model: OpenAI model

        Yields:
            str: Response chunks
        """
        try:
            # Ensure Rate Limiter processors are started
            await self._ensure_started()

            if model is None:
                model = self.chat_model

            # Generate request ID
            request_id = f"chat_stream_req_{int(time.time() * 1000)}"

            # Enqueue in Rate Limiter
            queue_timing_info = await self.chat_rate_limiter.enqueue_request(
                request_id=request_id,
                system_prompt=agent.instructions or "",
                user_message=prompt
            )

            # Get API client
            api_type, client = self._get_chat_client()

            # Update statistics
            with self.stats_lock:
                self.stats[f"{api_type}_api"]["total_calls"] += 1

            # Stream response
            messages = [
                {"role": "system", "content": agent.instructions or ""},
                {"role": "user", "content": prompt}
            ]

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=4000
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

            # Update statistics
            with self.stats_lock:
                self.stats[f"{api_type}_api"]["success"] += 1

        except Exception as e:
            logger.error(f"❌ run_agent_stream failed: {e}")
            yield f"[OpenAI Error] {str(e)}"

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Memory Rate Limiter + MEMORY_API

        Flow:
        1. Enqueue in memory Rate Limiter (400 RPM control)
        2. Wait for queue release
        3. Execute in thread pool with MEMORY_API
        4. Return embedding

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector (1536 dimensions)
        """
        try:
            # Ensure Rate Limiter processors are started
            await self._ensure_started()

            # Generate request ID
            request_id = f"embed_req_{int(time.time() * 1000)}"

            # Step 1: Enqueue in Memory Rate Limiter
            queue_timing_info = await self.memory_rate_limiter.enqueue_request(
                request_id=request_id,
                system_prompt="",  # Not used for embedding
                user_message=text
            )

            # Step 2: Get memory client
            api_type, client = self._get_memory_client()

            # Step 3: Execute in thread pool
            loop = asyncio.get_event_loop()

            def _sync_embedding_call():
                start_time = time.time()
                try:
                    response = client.embeddings.create(
                        model=self.embedding_model,
                        input=text
                    )
                    processing_time = time.time() - start_time

                    logger.info(
                        f"✅ {api_type.upper()}_EMBEDDING {request_id} "
                        f"completed: {processing_time:.2f}s"
                    )

                    # Update statistics
                    with self.stats_lock:
                        self.stats[f"{api_type}_api"]["total_calls"] += 1
                        self.stats[f"{api_type}_api"]["success"] += 1

                    return response.data[0].embedding

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(
                        f"❌ {api_type.upper()}_EMBEDDING {request_id} "
                        f"failed: {e} ({processing_time:.2f}s)"
                    )

                    # Update statistics
                    with self.stats_lock:
                        self.stats[f"{api_type}_api"]["total_calls"] += 1
                        self.stats[f"{api_type}_api"]["failures"] += 1

                    raise e

            embedding = await loop.run_in_executor(self.executor, _sync_embedding_call)
            return embedding

        except Exception as e:
            logger.error(f"❌ generate_embedding failed: {e}")
            # Return default embedding (all zeros) as fallback
            return [0.0] * 1536

    async def extract_memory_content(
        self,
        conversation_text: str,
        memory_type: str
    ) -> str:
        """
        Extract memory content using Memory Rate Limiter + MEMORY_API

        Flow:
        1. Enqueue in memory Rate Limiter (400 RPM control)
        2. Wait for queue release
        3. Execute in thread pool with MEMORY_API (gpt-4o-mini for cost)
        4. Return extracted summary

        Args:
            conversation_text: Full conversation text
            memory_type: "round_summary", "conversation_chunk", etc.

        Returns:
            str: Extracted memory summary
        """
        try:
            # Generate request ID
            request_id = f"memory_extract_req_{int(time.time() * 1000)}"

            # Create prompt based on memory type
            if memory_type == "round_summary":
                system_prompt = """Extract the key insights from this round of chat conversation. Focus on:
- Main topics discussed
- Key decisions made
- Important information shared
- Technical solutions mentioned"""
            elif memory_type == "conversation_chunk":
                system_prompt = """Extract the key information from this conversation chunk. Focus on:
- Important points discussed
- Key decisions made
- Critical information shared
- Session highlights"""
            else:
                system_prompt = """Extract the key insights from this conversation. Focus on:
- Important points discussed
- Key decisions made
- Critical information shared"""

            # Step 1: Enqueue in Memory Rate Limiter
            queue_timing_info = await self.memory_rate_limiter.enqueue_request(
                request_id=request_id,
                system_prompt=system_prompt,
                user_message=conversation_text
            )

            # Step 2: Get memory client
            api_type, client = self._get_memory_client()

            # Step 3: Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nSummary:"}
            ]

            # Step 4: Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_openai_call,
                client,
                messages,
                self.memory_model,  # Use gpt-4o-mini for cost
                request_id,
                api_type
            )

            return result["content"]

        except Exception as e:
            logger.error(f"❌ extract_memory_content failed: {e}")
            # Return simple fallback
            return f"[Memory extraction failed: {e}]"

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self.stats_lock:
            stats_copy = {
                api: dict(stats) for api, stats in self.stats.items()
            }

        # Add Rate Limiter stats
        chat_limiter_stats = self.chat_rate_limiter.get_stats()
        memory_limiter_stats = self.memory_rate_limiter.get_stats()

        return {
            "api_stats": stats_copy,
            "chat_rate_limiter": chat_limiter_stats,
            "memory_rate_limiter": memory_limiter_stats,
            "thread_pool": {
                "max_workers": self.executor._max_workers,
                "active_threads": threading.active_count()
            }
        }


# =================== Global Instance ===================
ai_service_manager = AIServiceManager()


# =================== Backward Compatibility ===================
class Runner:
    """
    Backward compatibility wrapper
    Preserves the original static method interface
    """

    @staticmethod
    async def run(agent, prompt: str, model: Optional[str] = None) -> str:
        """Legacy interface for run_agent"""
        return await ai_service_manager.run_agent(agent, prompt, model)

    @staticmethod
    async def run_stream(agent, prompt: str, model: Optional[str] = None):
        """Legacy interface for run_agent_stream"""
        async for chunk in ai_service_manager.run_agent_stream(agent, prompt, model):
            yield chunk
