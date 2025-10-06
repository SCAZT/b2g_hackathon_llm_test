import os
import asyncio
import threading
import logging
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Tuple

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class AIServiceManager:
    """
    Comprehensive AI service manager with dual architecture:
    - Evaluation: Dual-account API (5:1 load balancing) - NEW for Phase 1
    - Chat/Memory: Original overlapping key allocation - PRESERVED
    """

    def __init__(self):
        # =================== NEW: Dual-Account API for Evaluation ===================
        # Two separate OpenAI accounts for evaluation load balancing
        self.main_api_key = os.getenv("MAIN_API_KEY")
        self.backup_api_key = os.getenv("BACKUP_API_KEY")

        # Create OpenAI clients for evaluation
        if self.main_api_key:
            self.main_client = OpenAI(api_key=self.main_api_key)
            logger.info(f"✅ Evaluation Main API: ...{self.main_api_key[-4:]}")
        else:
            self.main_client = None
            logger.warning("⚠️ MAIN_API_KEY not configured, evaluation will use fallback")

        if self.backup_api_key:
            self.backup_client = OpenAI(api_key=self.backup_api_key)
            logger.info(f"✅ Evaluation Backup API: ...{self.backup_api_key[-4:]}")
        else:
            self.backup_client = None
            logger.warning("⚠️ BACKUP_API_KEY not configured")

        # Global request counter for 5:1 time-based distribution
        self.global_request_counter = 0
        self.counter_lock = threading.Lock()

        # Statistics for main API
        self.main_api_stats = {
            "in_flight_calls": 0,
            "max_in_flight": 0,
            "total_calls": 0,
            "completed_calls": 0,
            "failed_calls": 0
        }

        # Statistics for backup API
        self.backup_api_stats = {
            "in_flight_calls": 0,
            "max_in_flight": 0,
            "total_calls": 0,
            "completed_calls": 0,
            "failed_calls": 0
        }

        self.stats_lock = threading.Lock()

        # Thread pool executor for evaluation (250 workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=250,
            thread_name_prefix="EvalOpenAI"
        )
        logger.info(f"   🧵 Thread Pool: 250 workers initialized for evaluation")

        # =================== PRESERVED: Original Multi-Key for Chat/Memory ===================
        # Multiple API keys for load balancing (backward compatibility)
        self.api_keys = [
            os.getenv("OPENAI_API_KEY_1"),
            os.getenv("OPENAI_API_KEY_2"),
            os.getenv("OPENAI_API_KEY_3"),
            os.getenv("OPENAI_API_KEY_4"),
            os.getenv("OPENAI_API_KEY_5"),
        ]

        # Filter out None values and validate
        self.api_keys = [key for key in self.api_keys if key]
        if not self.api_keys:
            logger.warning("⚠️ No OPENAI_API_KEY_1-5 found, chat/memory will be limited")

        # Overlapping key allocation strategy
        # Key 1, 2, 3: Priority for chat
        # Key 2, 3, 4: Shared between chat and memory
        # Key 4, 5: Priority for memory
        self.chat_priority_indices = [0, 1, 2]  # Key 1, 2, 3
        self.shared_indices = [1, 2, 3]         # Key 2, 3, 4
        self.memory_priority_indices = [3, 4]   # Key 4, 5

        # Current indices for round-robin within each group
        self.chat_index = 0
        self.shared_index = 0
        self.memory_index = 0

        # Model configuration for different phases
        self.eval_model = "gpt-4o"  # Evaluation phase model
        self.chat_model = "gpt-4o"  # Chat phase model
        self.memory_model = "gpt-4o-mini"  # Memory extraction model (can be optimized for cost/speed)

        # Logging
        logger.info(f"🚀 AIServiceManager initialized")
        if self.main_client or self.backup_client:
            logger.info(f"   📊 Evaluation: Dual-Account API (5:1 strategy)")
        if self.api_keys:
            logger.info(f"   💬 Chat/Memory: {len(self.api_keys)} keys (overlapping allocation)")
            logger.info(f"      Chat priority: {[i+1 for i in self.chat_priority_indices]}")
            logger.info(f"      Memory priority: {[i+1 for i in self.memory_priority_indices]}")

    # =================== NEW: Dual-Account API Methods ===================

    def get_api_for_next_request(self) -> Tuple[str, OpenAI]:
        """
        Get API client for next evaluation request using 5:1 time-based distribution

        Thread-safe atomic counter increments to ensure correct load balancing:
        - Requests 1-5: Main API
        - Request 6: Backup API
        - Requests 7-11: Main API
        - Request 12: Backup API
        - ...

        Returns:
            Tuple[str, OpenAI]: (api_type, client)
                api_type: "main", "backup", or "main_fallback"
                client: OpenAI client instance
        """
        # Atomic counter increment (thread-safe)
        with self.counter_lock:
            self.global_request_counter += 1
            current_count = self.global_request_counter

        # 5:1 distribution logic
        position_in_cycle = current_count % 6

        if position_in_cycle == 0:
            # Every 6th request uses backup API
            if self.backup_client:
                logger.info(f"🎯 REQUEST #{current_count} -> BACKUP API (position: 0/6)")
                return "backup", self.backup_client
            else:
                logger.warning(f"🔄 REQUEST #{current_count} -> MAIN API (backup not configured)")
                return "main_fallback", self.main_client if self.main_client else None
        else:
            # First 5 requests use main API
            logger.info(f"🎯 REQUEST #{current_count} -> MAIN API (position: {position_in_cycle}/6)")
            if self.main_client:
                return "main", self.main_client
            else:
                logger.error(f"❌ REQUEST #{current_count} -> NO API AVAILABLE")
                raise ValueError("No evaluation API keys configured (MAIN_API_KEY or BACKUP_API_KEY)")

    def start_call(self, call_id: str, api_type: str):
        """Record API call start"""
        with self.stats_lock:
            if api_type.startswith("main"):
                stats = self.main_api_stats
                api_label = "MAIN"
            else:
                stats = self.backup_api_stats
                api_label = "BACKUP"

            stats["in_flight_calls"] += 1
            stats["total_calls"] += 1

            if stats["in_flight_calls"] > stats["max_in_flight"]:
                stats["max_in_flight"] = stats["in_flight_calls"]

            total_in_flight = (self.main_api_stats["in_flight_calls"] +
                              self.backup_api_stats["in_flight_calls"])

            logger.info(
                f"🚀 {api_label}_START {call_id} | "
                f"This API in-flight: {stats['in_flight_calls']} | "
                f"Total in-flight: {total_in_flight} | "
                f"Global counter: {self.global_request_counter}"
            )

    def end_call(self, call_id: str, api_type: str, success: bool, error_type: str = None):
        """Record API call end"""
        with self.stats_lock:
            if api_type.startswith("main"):
                stats = self.main_api_stats
                api_label = "MAIN"
            else:
                stats = self.backup_api_stats
                api_label = "BACKUP"

            stats["in_flight_calls"] = max(0, stats["in_flight_calls"] - 1)

            if success:
                stats["completed_calls"] += 1
            else:
                stats["failed_calls"] += 1

            status_icon = "✅" if success else "❌"
            error_msg = f" ({error_type})" if error_type else ""

            logger.info(
                f"{status_icon} {api_label}_END {call_id} | "
                f"This API in-flight: {stats['in_flight_calls']}{error_msg}"
            )

    async def run_agent_threadpool(self, agent, prompt: str, model: Optional[str] = None) -> str:
        """
        Execute agent using thread pool + dual-account API (for evaluation only)
        Based on tested implementation from dual_api_threadpool_api.py

        Args:
            agent: Agent instance with instructions
            prompt: User prompt/message
            model: OpenAI model to use (default: self.eval_model)

        Returns:
            str: Agent's response message
        """
        try:
            # Use default eval model if not specified
            if model is None:
                model = self.eval_model

            # Select API using 5:1 distribution
            api_type, client = self.get_api_for_next_request()

            if not client:
                raise ValueError("No evaluation API client available")

            # Generate call ID
            call_id = f"eval_req_{self.global_request_counter}"

            # Record call start
            self.start_call(call_id, api_type)

            try:
                # Execute in thread pool (matches test implementation)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._sync_openai_call,
                    client, agent.instructions or "", prompt, call_id, api_type, model
                )

                # Record call end (success)
                self.end_call(call_id, api_type, success=True)

                # Return content (simplified for evaluation)
                return result["content"]

            except Exception as e:
                error_type = type(e).__name__
                self.end_call(call_id, api_type, success=False, error_type=error_type)
                raise e

        except Exception as e:
            logger.error(f"❌ Evaluation request failed: {e}")
            return f"[OpenAI Error] {str(e)}"

    def _sync_openai_call(self, client: OpenAI, system_prompt: str, user_message: str,
                          call_id: str, api_type: str, model: str):
        """
        Synchronous OpenAI API call executed in thread pool
        EXACTLY matches test implementation from dual_api_threadpool_api.py

        Args:
            client: OpenAI client instance
            system_prompt: System prompt (agent.instructions)
            user_message: User message
            call_id: Unique call identifier
            api_type: "main" or "backup"
            model: Model name

        Returns:
            dict: Response with content, token counts, etc.
        """
        import time
        ai_processing_start_time = time.time()
        api_label = "MAIN" if api_type.startswith("main") else "BACKUP"

        try:
            logger.info(f"🔧 {api_label}_EXEC {call_id} starting OpenAI API call...")

            # Create OpenAI request
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            ai_processing_time = time.time() - ai_processing_start_time
            content = response.choices[0].message.content.strip()

            # Parse token usage
            total_tokens = response.usage.total_tokens if response.usage else 0
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            logger.info(
                f"🔧 {api_label}_EXEC {call_id} completed: "
                f"{total_tokens} tokens ({input_tokens}in+{output_tokens}out), "
                f"{ai_processing_time:.2f}s"
            )

            return {
                "content": content,
                "token_count": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_time": ai_processing_time,
                "ai_processing_start_time": ai_processing_start_time,
                "ai_processing_time": ai_processing_time
            }

        except Exception as e:
            ai_processing_time = time.time() - ai_processing_start_time
            logger.error(f"🔧 {api_label}_EXEC {call_id} failed: {e} ({ai_processing_time:.2f}s)")
            raise e

    # =================== PRESERVED: Original Methods for Chat/Memory ===================
    
    def _get_next_chat_client(self) -> Optional[OpenAI]:
        """Get next client from chat priority keys"""
        if not self.chat_priority_indices:
            return None
        
        key_index = self.chat_priority_indices[self.chat_index]
        self.chat_index = (self.chat_index + 1) % len(self.chat_priority_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    def _get_next_memory_client(self) -> Optional[OpenAI]:
        """Get next client from memory priority keys"""
        if not self.memory_priority_indices:
            return None
        
        key_index = self.memory_priority_indices[self.memory_index]
        self.memory_index = (self.memory_index + 1) % len(self.memory_priority_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    def _get_next_shared_client(self) -> Optional[OpenAI]:
        """Get next client from shared keys"""
        if not self.shared_indices:
            return None
        
        key_index = self.shared_indices[self.shared_index]
        self.shared_index = (self.shared_index + 1) % len(self.shared_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    async def _call_with_fallback(self, priority_clients: List[OpenAI], fallback_clients: List[OpenAI], 
                                 operation: str, **kwargs) -> Any:
        """Execute operation with priority and fallback clients"""
        # Try priority clients first
        for client in priority_clients:
            try:
                if operation == "chat_completion":
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content.strip()
                elif operation == "embedding":
                    response = client.embeddings.create(**kwargs)
                    return response.data[0].embedding
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as e:
                print(f"Priority client failed: {e}")
                continue
        
        # Try fallback clients if priority clients fail
        for client in fallback_clients:
            try:
                if operation == "chat_completion":
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content.strip()
                elif operation == "embedding":
                    response = client.embeddings.create(**kwargs)
                    return response.data[0].embedding
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as e:
                print(f"Fallback client failed: {e}")
                continue
        
        # If all clients fail, return error response
        if operation == "chat_completion":
            return f"[OpenAI Error] All API keys failed for chat completion"
        elif operation == "embedding":
            return [0.0] * 1536  # Default embedding
        else:
            raise Exception(f"All API keys failed for operation: {operation}")
    
    async def run_agent(self, agent, prompt: str, model: Optional[str] = None) -> str:
        """
        Execute an agent with the given prompt using OpenAI API
        Priority: Chat keys (1,2,3) -> Shared keys (2,3,4)
        
        Args:
            agent: Agent instance with instructions and tools
            prompt (str): User prompt/message to send to the agent
            model (str): OpenAI model to use (if None, uses default based on phase)
        
        Returns:
            str: Agent's response message
        """
        try:
            # Use default model if not specified
            if model is None:
                model = self.chat_model
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": agent.instructions or ""},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add model-specific parameters
            if model == "o3":
                # o3 model only supports basic parameters
                pass
            else:
                # Standard models support more parameters
                params.update({
                    "temperature": 0.7,  # Controls response creativity
                    "max_tokens": 4000
                })
            
            # Get priority clients (chat keys)
            priority_clients = []
            for _ in range(len(self.chat_priority_indices)):
                client = self._get_next_chat_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "chat_completion", **params
            )
            
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
    
    async def run_agent_stream(self, agent, prompt: str, model: Optional[str] = None):
        """
        Execute an agent with streaming response using OpenAI API
        Priority: Chat keys (1,2,3) -> Shared keys (2,3,4)
        
        Args:
            agent: Agent instance with instructions and tools
            prompt (str): User prompt/message to send to the agent
            model (str): OpenAI model to use (if None, uses default based on phase)
        
        Yields:
            str: Chunks of agent's response message
        """
        try:
            # Use default model if not specified
            if model is None:
                model = self.chat_model
            
            # o3 model doesn't support streaming, fall back to regular response
            if model == "o3":
                response = await self.run_agent(agent, prompt, model)
                yield response
                return
            
            # Prepare parameters for streaming
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": agent.instructions or ""},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            # Get priority clients (chat keys)
            priority_clients = []
            for _ in range(len(self.chat_priority_indices)):
                client = self._get_next_chat_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Try streaming with priority clients first
            for client in priority_clients:
                try:
                    # Create streaming response
                    stream = client.chat.completions.create(**params)
                    
                    # Iterate through stream chunks
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    print(f"Priority streaming client failed with {model}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    continue
            
            # Try fallback clients if priority clients fail
            for client in fallback_clients:
                try:
                    # Create streaming response
                    stream = client.chat.completions.create(**params)
                    
                    # Iterate through stream chunks
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    print(f"Fallback streaming client failed with {model}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    continue
            
            # If all streaming fails, fall back to regular response
            yield f"[OpenAI Error] All streaming clients failed, falling back to regular response"
            response = await self.run_agent(agent, prompt, model)
            yield response
            
        except Exception as e:
            yield f"[OpenAI Error] {str(e)}"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI API
        Priority: Memory keys (4,5) -> Shared keys (2,3,4)
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Get priority clients (memory keys)
            priority_clients = []
            for _ in range(len(self.memory_priority_indices)):
                client = self._get_next_memory_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "embedding",
                model="text-embedding-3-small",
                input=text
            )
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a default embedding (all zeros) as fallback
            return [0.0] * 1536
    
    async def extract_memory_content(self, conversation_text: str, memory_type: str) -> str:
        """
        Extract key memory content from conversation using OpenAI API
        Priority: Memory keys (4,5) -> Shared keys (2,3,4)
        
        Args:
            conversation_text: Full conversation text
            memory_type: Type of memory to extract ("eval" or "chat")
            
        Returns:
            str: Extracted memory content
        """
        try:
            # Create prompt based on memory type
            if memory_type == "eval_summary":
                prompt = f"""Extract the key insights and decisions from this evaluation conversation. Focus on:
- Main ideas discussed
- Decisions made
- Problems identified
- Solutions proposed

Conversation:
{conversation_text}

Summary:"""
            elif memory_type == "round_summary":
                prompt = f"""Extract the key insights from this round of chat conversation. Focus on:
- Main topics discussed
- Key decisions made
- Important information shared
- Technical solutions mentioned

Conversation:
{conversation_text}

Summary:"""
            # key_insight removed - too arbitrary based on keywords
            elif memory_type == "conversation_chunk":
                prompt = f"""Extract the key information from this conversation chunk. Focus on:
- Important points discussed
- Key decisions made
- Critical information shared
- Session highlights

Conversation:
{conversation_text}

Summary:"""
            else:  # fallback
                prompt = f"""Extract the key insights from this conversation. Focus on:
- Important points discussed
- Key decisions made
- Critical information shared

Conversation:
{conversation_text}

Summary:"""
            
            # Get priority clients (memory keys)
            priority_clients = []
            for _ in range(len(self.memory_priority_indices)):
                client = self._get_next_memory_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "chat_completion",
                model=self.memory_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key information from conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
        except Exception as e:
            print(f"Error extracting memory content: {e}")
            return f"Memory extraction failed: {str(e)}"
    
    def get_model_for_phase(self, phase: str) -> str:
        """
        Get the appropriate model for a given phase
        
        Args:
            phase: "eval" or "chat"
            
        Returns:
            str: Model name
        """
        if phase == "eval":
            return self.eval_model
        elif phase == "chat":
            return self.chat_model
        else:
            return self.chat_model  # Default to chat model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [self.eval_model, self.chat_model, self.memory_model]
    
    def get_api_key_count(self) -> int:
        """Get number of available API keys"""
        return len(self.api_keys)
    
    def get_allocation_info(self) -> Dict[str, Any]:
        """Get current allocation strategy information (for both evaluation and chat/memory)"""
        with self.stats_lock:
            # Calculate total calls for percentage
            total_eval_calls = (self.main_api_stats["total_calls"] +
                               self.backup_api_stats["total_calls"])

            allocation_info = {
                # Evaluation dual-account stats
                "evaluation": {
                    "strategy": "5:1 Time-Based Request Distribution",
                    "global_request_counter": self.global_request_counter,
                    "main_api": {
                        "total_calls": self.main_api_stats["total_calls"],
                        "percentage": (self.main_api_stats["total_calls"] / max(1, total_eval_calls)) * 100,
                        "in_flight": self.main_api_stats["in_flight_calls"],
                        "max_concurrent": self.main_api_stats["max_in_flight"],
                        "completed": self.main_api_stats["completed_calls"],
                        "failed": self.main_api_stats["failed_calls"],
                        "success_rate": (self.main_api_stats["completed_calls"] /
                                       max(1, self.main_api_stats["total_calls"])) * 100
                    },
                    "backup_api": {
                        "total_calls": self.backup_api_stats["total_calls"],
                        "percentage": (self.backup_api_stats["total_calls"] / max(1, total_eval_calls)) * 100,
                        "in_flight": self.backup_api_stats["in_flight_calls"],
                        "max_concurrent": self.backup_api_stats["max_in_flight"],
                        "completed": self.backup_api_stats["completed_calls"],
                        "failed": self.backup_api_stats["failed_calls"],
                        "success_rate": (self.backup_api_stats["completed_calls"] /
                                       max(1, self.backup_api_stats["total_calls"])) * 100
                                       if self.backup_api_stats["total_calls"] > 0 else 0
                    },
                    "allocation_accuracy": self._verify_allocation()
                },
                # Chat/Memory multi-key stats (preserved)
                "chat_memory": {
                    "total_keys": len(self.api_keys),
                    "chat_priority_keys": [i+1 for i in self.chat_priority_indices],
                    "shared_keys": [i+1 for i in self.shared_indices],
                    "memory_priority_keys": [i+1 for i in self.memory_priority_indices],
                    "current_chat_index": self.chat_index,
                    "current_shared_index": self.shared_index,
                    "current_memory_index": self.memory_index
                }
            }

        return allocation_info

    def _verify_allocation(self) -> Dict[str, Any]:
        """Verify if actual API allocation matches expected 5:1 ratio"""
        main_calls = self.main_api_stats["total_calls"]
        backup_calls = self.backup_api_stats["total_calls"]
        total = main_calls + backup_calls

        if total == 0:
            return {
                "status": "no_data",
                "message": "No evaluation requests processed yet",
                "expected_ratio": "5:1",
                "actual_ratio": "0:0"
            }

        # Calculate actual percentages
        main_percentage = (main_calls / total) * 100
        backup_percentage = (backup_calls / total) * 100

        # Expected: Main 83.3%, Backup 16.7% (tolerance: ±2%)
        expected_main = 83.3
        expected_backup = 16.7
        tolerance = 2.0

        # Verify allocation accuracy
        main_accurate = abs(main_percentage - expected_main) <= tolerance
        backup_accurate = abs(backup_percentage - expected_backup) <= tolerance
        allocation_accurate = main_accurate and backup_accurate

        # Calculate actual ratio
        if backup_calls > 0:
            ratio = main_calls / backup_calls
            actual_ratio = f"{ratio:.1f}:1"
        else:
            actual_ratio = f"{main_calls}:0"

        return {
            "status": "verified",
            "expected_ratio": "5:1",
            "actual_ratio": actual_ratio,
            "main_calls": main_calls,
            "backup_calls": backup_calls,
            "total_calls": total,
            "main_percentage": round(main_percentage, 1),
            "backup_percentage": round(backup_percentage, 1),
            "expected_main_percentage": expected_main,
            "expected_backup_percentage": expected_backup,
            "allocation_accurate": allocation_accurate,
            "main_within_tolerance": main_accurate,
            "backup_within_tolerance": backup_accurate,
            "tolerance_percentage": tolerance,
            "verification_result": "✅ PASS" if allocation_accurate else "⚠️ DEVIATION"
        }

# Global instance for easy access
ai_service_manager = AIServiceManager()

# Backward compatibility - keep the old Runner class for existing code
class Runner:
    """Static class for executing agents with OpenAI API (backward compatibility)"""
    
    @staticmethod
    async def run(agent, prompt: str, model: str) -> str:
        """
        Execute an agent with the given prompt using OpenAI API
        
        Args:
            agent: Agent instance with instructions and tools
            prompt (str): User prompt/message to send to the agent
            model (str): OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        
        Returns:
            str: Agent's response message
            
        Raises:
            Exception: If OpenAI API call fails, returns error message
        """
        return await ai_service_manager.run_agent(agent, prompt, model)
