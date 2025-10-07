from agents import Agent, Runner, WebSearchTool
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_prompt(file_name: str) -> str:
    """Load agent instructions from text files in the instructions directory"""
    base_dir = os.path.join(os.path.dirname(__file__), "../instructions")
    file_path = os.path.join(base_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""  # Return empty string if prompt file not found


# ============================================
# User History Management (3-round limit)
# ============================================

class UserHistoryManager:
    """Manage conversation history for a single user (max 3 rounds)"""
    MAX_ROUNDS = 3  # 3 rounds = 6 messages

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.history = []  # [(role, content), ...]
        self._initialized = False
        self.last_active = time.time()

    async def ensure_initialized(self):
        """Load recent 3 rounds from database on first access"""
        if self._initialized:
            return

        # Lazy import to avoid circular dependency
        from .database import get_user_conversations_async

        conversations = await get_user_conversations_async(
            self.user_id,
            limit=self.MAX_ROUNDS * 2  # 6 messages
        )

        # Convert format: [{role, content, ...}] → [(role, content)]
        # Database returns newest first, reverse to oldest first
        conversations.reverse()
        self.history = [(c['role'], c['content']) for c in conversations]
        self._initialized = True

    def add_message(self, role: str, content: str):
        """Add message and auto-truncate to 3 rounds"""
        self.history.append((role, content))
        # Keep only the most recent 6 messages (3 rounds)
        if len(self.history) > self.MAX_ROUNDS * 2:
            self.history = self.history[-(self.MAX_ROUNDS * 2):]
        self.last_active = time.time()

    def get_history(self):
        """Get current history"""
        return self.history.copy()

    def clear(self):
        """Clear history (for testing)"""
        self.history = []
        self._initialized = False


class HistoryManager:
    """Global history manager for all users"""

    def __init__(self):
        self.users = {}  # {user_id: UserHistoryManager}

    async def get_user_history_manager(self, user_id: int) -> UserHistoryManager:
        """Get user history manager (auto-initialize from database)"""
        if user_id not in self.users:
            self.users[user_id] = UserHistoryManager(user_id)

        manager = self.users[user_id]
        await manager.ensure_initialized()

        return manager


# Global instance
history_manager = HistoryManager()


# Chatbot agent for general conversation
chatbot_agent = Agent(
    name="AI Chatbot",
    instructions=load_prompt("chatbot_agent.txt")
)

# Enhanced agent execution logic with memory integration
async def run_agent(agent, user_id: int, user_message: str, history: list[tuple[str, str]] = None, model: str = "gpt-4o", mode: str = "chat") -> str:
    """
    Execute agent with conversation history and memory context

    Args:
        agent: The agent instance to run
        user_id: User ID for memory retrieval
        user_message: Current user input message
        history: List of (role, message) tuples. If None, uses UserHistoryManager (auto-managed)
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        mode: Operation mode ("eval" or "chat")

    Returns:
        str: Agent's response message
    """
    # Lazy import to avoid circular dependency
    from .memory_system import memory_system

    # Determine if using auto-managed history
    use_history_manager = (history is None)

    # Get user history manager if needed
    if use_history_manager:
        user_manager = await history_manager.get_user_history_manager(user_id)
        history = user_manager.get_history()

    # Get relevant memories for context
    memory_context = await memory_system.create_memory_context(user_id, user_message, top_k=3)

    # Build conversation prompt from history
    prompt = ""

    # Add recent conversation history with explanatory header
    if history:
        prompt += "Recent conversation history:\n"
        for role, msg in history:
            prompt += f"{role}: {msg}\n"
        prompt += "\n"

    # Add memory context if available
    if memory_context:
        prompt += "Previous relevant context from our conversations:\n"
        prompt += f"{memory_context}\n\n"

    # Add current user input
    prompt += f"User: {user_message}\nAssistant:"

    # Execute agent with the enhanced prompt
    assistant_reply = await Runner.run(agent, prompt, model)

    # If using history manager, auto-save to history
    if use_history_manager:
        user_manager.add_message("User", user_message)
        user_manager.add_message("Assistant", assistant_reply.strip())

    return assistant_reply.strip()

# Streaming version of run_agent for evaluation feedback
async def run_agent_stream(agent, user_id: int, user_message: str, history: list[tuple[str, str]] = None, model: str = "gpt-4o", mode: str = "eval"):
    """
    Execute agent with streaming response for evaluation feedback

    Args:
        agent: The agent instance to run
        user_id: User ID for memory retrieval
        user_message: Current user input message
        history: List of (role, message) tuples. If None, uses UserHistoryManager (auto-managed)
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        mode: Operation mode ("eval" or "chat")

    Yields:
        str: Chunks of agent's response message
    """
    # Lazy import to avoid circular dependency
    from .memory_system import memory_system
    from .runner import ai_service_manager

    # Determine if using auto-managed history
    use_history_manager = (history is None)

    # Get user history manager if needed
    if use_history_manager:
        user_manager = await history_manager.get_user_history_manager(user_id)
        history = user_manager.get_history()

    # Get relevant memories for context (only for chat mode)
    memory_context = ""
    if mode == "chat":
        memory_context = await memory_system.create_memory_context(user_id, user_message, top_k=3)

    # Build conversation prompt from history
    prompt = ""

    # Add recent conversation history with explanatory header
    if history:
        prompt += "Recent conversation history:\n"
        for role, msg in history:
            prompt += f"{role}: {msg}\n"
        prompt += "\n"

    # Add memory context if available
    if memory_context:
        prompt += "Previous relevant context from our conversations:\n"
        prompt += f"{memory_context}\n\n"

    # Add current user input
    prompt += f"User: {user_message}\nAssistant:"

    # Collect full response for history manager
    full_response = ""

    # Execute agent with streaming response
    async for chunk in ai_service_manager.run_agent_stream(agent, prompt, model):
        full_response += chunk
        yield chunk

    # If using history manager, auto-save to history
    if use_history_manager:
        user_manager.add_message("User", user_message)
        user_manager.add_message("Assistant", full_response.strip())
