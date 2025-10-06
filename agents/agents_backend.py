from agents import Agent, Runner, WebSearchTool
import os
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

# Chatbot agent for general conversation
chatbot_agent = Agent(
    name="AI Chatbot",
    instructions=load_prompt("chatbot_agent.txt")
)

# Enhanced agent execution logic with memory integration
async def run_agent(agent, user_id: int, user_message: str, history: list[tuple[str, str]], model: str, mode: str = "chat") -> str:
    """
    Execute agent with conversation history and memory context

    Args:
        agent: The agent instance to run
        user_id: User ID for memory retrieval
        user_message: Current user input message
        history: List of (role, message) tuples representing conversation history
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        mode: Operation mode ("eval" or "chat")

    Returns:
        str: Agent's response message
    """
    # Lazy import to avoid circular dependency
    from .memory_system import memory_system

    # Get relevant memories for context
    memory_context = await memory_system.create_memory_context(user_id, user_message, top_k=3)

    # Build conversation prompt from history
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"

    # Add memory context if available
    if memory_context:
        prompt += f"\nPrevious relevant context from our conversations:\n{memory_context}\n"

    prompt += f"User: {user_message}\nAssistant:"

    # Execute agent with the enhanced prompt
    assistant_reply = await Runner.run(agent, prompt, model)
    return assistant_reply.strip()

# Streaming version of run_agent for evaluation feedback
async def run_agent_stream(agent, user_id: int, user_message: str, history: list[tuple[str, str]], model: str, mode: str = "eval"):
    """
    Execute agent with streaming response for evaluation feedback

    Args:
        agent: The agent instance to run
        user_id: User ID for memory retrieval
        user_message: Current user input message
        history: List of (role, message) tuples representing conversation history
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        mode: Operation mode ("eval" or "chat")

    Yields:
        str: Chunks of agent's response message
    """
    # Lazy import to avoid circular dependency
    from .memory_system import memory_system
    from .runner import ai_service_manager

    # Get relevant memories for context (only for chat mode)
    memory_context = ""
    if mode == "chat":
        memory_context = await memory_system.create_memory_context(user_id, user_message, top_k=3)

    # Build conversation prompt from history
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"

    # Add memory context if available
    if memory_context:
        prompt += f"\nPrevious relevant context from our conversations:\n{memory_context}\n"

    prompt += f"User: {user_message}\nAssistant:"

    # Execute agent with streaming response
    async for chunk in ai_service_manager.run_agent_stream(agent, prompt, model):
        yield chunk
