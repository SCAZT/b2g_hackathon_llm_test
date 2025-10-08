#!/usr/bin/env python3
"""Quick test to verify interactive chat works"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.agents_backend import run_agent, chatbot_agent

async def main():
    print("Testing quick chat...")

    # Test message
    response = await run_agent(
        agent=chatbot_agent,
        user_id=1,
        user_message="Hi, can you help me test if you're working?",
        history=None,
        model="gpt-4o",
        mode="chat"
    )

    print(f"✅ Agent response: {response}")
    print("\n🎉 Chat system is working!")

if __name__ == "__main__":
    asyncio.run(main())
