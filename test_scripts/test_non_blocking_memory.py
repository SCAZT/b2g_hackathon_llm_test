#!/usr/bin/env python3
"""
Test to verify that memory creation doesn't block chat responses
"""
import asyncio
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.agents_backend import run_agent, chatbot_agent
from agents.memory_system import memory_system
from agents.database import save_conversation_message_async, get_user_message_count_async

async def test_blocking_vs_nonblocking():
    """Compare response time with and without memory creation"""
    user_id = 5000  # Test user

    print("=" * 60)
    print("🧪 Testing Non-Blocking Memory Creation")
    print("=" * 60)

    # Test 1: Regular message (no memory trigger)
    print("\n📝 Test 1: Regular message (no memory creation)")
    start = time.time()

    response = await run_agent(
        agent=chatbot_agent,
        user_id=user_id,
        user_message="Hello, this is message 1",
        history=None,
        model="gpt-4o",
        mode="chat"
    )

    await save_conversation_message_async(user_id, "Hello, this is message 1", "user", "chat", 1)
    await save_conversation_message_async(user_id, response, "assistant", "chat", 1)

    time1 = time.time() - start
    print(f"✅ Response time: {time1:.2f}s")
    print(f"   Response: {response[:100]}...")

    # Test 2: Second message (no memory trigger)
    print("\n📝 Test 2: Second message (no memory creation)")
    start = time.time()

    response = await run_agent(
        agent=chatbot_agent,
        user_id=user_id,
        user_message="This is message 2",
        history=None,
        model="gpt-4o",
        mode="chat"
    )

    await save_conversation_message_async(user_id, "This is message 2", "user", "chat", 1)
    await save_conversation_message_async(user_id, response, "assistant", "chat", 1)

    time2 = time.time() - start
    print(f"✅ Response time: {time2:.2f}s")
    print(f"   Response: {response[:100]}...")

    # Test 3: Third message (triggers memory creation)
    print("\n📝 Test 3: Third message (TRIGGERS memory creation)")
    start = time.time()

    response = await run_agent(
        agent=chatbot_agent,
        user_id=user_id,
        user_message="This is message 3, should trigger memory",
        history=None,
        model="gpt-4o",
        mode="chat"
    )

    await save_conversation_message_async(user_id, "This is message 3", "user", "chat", 1)
    await save_conversation_message_async(user_id, response, "assistant", "chat", 1)

    # Check trigger and create memory in background
    message_count = await get_user_message_count_async(user_id, "chat")
    should_create, triggers = await memory_system.check_memory_trigger(
        user_id, "chat", message_count=message_count
    )

    if should_create:
        print("   🔄 Memory creation triggered (running in background)")
        # Non-blocking: create_task instead of await
        asyncio.create_task(
            memory_system.create_memory_async(user_id, "chat", triggers, agent_type=1)
        )

    time3 = time.time() - start
    print(f"✅ Response time: {time3:.2f}s (with background memory task)")
    print(f"   Response: {response[:100]}...")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Performance Summary")
    print("=" * 60)
    print(f"Message 1 (no memory): {time1:.2f}s")
    print(f"Message 2 (no memory): {time2:.2f}s")
    print(f"Message 3 (with memory): {time3:.2f}s")

    avg_normal = (time1 + time2) / 2
    print(f"\nAverage without memory: {avg_normal:.2f}s")
    print(f"With background memory: {time3:.2f}s")

    if time3 < avg_normal + 0.5:  # Allow 0.5s variance
        print("\n✅ SUCCESS: Memory creation is NON-BLOCKING!")
        print("   Response time is similar with or without memory creation")
    else:
        print("\n⚠️  WARNING: Memory creation may be blocking")
        print(f"   Response time increased by {time3 - avg_normal:.2f}s")

    # Give background task time to complete
    print("\n⏳ Waiting for background memory task to complete...")
    await asyncio.sleep(5)

    # Verify memory was created
    memory_count = await memory_system.get_memory_count(user_id)
    print(f"✅ Total memories created: {memory_count}")

if __name__ == "__main__":
    asyncio.run(test_blocking_vs_nonblocking())
