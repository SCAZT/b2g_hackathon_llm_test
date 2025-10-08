#!/usr/bin/env python3
"""
Diagnostic script to test system components individually
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

async def test_database():
    """Test database connection"""
    print("\n" + "=" * 60)
    print("1️⃣ Testing Database Connection")
    print("=" * 60)

    try:
        from agents.database import get_user_profile, get_user_conversations

        # Test getting user profile
        print("Testing get_user_profile(1)...")
        profile = get_user_profile(1)

        if profile:
            print(f"✅ Database connection OK")
            print(f"   User ID: {profile['user_id']}")
            print(f"   Email: {profile.get('email', 'N/A')}")
        else:
            print(f"⚠️  User 1 not found - creating test user...")
            # User doesn't exist, that's fine

        # Test getting conversations
        print("\nTesting get_user_conversations(1)...")
        conversations = get_user_conversations(1, limit=5)
        print(f"✅ Found {len(conversations)} conversations for user 1")

        return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def test_memory_system():
    """Test memory system initialization"""
    print("\n" + "=" * 60)
    print("2️⃣ Testing Memory System")
    print("=" * 60)

    try:
        from agents.memory_system import memory_system

        print("Testing memory system health...")
        status = memory_system.get_health_status()
        print(f"✅ Memory system status: {status}")

        print("\nTesting get_memory_count...")
        count = await memory_system.get_memory_count(1)
        print(f"✅ User 1 has {count} memories")

        return True

    except Exception as e:
        print(f"❌ Memory system error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_openai_api():
    """Test OpenAI API connection"""
    print("\n" + "=" * 60)
    print("3️⃣ Testing OpenAI API")
    print("=" * 60)

    try:
        from agents.runner import ai_service_manager

        print("Testing embedding generation...")
        embedding = await ai_service_manager.generate_embedding("test message")
        print(f"✅ Embedding generated successfully (length: {len(embedding)})")

        print("\nTesting chat completion...")
        from agents import Agent
        test_agent = Agent(
            name="Test Agent",
            instructions="You are a helpful assistant. Respond with exactly: 'API test successful!'"
        )

        response = await ai_service_manager.run_agent(
            test_agent,
            "Say: API test successful!",
            "gpt-4o"
        )
        print(f"✅ Chat API response: {response[:100]}")

        return True

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chatbot_agent():
    """Test chatbot agent with a simple message"""
    print("\n" + "=" * 60)
    print("4️⃣ Testing Chatbot Agent")
    print("=" * 60)

    try:
        from agents.agents_backend import run_agent, chatbot_agent

        print("Sending test message to chatbot...")
        print("Message: 'Hello, how are you?'")

        response = await run_agent(
            agent=chatbot_agent,
            user_id=1,
            user_message="Hello, how are you?",
            history=None,
            model="gpt-4o",
            mode="chat"
        )

        print(f"✅ Chatbot response: {response[:200]}")
        return True

    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("🔍 System Diagnostic Test")
    print("=" * 60)

    results = []

    # Test 1: Database
    results.append(("Database", await test_database()))

    # Test 2: Memory System
    results.append(("Memory System", await test_memory_system()))

    # Test 3: OpenAI API
    results.append(("OpenAI API", await test_openai_api()))

    # Test 4: Chatbot Agent
    results.append(("Chatbot Agent", await test_chatbot_agent()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
