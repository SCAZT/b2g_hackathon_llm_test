"""
Test Three-API Distribution and Rate Limiter

Purpose: Verify that the new architecture works correctly
- MAIN/BACKUP 5:1 distribution for chat
- MEMORY_API for memory operations
- Rate Limiters functioning
"""
import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.runner import ai_service_manager
from agents.agents_backend import chatbot_agent


async def test_three_api_distribution():
    """Test 5:1 distribution and memory API allocation"""
    print("=" * 60)
    print("🧪 Three-API Distribution Test")
    print("=" * 60)

    # Step 1: Start Rate Limiter processors
    print("\n📊 Starting Rate Limiter processors...")
    await ai_service_manager.start()
    print("✅ Rate Limiters started")

    # Step 2: Test chat API distribution (should follow 5:1 pattern)
    print("\n🔍 Testing Chat API Distribution (12 requests)")
    print("Expected: Requests 1-5, 7-11 → MAIN, Requests 6, 12 → BACKUP")
    print("-" * 60)

    chat_results = []
    for i in range(12):
        try:
            # Simple prompt to minimize tokens
            response = await ai_service_manager.run_agent(
                chatbot_agent,
                f"Say 'Hello {i+1}'",
                model="gpt-4o-mini"  # Use cheaper model for testing
            )
            print(f"✅ Request {i+1}: {response[:30]}...")
            chat_results.append("success")
        except Exception as e:
            print(f"❌ Request {i+1} failed: {e}")
            chat_results.append("failed")

        # Small delay to avoid overwhelming
        await asyncio.sleep(0.5)

    # Step 3: Test memory API
    print("\n🧠 Testing Memory API (3 embedding requests)")
    print("-" * 60)

    memory_results = []
    for i in range(3):
        try:
            embedding = await ai_service_manager.generate_embedding(f"Test text {i+1}")
            if len(embedding) == 1536:
                print(f"✅ Embedding {i+1}: {len(embedding)} dimensions")
                memory_results.append("success")
            else:
                print(f"⚠️ Embedding {i+1}: Wrong dimensions ({len(embedding)})")
                memory_results.append("wrong_size")
        except Exception as e:
            print(f"❌ Embedding {i+1} failed: {e}")
            memory_results.append("failed")

        await asyncio.sleep(0.5)

    # Step 4: Get system statistics
    print("\n📊 System Statistics")
    print("=" * 60)
    stats = ai_service_manager.get_stats()

    print("\n🔑 API Call Distribution:")
    for api_name, api_stats in stats["api_stats"].items():
        total = api_stats["total_calls"]
        success = api_stats["success"]
        failures = api_stats["failures"]
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"  {api_name.upper()}: {total} calls ({success} ✅, {failures} ❌, {success_rate:.1f}% success)")

    print("\n📊 Chat Rate Limiter:")
    chat_limiter = stats["chat_rate_limiter"]
    print(f"  Enqueued: {chat_limiter['request_stats']['total_enqueued']}")
    print(f"  Processed: {chat_limiter['request_stats']['total_processed']}")
    print(f"  Rejected: {chat_limiter['request_stats']['total_rejected']}")
    print(f"  Success Rate: {chat_limiter['request_stats']['success_rate']:.1f}%")

    print("\n🧠 Memory Rate Limiter:")
    memory_limiter = stats["memory_rate_limiter"]
    print(f"  Enqueued: {memory_limiter['request_stats']['total_enqueued']}")
    print(f"  Processed: {memory_limiter['request_stats']['total_processed']}")
    print(f"  Rejected: {memory_limiter['request_stats']['total_rejected']}")
    print(f"  Success Rate: {memory_limiter['request_stats']['success_rate']:.1f}%")

    # Step 5: Verify distribution
    print("\n✅ Verification")
    print("=" * 60)

    main_calls = stats["api_stats"]["main_api"]["total_calls"]
    backup_calls = stats["api_stats"]["backup_api"]["total_calls"]
    memory_calls = stats["api_stats"]["memory_api"]["total_calls"]

    # Expected: 10 MAIN, 2 BACKUP (from 12 chat requests)
    expected_main = 10
    expected_backup = 2
    expected_memory = 3

    print(f"Chat API Distribution:")
    print(f"  MAIN: {main_calls} (expected {expected_main}) {'✅' if main_calls == expected_main else '❌'}")
    print(f"  BACKUP: {backup_calls} (expected {expected_backup}) {'✅' if backup_calls == expected_backup else '❌'}")
    print(f"\nMemory API:")
    print(f"  MEMORY: {memory_calls} (expected {expected_memory}) {'✅' if memory_calls == expected_memory else '❌'}")

    # Calculate ratio
    if backup_calls > 0:
        ratio = main_calls / backup_calls
        print(f"\nActual ratio: {ratio:.1f}:1 (expected 5:1)")

    # Step 6: Stop Rate Limiters
    print("\n🛑 Stopping Rate Limiter processors...")
    await ai_service_manager.stop()
    print("✅ Shutdown complete")

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    total_success = chat_results.count("success") + memory_results.count("success")
    total_tests = len(chat_results) + len(memory_results)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_tests - total_success}")

    if total_success == total_tests:
        print("\n🎉 ✅ All tests passed!")
    else:
        print(f"\n⚠️ {total_tests - total_success} tests failed")


if __name__ == "__main__":
    asyncio.run(test_three_api_distribution())
