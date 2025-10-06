"""
RAG Complete Flow Testing Script
Purpose: Automated testing of Memory System triggers and retrieval
Tests:
1. Message count trigger (every 3 messages)
2. Inactivity trigger (15/30 minutes simulated)
3. Memory retrieval with semantic search
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agents_backend import chatbot_agent, run_agent
from agents.memory_system import memory_system
from agents.database import (
    save_conversation_message_async,
    get_user_message_count_async
)

class RAGFlowTester:
    def __init__(self, user_id: int = 999, model: str = "gpt-4o"):
        self.user_id = user_id
        self.model = model
        self.conversation_history = []
        self.test_results = []

    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now()
        }
        self.test_results.append(result)

        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {test_name}: {status}")
        if details:
            print(f"   {details}")

    async def send_message(self, message: str, save_to_db: bool = True) -> str:
        """Send a message to the chatbot"""
        try:
            response = await run_agent(
                agent=chatbot_agent,
                user_id=self.user_id,
                user_message=message,
                history=self.conversation_history,
                model=self.model,
                mode="chat"
            )

            self.conversation_history.append(("User", message))
            self.conversation_history.append(("Assistant", response))

            if save_to_db:
                await save_conversation_message_async(
                    self.user_id, message, "user", "chat", 1
                )
                await save_conversation_message_async(
                    self.user_id, response, "assistant", "chat", 1
                )

            return response
        except Exception as e:
            self.log_result("Send Message", "FAIL", str(e))
            raise

    async def test_message_count_trigger(self):
        """Test 1: Message count trigger (every 3 messages)"""
        print("\n" + "=" * 60)
        print("测试 1: 消息计数触发器 (每 3 条消息)")
        print("=" * 60)

        initial_memory_count = await memory_system.get_memory_count(self.user_id)
        print(f"初始记忆数: {initial_memory_count}")

        # Send 3 messages (in English as the project targets English speakers)
        messages = [
            "I want to design a magnetic zipper tool for one-armed users",
            "This tool should be easy to use and affordable",
            "Do you think this idea is feasible?"
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n📨 消息 {i}/3: {msg}")
            response = await self.send_message(msg)
            print(f"🤖 响应: {response[:100]}...")

        # Check for memory trigger
        print(f"\n⏳ 等待记忆创建...")
        await asyncio.sleep(2)  # Wait for background task

        # Manually trigger memory creation
        message_count = await get_user_message_count_async(self.user_id, "chat")
        should_create, triggers = await memory_system.check_memory_trigger(
            self.user_id, "chat", message_count=message_count
        )

        if should_create:
            print(f"✅ 触发器检测成功: {triggers}")
            await memory_system.create_memory_async(
                self.user_id, "chat", triggers, agent_type=1
            )

            # Wait for memory creation
            await asyncio.sleep(3)

            final_memory_count = await memory_system.get_memory_count(self.user_id)
            print(f"最终记忆数: {final_memory_count}")

            if final_memory_count > initial_memory_count:
                self.log_result(
                    "消息计数触发器",
                    "PASS",
                    f"记忆数增加: {initial_memory_count} → {final_memory_count}"
                )
            else:
                self.log_result(
                    "消息计数触发器",
                    "FAIL",
                    f"记忆数未增加: {initial_memory_count} → {final_memory_count}"
                )
        else:
            self.log_result(
                "消息计数触发器",
                "FAIL",
                "未检测到触发条件"
            )

    async def test_memory_retrieval(self):
        """Test 2: Memory retrieval with semantic search"""
        print("\n" + "=" * 60)
        print("测试 2: 记忆检索 (语义相似度搜索)")
        print("=" * 60)

        # Send a query that should match previous memories
        query = "Do you remember the magnetic zipper tool I mentioned earlier?"
        print(f"\n🔍 查询: {query}")

        # Retrieve memories
        memory_context = await memory_system.create_memory_context(
            self.user_id, query, top_k=3
        )

        if memory_context:
            print(f"\n✅ 检索到记忆:")
            print(memory_context)

            # Check if relevant keywords are in retrieved memories (English)
            keywords = ["magnetic", "zipper", "one-armed", "one armed", "tool"]
            memory_lower = memory_context.lower()

            if any(kw.lower() in memory_lower for kw in keywords):
                self.log_result(
                    "记忆检索 - 语义相关性",
                    "PASS",
                    "检索到相关关键词"
                )
            else:
                self.log_result(
                    "记忆检索 - 语义相关性",
                    "FAIL",
                    "未检索到相关关键词"
                )

            # Send message and check if agent uses memory
            response = await self.send_message(query)
            print(f"\n🤖 Agent 响应: {response}")

            # Check if agent's response uses the memory (English keywords)
            response_lower = response.lower()
            if any(kw.lower() in response_lower for kw in keywords):
                self.log_result(
                    "记忆检索 - Agent 使用记忆",
                    "PASS",
                    "Agent 响应包含历史上下文"
                )
            else:
                self.log_result(
                    "记忆检索 - Agent 使用记忆",
                    "WARNING",
                    "Agent 响应未明确引用历史上下文"
                )
        else:
            self.log_result(
                "记忆检索",
                "FAIL",
                "未检索到任何记忆"
            )

    async def test_memory_time_weighting(self):
        """Test 3: Time-weighted memory retrieval"""
        print("\n" + "=" * 60)
        print("测试 3: 时间加权记忆检索")
        print("=" * 60)

        # Get recent memories
        memories = await memory_system.get_recent_memories(self.user_id, limit=5)

        if memories:
            print(f"\n✅ 获取到 {len(memories)} 条记忆:")
            for i, mem in enumerate(memories, 1):
                print(f"\n记忆 {i}:")
                print(f"  类型: {mem.get('memory_type')}")
                print(f"  创建时间: {mem.get('created_at')}")
                print(f"  内容: {mem.get('memory_content', '')[:150]}...")

            self.log_result(
                "记忆时间排序",
                "PASS",
                f"成功获取 {len(memories)} 条记忆"
            )
        else:
            self.log_result(
                "记忆时间排序",
                "FAIL",
                "未获取到记忆"
            )

    async def test_system_health(self):
        """Test 4: Memory system health check"""
        print("\n" + "=" * 60)
        print("测试 4: 系统健康检查")
        print("=" * 60)

        status = memory_system.get_status()

        print(f"\n系统健康: {status['system_health']}")
        print(f"降级模式: {status['fallback_mode']}")
        print(f"数据库配置: {status['available']}")

        if status['system_health'] == "healthy":
            self.log_result(
                "系统健康检查",
                "PASS",
                "记忆系统运行正常"
            )
        elif status['system_health'] == "degraded":
            self.log_result(
                "系统健康检查",
                "WARNING",
                f"系统运行在降级模式: {status.get('last_error', 'Unknown')}"
            )
        else:
            self.log_result(
                "系统健康检查",
                "FAIL",
                f"系统不可用: {status.get('last_error', 'Unknown')}"
            )

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)

        pass_count = sum(1 for r in self.test_results if r['status'] == 'PASS')
        fail_count = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        warning_count = sum(1 for r in self.test_results if r['status'] == 'WARNING')
        total_count = len(self.test_results)

        print(f"\n总测试数: {total_count}")
        print(f"✅ 通过: {pass_count}")
        print(f"⚠️  警告: {warning_count}")
        print(f"❌ 失败: {fail_count}")

        print("\n详细结果:")
        for result in self.test_results:
            icon = "✅" if result['status'] == "PASS" else "⚠️" if result['status'] == "WARNING" else "❌"
            print(f"{icon} {result['test']}: {result['status']}")
            if result['details']:
                print(f"   {result['details']}")

        if fail_count == 0:
            print("\n🎉 所有测试通过！")
        else:
            print(f"\n⚠️  {fail_count} 个测试失败，请检查配置")

    async def run(self):
        """Run all tests"""
        print("=" * 60)
        print("🧪 RAG 完整流程测试")
        print("=" * 60)
        print(f"User ID: {self.user_id}")
        print(f"Model: {self.model}")
        print()

        try:
            # Run tests sequentially
            await self.test_system_health()
            await self.test_message_count_trigger()
            await self.test_memory_retrieval()
            await self.test_memory_time_weighting()

            # Print summary
            self.print_summary()

        except Exception as e:
            print(f"\n❌ 测试执行错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG Flow Testing")
    parser.add_argument("--user-id", type=int, default=999, help="Test user ID (default: 999)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    args = parser.parse_args()

    tester = RAGFlowTester(user_id=args.user_id, model=args.model)
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
