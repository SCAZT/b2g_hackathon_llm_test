"""
Interactive Chat Testing Script
Purpose: Test Chatbot Agent with RAG Memory System in an interactive terminal session
"""

import sys
import os
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agents_backend import chatbot_agent, run_agent
from agents.memory_system import memory_system
from agents.database import (
    save_conversation_message_async,
    get_user_message_count_async,
    get_user_by_token_async
)

class InteractiveChatTester:
    def __init__(self, user_id: int = 1, model: str = "gpt-4o"):
        self.user_id = user_id
        self.model = model
        self.message_count = 0

    async def initialize(self):
        """Initialize the chat session"""
        print("=" * 60)
        print("🤖 Chatbot Agent 交互式测试")
        print("=" * 60)
        print(f"User ID: {self.user_id}")
        print(f"Model: {self.model}")
        print()

        # Check memory system status
        status = memory_system.get_status()
        print(f"💾 记忆系统状态: {status['system_health']}")
        if status['fallback_mode']:
            print(f"⚠️  警告: 运行在降级模式 - {status.get('last_error', 'Unknown')}")
        print()

        # Get existing message count
        self.message_count = await get_user_message_count_async(self.user_id, "chat")
        print(f"📊 历史消息数: {self.message_count}")
        print()

        # Get memory count
        memory_count = await memory_system.get_memory_count(self.user_id)
        print(f"🧠 现有记忆数: {memory_count}")
        print()

        print("命令:")
        print("  输入消息 → 发送给 Agent")
        print("  /memories → 查看所有记忆")
        print("  /status → 查看系统状态")
        print("  /clear → 清空会话历史")
        print("  /quit → 退出")
        print("=" * 60)
        print()

    async def chat(self, user_message: str) -> str:
        """Send a message and get response"""
        # Increment message count
        self.message_count += 1

        # Show memory retrieval
        print(f"\n🔍 检索相关记忆...")
        memory_context = await memory_system.create_memory_context(
            self.user_id, user_message, top_k=3
        )

        if memory_context:
            print(f"✅ 检索到 {len(memory_context.split('Memory'))-1} 条相关记忆")
            print(f"📝 记忆内容:\n{memory_context}\n")
        else:
            print("ℹ️  未检索到相关记忆\n")

        # Call agent
        print(f"🤖 Agent 正在生成回复...")
        try:
            response = await run_agent(
                agent=chatbot_agent,
                user_id=self.user_id,
                user_message=user_message,
                history=None,  # Use UserHistoryManager (auto-managed)
                model=self.model,
                mode="chat"
            )

            # Note: History is auto-managed by UserHistoryManager in run_agent()

            # Save to database
            await save_conversation_message_async(
                self.user_id, user_message, "user", "chat", 1
            )
            await save_conversation_message_async(
                self.user_id, response, "assistant", "chat", 1
            )

            # Check memory triggers
            print(f"\n📊 消息计数: {self.message_count}")
            trigger_status = self.message_count % 3
            if trigger_status == 0:
                print(f"✅ 达到记忆触发条件 (每3条消息)")
                print(f"🔄 后台创建记忆中...")

                # Trigger memory creation
                should_create, triggers = await memory_system.check_memory_trigger(
                    self.user_id, "chat", message_count=self.message_count
                )
                if should_create:
                    await memory_system.create_memory_async(
                        self.user_id, "chat", triggers, agent_type=1
                    )
                    print(f"✅ 记忆创建完成 (triggers: {triggers})")
            else:
                remaining = 3 - trigger_status
                print(f"ℹ️  还需 {remaining} 条消息触发记忆创建")

            return response

        except Exception as e:
            print(f"❌ 错误: {e}")
            return f"[Error] {str(e)}"

    async def show_memories(self):
        """Show all user memories"""
        print("\n" + "=" * 60)
        print("🧠 用户记忆")
        print("=" * 60)

        memories = await memory_system.get_recent_memories(self.user_id, limit=10)

        if not memories:
            print("ℹ️  暂无记忆")
        else:
            for i, mem in enumerate(memories, 1):
                print(f"\n记忆 {i}:")
                print(f"  类型: {mem.get('memory_type', 'unknown')}")
                print(f"  创建时间: {mem.get('created_at', 'unknown')}")
                print(f"  内容: {mem.get('memory_content', 'N/A')[:200]}...")

        print("=" * 60)

    async def show_status(self):
        """Show system status"""
        print("\n" + "=" * 60)
        print("📊 系统状态")
        print("=" * 60)

        status = memory_system.get_status()
        print(f"记忆系统健康: {status['system_health']}")
        print(f"降级模式: {status['fallback_mode']}")
        print(f"数据库URL: {status['database_url'][:50]}...")

        if status.get('last_error'):
            print(f"最后错误: {status['last_error']}")

        memory_count = await memory_system.get_memory_count(self.user_id)
        print(f"\n当前用户记忆数: {memory_count}")
        print(f"会话消息数: {self.message_count}")

        print("=" * 60)

    async def clear_history(self):
        """Clear conversation history"""
        from agents.agents_backend import history_manager

        # Clear user's history in memory
        if self.user_id in history_manager.users:
            history_manager.users[self.user_id].clear()

        print("\n✅ 会话历史已清空")

    async def run(self):
        """Run interactive chat loop"""
        await self.initialize()

        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input == "/quit":
                    print("\n👋 再见!")
                    break
                elif user_input == "/memories":
                    await self.show_memories()
                    continue
                elif user_input == "/status":
                    await self.show_status()
                    continue
                elif user_input == "/clear":
                    await self.clear_history()
                    continue

                # Normal chat
                response = await self.chat(user_input)
                print(f"\n🤖 Agent: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback
                traceback.print_exc()


async def main():
    """Main entry point"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Chatbot Testing")
    parser.add_argument("--user-id", type=int, default=1, help="User ID (default: 1)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    args = parser.parse_args()

    # Run tester
    tester = InteractiveChatTester(user_id=args.user_id, model=args.model)
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
