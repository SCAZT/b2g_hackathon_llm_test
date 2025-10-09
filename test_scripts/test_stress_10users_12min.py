#!/usr/bin/env python3
"""
Stress Test: 10 Users × 60 Questions Each (Max 15 Minutes)
Tests system performance with continuous question flow and memory creation
"""
import asyncio
import sys
import os
import json
import time
import csv
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from statistics import mean, stdev

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.agents_backend import run_agent, chatbot_agent
from agents.memory_system import memory_system
from agents.database import save_conversation_message_async, get_user_message_count_async

# ============================================================
# Configuration
# ============================================================

@dataclass
class TestConfig:
    """Test configuration"""
    num_users: int = 10
    questions_per_user: int = 60
    max_duration_minutes: int = 15
    question_timeout_seconds: int = 60
    user_id_start: int = 6001
    model: str = "gpt-4o-mini"
    questions_file: str = "questions_60_simple.json"
    results_dir: str = "results"

    @property
    def total_requests(self):
        return self.num_users * self.questions_per_user

    @property
    def user_ids(self):
        return list(range(self.user_id_start, self.user_id_start + self.num_users))


@dataclass
class RequestRecord:
    """Record for a single request"""
    user_id: int
    question_num: int
    question_text: str
    sent_time: Optional[str] = None
    received_time: Optional[str] = None
    response_time_sec: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    status: str = "pending"  # pending, success, timeout, error
    error_msg: Optional[str] = None


# ============================================================
# Statistics Tracker
# ============================================================

class StatsTracker:
    """Track test statistics"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.records: List[RequestRecord] = []
        self.memory_stats = {
            "total_creations": 0,
            "total_extract_tokens": 0,
            "total_embedding_tokens": 0
        }
        self.start_time = None
        self.log_file = None

    def add_record(self, record: RequestRecord):
        """Add a request record"""
        self.records.append(record)

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        if self.log_file:
            self.log_file.write(log_msg + "\n")
            self.log_file.flush()

    def get_success_records(self):
        """Get all successful records"""
        return [r for r in self.records if r.status == "success"]

    def get_per_user_stats(self, user_id: int):
        """Get statistics for a specific user"""
        user_records = [r for r in self.records if r.user_id == user_id]
        success = [r for r in user_records if r.status == "success"]
        failed = [r for r in user_records if r.status != "success"]

        avg_response = None
        total_time = None
        if success:
            response_times = [r.response_time_sec for r in success if r.response_time_sec]
            if response_times:
                avg_response = mean(response_times)

            # Calculate total time from first sent to last received
            if user_records:
                sent_times = [datetime.strptime(r.sent_time, "%Y-%m-%d %H:%M:%S.%f") for r in user_records if r.sent_time]
                received_times = [datetime.strptime(r.received_time, "%Y-%m-%d %H:%M:%S.%f") for r in success if r.received_time]
                if sent_times and received_times:
                    total_time = (max(received_times) - min(sent_times)).total_seconds()

        return {
            "user_id": user_id,
            "total_questions": len(user_records),
            "success": len(success),
            "failed": len(failed),
            "success_rate": len(success) / len(user_records) * 100 if user_records else 0,
            "avg_response_time": avg_response,
            "total_time_seconds": total_time
        }

    def generate_report(self):
        """Generate final test report"""
        success_records = self.get_success_records()

        # Overall stats
        total_requests = len(self.records)
        success_count = len(success_records)
        failed_count = total_requests - success_count
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0

        # Response time stats
        response_times = [r.response_time_sec for r in success_records if r.response_time_sec]
        response_stats = {}
        if response_times:
            response_stats = {
                "min": min(response_times),
                "max": max(response_times),
                "mean": mean(response_times),
                "stdev": stdev(response_times) if len(response_times) > 1 else 0
            }

        # Token stats
        total_input_tokens = sum(r.input_tokens or 0 for r in success_records)
        total_output_tokens = sum(r.output_tokens or 0 for r in success_records)
        total_conversation_tokens = total_input_tokens + total_output_tokens

        # Per-user stats
        per_user_stats = []
        for user_id in self.config.user_ids:
            per_user_stats.append(self.get_per_user_stats(user_id))

        # Build report
        report = {
            "test_config": {
                "num_users": self.config.num_users,
                "questions_per_user": self.config.questions_per_user,
                "max_duration_minutes": self.config.max_duration_minutes,
                "question_timeout_seconds": self.config.question_timeout_seconds,
                "total_requests": self.config.total_requests,
                "model": self.config.model
            },
            "overall_stats": {
                "total_requests": total_requests,
                "successful": success_count,
                "failed": failed_count,
                "success_rate_percent": round(success_rate, 2)
            },
            "response_time_stats": response_stats,
            "token_costs": {
                "conversation": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_conversation_tokens
                },
                "memory": {
                    "total_creations": self.memory_stats["total_creations"],
                    "extract_tokens": self.memory_stats["total_extract_tokens"],
                    "embedding_tokens": self.memory_stats["total_embedding_tokens"],
                    "total_tokens": self.memory_stats["total_extract_tokens"] + self.memory_stats["total_embedding_tokens"]
                },
                "grand_total_tokens": total_conversation_tokens + self.memory_stats["total_extract_tokens"] + self.memory_stats["total_embedding_tokens"]
            },
            "per_user_stats": per_user_stats
        }

        return report


# ============================================================
# User Simulator
# ============================================================

class UserSimulator:
    """Simulates a single user sending questions"""

    def __init__(self, user_id: int, questions: List[str], config: TestConfig, stats: StatsTracker):
        self.user_id = user_id
        self.questions = questions
        self.config = config
        self.stats = stats

    async def run_all_questions(self, start_time: float):
        """Run all 60 questions continuously until completion or timeout"""
        for q_idx in range(len(self.questions)):
            q_num = q_idx + 1
            q_text = self.questions[q_idx]

            # Check if we exceeded max test duration
            elapsed = time.time() - start_time
            if elapsed > self.config.max_duration_minutes * 60:
                self.stats.log(f"⏱️  User {self.user_id} | Stopped at question {q_num}/60 (max duration reached)")
                break

            # Create record
            record = RequestRecord(
                user_id=self.user_id,
                question_num=q_num,
                question_text=q_text
            )

            try:
                # Record sent time
                sent_time = datetime.now()
                record.sent_time = sent_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                self.stats.log(f"📤 User {self.user_id} | Question {q_num}/60 | Sent")

                # Send request with timeout
                try:
                    response = await asyncio.wait_for(
                        run_agent(
                            agent=chatbot_agent,
                            user_id=self.user_id,
                            user_message=q_text,
                            history=None,
                            model=self.config.model,
                            mode="chat"
                        ),
                        timeout=self.config.question_timeout_seconds
                    )

                    # Record received time
                    received_time = datetime.now()
                    record.received_time = received_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    record.response_time_sec = (received_time - sent_time).total_seconds()

                    # Save to database
                    await save_conversation_message_async(self.user_id, q_text, "user", "chat", 1)
                    await save_conversation_message_async(self.user_id, response, "assistant", "chat", 1)

                    # Estimate tokens (rough estimation based on character count)
                    record.input_tokens = len(q_text) // 4 + 400  # Question + system prompt
                    record.output_tokens = len(response) // 4
                    record.total_tokens = record.input_tokens + record.output_tokens

                    record.status = "success"

                    self.stats.log(
                        f"✅ User {self.user_id} | Question {q_num}/60 | "
                        f"Response: {record.response_time_sec:.2f}s | "
                        f"Tokens: {record.input_tokens}in/{record.output_tokens}out"
                    )

                    # Check and trigger memory creation (non-blocking)
                    message_count = await get_user_message_count_async(self.user_id, "chat")
                    if message_count % 3 == 0:
                        should_create, triggers = await memory_system.check_memory_trigger(
                            self.user_id, "chat", message_count=message_count
                        )
                        if should_create:
                            asyncio.create_task(
                                memory_system.create_memory_async(self.user_id, "chat", triggers, agent_type=1)
                            )
                            self.stats.log(f"🧠 User {self.user_id} | Memory creation triggered (background)")

                except asyncio.TimeoutError:
                    # Timeout
                    record.status = "timeout"
                    record.error_msg = f"Request timed out after {self.config.question_timeout_seconds}s"
                    self.stats.log(f"⏱️  User {self.user_id} | Question {q_num}/60 | Timeout ({self.config.question_timeout_seconds}s)")

            except Exception as e:
                # Error
                record.status = "error"
                record.error_msg = str(e)
                self.stats.log(f"❌ User {self.user_id} | Question {q_num}/60 | Error: {e}")

            finally:
                self.stats.add_record(record)

        # Log completion
        elapsed = time.time() - start_time
        completed = sum(1 for r in self.stats.records if r.user_id == self.user_id and r.status == "success")
        self.stats.log(f"🏁 User {self.user_id} | Completed {completed}/60 questions in {elapsed/60:.2f} minutes")


# ============================================================
# Database Cleanup
# ============================================================

async def cleanup_test_users(config: TestConfig):
    """Clean up test user data from database"""
    from agents.database import DatabaseManager
    from sqlalchemy import text

    print("\n🧹 Cleaning up test user data...")

    with DatabaseManager() as db:
        try:
            # Delete conversations
            db.db.execute(
                text("DELETE FROM conversation WHERE user_id BETWEEN :start AND :end"),
                {"start": config.user_id_start, "end": config.user_id_start + config.num_users - 1}
            )

            # Delete memory vectors
            db.db.execute(
                text("DELETE FROM memory_vectors WHERE user_id BETWEEN :start AND :end"),
                {"start": config.user_id_start, "end": config.user_id_start + config.num_users - 1}
            )

            db.db.commit()
            print(f"✅ Cleaned data for users {config.user_id_start}-{config.user_id_start + config.num_users - 1}")

        except Exception as e:
            print(f"❌ Error cleaning database: {e}")
            db.db.rollback()
            raise


# ============================================================
# Environment Check
# ============================================================

async def check_environment(config: TestConfig):
    """Check if environment is ready for testing"""
    print("\n🔍 Checking environment...")

    checks = []

    # 1. Database connection
    try:
        from agents.database import DatabaseManager
        from sqlalchemy import text
        with DatabaseManager() as db:
            db.db.execute(text("SELECT 1"))
        checks.append(("Database connection", True, None))
    except Exception as e:
        checks.append(("Database connection", False, str(e)))

    # 2. Rate Limiter
    try:
        from agents.runner import ai_service_manager
        # Try a simple operation
        checks.append(("Rate Limiter", True, None))
    except Exception as e:
        checks.append(("Rate Limiter", False, str(e)))

    # 3. Questions file
    questions_path = os.path.join(os.path.dirname(__file__), config.questions_file)
    if os.path.exists(questions_path):
        with open(questions_path, 'r') as f:
            data = json.load(f)
            if len(data['questions']) == 60:
                checks.append(("Questions file (60 questions)", True, None))
            else:
                checks.append(("Questions file (60 questions)", False, f"Found {len(data['questions'])} questions"))
    else:
        checks.append(("Questions file", False, "File not found"))

    # 4. Results directory
    results_dir = os.path.join(os.path.dirname(__file__), config.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(results_dir) and os.access(results_dir, os.W_OK):
        checks.append(("Results directory", True, None))
    else:
        checks.append(("Results directory", False, "Not writable"))

    # 5. Memory system
    try:
        count = await memory_system.get_memory_count(config.user_id_start)
        checks.append(("Memory system", True, None))
    except Exception as e:
        checks.append(("Memory system", False, str(e)))

    # Print results
    all_passed = True
    for check_name, passed, error in checks:
        if passed:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}: {error}")
            all_passed = False

    if not all_passed:
        raise RuntimeError("Environment check failed")

    print("✅ All checks passed")


# ============================================================
# Save Results
# ============================================================

def save_results(stats: StatsTracker, config: TestConfig):
    """Save test results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), config.results_dir)

    # 1. Save detailed CSV
    csv_path = os.path.join(results_dir, f"stress_test_details_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['user_id', 'question_num', 'minute_num', 'question_text',
                     'sent_time', 'received_time', 'response_time_sec',
                     'input_tokens', 'output_tokens', 'total_tokens',
                     'status', 'error_msg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in stats.records:
            writer.writerow(asdict(record))

    print(f"\n💾 Saved detailed results: {csv_path}")

    # 2. Save JSON report
    report = stats.generate_report()
    json_path = os.path.join(results_dir, f"stress_test_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"💾 Saved report: {json_path}")

    return csv_path, json_path


# ============================================================
# Display Report
# ============================================================

def display_report(report: Dict):
    """Display test report in console"""
    print("\n" + "=" * 60)
    print("📊 STRESS TEST REPORT")
    print("=" * 60)

    # Overall stats
    print("\n📈 Overall Statistics:")
    print(f"  Total Requests: {report['overall_stats']['total_requests']}")
    print(f"  Successful: {report['overall_stats']['successful']}")
    print(f"  Failed: {report['overall_stats']['failed']}")
    print(f"  Success Rate: {report['overall_stats']['success_rate_percent']}%")

    # Response time stats
    if report['response_time_stats']:
        print("\n⏱️  Response Time Statistics:")
        print(f"  Min: {report['response_time_stats']['min']:.3f}s")
        print(f"  Max: {report['response_time_stats']['max']:.3f}s")
        print(f"  Mean: {report['response_time_stats']['mean']:.3f}s")
        print(f"  Std Dev: {report['response_time_stats']['stdev']:.3f}s")

    # Token costs
    print("\n💰 Token Costs:")
    conv = report['token_costs']['conversation']
    mem = report['token_costs']['memory']
    print(f"  Conversation Tokens:")
    print(f"    - Input: {conv['input_tokens']:,} tokens")
    print(f"    - Output: {conv['output_tokens']:,} tokens")
    print(f"    - Total: {conv['total_tokens']:,} tokens")
    print(f"  Memory Creation Tokens:")
    print(f"    - Extractions: {mem['total_creations']} times, {mem['extract_tokens']:,} tokens")
    print(f"    - Embeddings: {mem['total_creations']} times, {mem['embedding_tokens']:,} tokens")
    print(f"    - Total: {mem['total_tokens']:,} tokens")
    print(f"  Grand Total: {report['token_costs']['grand_total_tokens']:,} tokens")

    # Per-user stats
    print("\n👥 Performance by User:")
    print(f"  {'User ID':<10} {'Completed':<12} {'Failed':<10} {'Success Rate':<15} {'Avg Response':<15} {'Total Time':<15}")
    print("  " + "-" * 80)
    for stat in report['per_user_stats']:
        avg_resp = f"{stat['avg_response_time']:.2f}s" if stat['avg_response_time'] else "N/A"
        total_time = f"{stat['total_time_seconds']/60:.2f}min" if stat['total_time_seconds'] else "N/A"
        print(f"  {stat['user_id']:<10} {stat['success']:<12} {stat['failed']:<10} "
              f"{stat['success_rate']:.1f}%{'':<10} {avg_resp:<15} {total_time:<15}")

    print("\n" + "=" * 60)


# ============================================================
# Main Test Function
# ============================================================

async def run_stress_test():
    """Run the stress test"""
    config = TestConfig()

    # Display configuration
    print("=" * 60)
    print("🚀 STRESS TEST CONFIGURATION")
    print("=" * 60)
    print(f"Test Type: {config.num_users} Users × {config.questions_per_user} Questions Each (Continuous)")
    print(f"Total Requests: {config.total_requests} ({config.num_users} users × {config.questions_per_user} questions)")
    print(f"Max Duration: {config.max_duration_minutes} minutes")
    print(f"Question Timeout: {config.question_timeout_seconds} seconds per question")
    print(f"User IDs: {config.user_id_start}-{config.user_id_start + config.num_users - 1}")
    print(f"Model: {config.model}")
    print(f"Memory Trigger: Every 3 messages")
    print(f"Question Bank: {config.questions_file} (60 questions)")
    print("=" * 60)

    # Check environment
    await check_environment(config)

    # Clean database
    await cleanup_test_users(config)

    # Load questions
    questions_path = os.path.join(os.path.dirname(__file__), config.questions_file)
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
        questions = questions_data['questions']

    # Initialize stats tracker
    stats = StatsTracker(config)

    # Open log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), config.results_dir)
    log_path = os.path.join(results_dir, f"stress_test_log_{timestamp}.txt")
    stats.log_file = open(log_path, 'w')

    try:
        print(f"\n🚀 Starting test immediately...")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()
        stats.start_time = start_time
        stats.log("\n🚀 TEST STARTED")
        stats.log("=" * 60)

        # Create user simulators
        simulators = []
        for user_id in config.user_ids:
            simulator = UserSimulator(user_id, questions, config, stats)
            simulators.append(simulator)

        # Run all users in parallel (each continuously sends 60 questions)
        stats.log(f"\n📤 All {config.num_users} users starting continuous question flow...")
        tasks = [simulator.run_all_questions(start_time) for simulator in simulators]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        stats.log(f"\n✅ TEST COMPLETED in {elapsed/60:.2f} minutes")
        stats.log("=" * 60)

        # Wait for background memory tasks to complete
        stats.log("\n⏳ Waiting for background memory tasks to complete...")
        await asyncio.sleep(5)

        # Query database to get actual memory creation count and estimate tokens
        from agents.database import DatabaseManager
        from sqlalchemy import text

        with DatabaseManager() as db:
            # Count memories created during test
            result = db.db.execute(
                text("""
                    SELECT user_id, COUNT(*) as count,
                           AVG(LENGTH(memory_content)) as avg_length
                    FROM memory_vectors
                    WHERE user_id BETWEEN :start AND :end
                    GROUP BY user_id
                """),
                {"start": config.user_id_start, "end": config.user_id_start + config.num_users - 1}
            ).fetchall()

            total_memories = sum(row.count for row in result)

            # Estimate tokens based on actual memory creation
            # Each memory requires:
            # 1. Extract: ~600 input (conversation) + ~200 output (summary) = 800 tokens
            # 2. Embedding: ~200 tokens (embedding the summary)
            # Total per memory: ~1000 tokens
            extract_tokens = total_memories * 800  # Extract operation
            embedding_tokens = total_memories * 200  # Embedding operation

            stats.memory_stats["total_creations"] = total_memories
            stats.memory_stats["total_extract_tokens"] = extract_tokens
            stats.memory_stats["total_embedding_tokens"] = embedding_tokens

            stats.log(f"🧠 Memory creation statistics:")
            stats.log(f"   - Total memories created: {total_memories}")
            stats.log(f"   - Estimated extract tokens: {extract_tokens:,}")
            stats.log(f"   - Estimated embedding tokens: {embedding_tokens:,}")
            stats.log(f"   - Total memory tokens: {extract_tokens + embedding_tokens:,}")

    except KeyboardInterrupt:
        stats.log("\n⚠️  TEST INTERRUPTED BY USER")

    finally:
        # Close log file
        if stats.log_file:
            stats.log_file.close()

        # Save results
        csv_path, json_path = save_results(stats, config)

        # Generate and display report
        report = stats.generate_report()
        display_report(report)

        print(f"\n📁 Output Files:")
        print(f"  - Details: {csv_path}")
        print(f"  - Report: {json_path}")
        print(f"  - Log: {log_path}")
        print("\n✅ Test completed successfully!")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_stress_test())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
