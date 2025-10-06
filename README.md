# Chatbot Agent + RAG 本地测试环境

本测试环境用于验证 Chatbot Agent 与 RAG 记忆系统的完整功能，包括对话、记忆创建和记忆检索。

## 📋 目录结构

```
local_test_phase_2/
├── agents/                       # Agent 核心代码
│   ├── __init__.py
│   ├── agent.py                  # Agent 基类
│   ├── tools.py                  # Agent 工具
│   ├── runner.py                 # AI Service Manager (完整版，包含5-key + 双API)
│   ├── agents_backend.py         # Chatbot Agent 定义 (精简版，仅Chatbot)
│   ├── memory_system.py          # 记忆系统包装
│   ├── memory_manager.py         # 记忆管理核心
│   └── database.py               # 数据库操作 (精简版，仅Chat相关)
│
├── instructions/
│   └── chatbot_agent.txt         # Chatbot 系统提示词
│
├── database/
│   ├── init_schema.sql           # 数据库建表脚本
│   └── seed_test_data.sql        # 测试数据
│
├── test_scripts/
│   ├── test_interactive_chat.py  # 交互式对话测试
│   └── test_rag_flow.py          # RAG 完整流程自动化测试
│
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
└── README.md                     # 本文档
```

## 🚀 快速开始

### 1. 环境准备

#### 1.1 安装 PostgreSQL 和 pgvector

**macOS (Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15

# 安装 pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# 安装 pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### 1.2 创建数据库

```bash
# 连接到 PostgreSQL
psql -U postgres

# 创建数据库
CREATE DATABASE hackathon_test;

# 切换到数据库
\c hackathon_test

# 安装 pgvector 扩展
CREATE EXTENSION vector;

# 退出
\q
```

#### 1.3 初始化数据库表

```bash
cd /Users/zac/Desktop/master_thesis/code/local_test_phase_2

# 执行建表脚本
psql -U postgres -d hackathon_test -f database/init_schema.sql

# (可选) 插入测试数据
psql -U postgres -d hackathon_test -f database/seed_test_data.sql
```

### 2. Python 环境配置

#### 2.1 创建虚拟环境

```bash
cd /Users/zac/Desktop/master_thesis/code/local_test_phase_2

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

#### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 文件
nano .env  # 或使用其他编辑器
```

**必需配置**:
```bash
# 数据库连接
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/hackathon_test

# OpenAI API Keys (5个key，来自同一账户)
OPENAI_API_KEY_1=sk-proj-...
OPENAI_API_KEY_2=sk-proj-...
OPENAI_API_KEY_3=sk-proj-...
OPENAI_API_KEY_4=sk-proj-...
OPENAI_API_KEY_5=sk-proj-...
```

## 🧪 运行测试

### 测试 1: 交互式对话测试

**目的**: 在终端中与 Chatbot 交互，手动测试对话和记忆功能

```bash
python test_scripts/test_interactive_chat.py
```

**功能**:
- 实时对话
- 查看记忆检索过程
- 手动触发记忆创建
- 查看系统状态

**命令**:
- 直接输入消息 → 发送给 Agent
- `/memories` → 查看所有记忆
- `/status` → 查看系统状态
- `/clear` → 清空会话历史
- `/quit` → 退出

**示例输出**:
```
🤖 Chatbot Agent 交互式测试
============================================================
User ID: 1
Model: gpt-4o

💾 记忆系统状态: healthy
📊 历史消息数: 0
🧠 现有记忆数: 0

💬 You: 我想为独臂用户设计一个磁性拉链工具

🔍 检索相关记忆...
ℹ️  未检索到相关记忆

🤖 Agent 正在生成回复...
🤖 Agent: 这是一个很有意义的设计方向...

📊 消息计数: 1
ℹ️  还需 2 条消息触发记忆创建
```

### 测试 2: RAG 完整流程自动化测试

**目的**: 自动化测试记忆触发器和检索功能

```bash
python test_scripts/test_rag_flow.py
```

**测试项**:
1. ✅ 系统健康检查
2. ✅ 消息计数触发器 (每3条消息)
3. ✅ 记忆检索 (语义相似度搜索)
4. ✅ 时间加权记忆排序

**示例输出**:
```
🧪 RAG 完整流程测试
============================================================

测试 1: 消息计数触发器 (每 3 条消息)
============================================================
初始记忆数: 0

📨 消息 1/3: 我想为独臂用户设计一个磁性拉链工具
🤖 响应: 这是一个很有意义的设计方向...

...

✅ 消息计数触发器: PASS
   记忆数增加: 0 → 1

📊 测试总结
============================================================
总测试数: 5
✅ 通过: 5
⚠️  警告: 0
❌ 失败: 0

🎉 所有测试通过！
```

## 🔍 验证记忆系统

### 手动查询数据库

```bash
psql -U postgres -d hackathon_test

-- 查看用户
SELECT * FROM "user";

-- 查看对话
SELECT conversation_id, user_id, role, content, timestamp
FROM conversation
ORDER BY timestamp DESC
LIMIT 10;

-- 查看记忆向量
SELECT memory_id, user_id, memory_type, memory_content, created_at
FROM memory_vectors
ORDER BY created_at DESC
LIMIT 5;

-- 查看记忆数量
SELECT user_id, memory_type, COUNT(*) as count
FROM memory_vectors
GROUP BY user_id, memory_type;
```

### Python 脚本查询

```python
import asyncio
from agents.memory_system import memory_system

async def check_memories(user_id=1):
    # 获取记忆数量
    count = await memory_system.get_memory_count(user_id)
    print(f"记忆总数: {count}")

    # 获取最近记忆
    memories = await memory_system.get_recent_memories(user_id, limit=5)
    for mem in memories:
        print(f"\n类型: {mem['memory_type']}")
        print(f"内容: {mem['memory_content'][:100]}...")

asyncio.run(check_memories())
```

## 📊 架构说明

### API Keys 分配策略 (5-key 系统)

当前测试环境使用 5-key 系统（来自同一账户）:

| Key | 用途 | 优先级 |
|-----|------|-------|
| Key 1 | Chat | 高 |
| Key 2 | Chat + Memory (共享) | 高 |
| Key 3 | Chat + Memory (共享) | 高 |
| Key 4 | Memory + Chat (共享) | 高 |
| Key 5 | Memory | 高 |

**调用模式**:
- **Chatbot 对话**: 优先使用 Key 1-3，回退到 Key 2-4
- **记忆 Embedding**: 优先使用 Key 4-5，回退到 Key 2-4
- **记忆提取**: 优先使用 Key 4-5，回退到 Key 2-4

### 记忆触发机制

| 触发条件 | 记忆类型 | 说明 |
|---------|---------|------|
| 每 3 条消息 | `round_summary` | 总结每轮对话的关键点 |
| 15 分钟不活跃 + 至少2条新对话 | `conversation_chunk` | 保存会话片段 |
| 30 分钟不活跃 + 至少1条新对话 | `conversation_chunk` | 保存会话片段 |

### 记忆检索算法

```
final_score = semantic_similarity + 0.1 * time_decay
time_decay = 1.0 / (1.0 + hours_old * 0.1)
```

- **语义相似度**: 余弦相似度（pgvector）
- **时间衰减**: 每小时衰减 10%
- **排序**: 按 final_score 降序，返回 top 3

## 🐛 常见问题

### 1. 数据库连接失败

```
ERROR: could not connect to database
```

**解决方法**:
```bash
# 检查 PostgreSQL 是否运行
brew services list  # macOS
sudo systemctl status postgresql  # Linux

# 检查 .env 中的 DATABASE_URL 是否正确
# 检查密码、端口、数据库名
```

### 2. pgvector 扩展未安装

```
ERROR: extension "vector" does not exist
```

**解决方法**:
```bash
# 重新安装 pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make clean
make
make install  # 或 sudo make install

# 在数据库中执行
psql -U postgres -d hackathon_test
CREATE EXTENSION vector;
```

### 3. OpenAI API 限流

```
Error: Rate limit exceeded
```

**解决方法**:
- 检查 API Keys 是否配置正确
- 减少并发请求
- 等待几分钟后重试

### 4. 记忆未创建

**检查步骤**:
```bash
# 1. 检查数据库表是否存在
psql -U postgres -d hackathon_test
\dt

# 2. 检查对话是否保存
SELECT COUNT(*) FROM conversation WHERE user_id = 1;

# 3. 检查记忆触发器
# 确保发送了至少 3 条消息
```

## 📝 后续测试计划

测试完成后，可以进行以下扩展：

1. **高并发测试**: 测试 Agent 在高并发下的性能
2. **双 API 测试**: 修改调用方式，测试双 API 架构
3. **压力测试**: 测试记忆系统在大量数据下的性能
4. **集成测试**: 将测试通过的代码合并回主程序

## 📚 参考文档

- [CLAUDE.md](../b2g-hackathon-api/CLAUDE.md) - 主程序架构文档
- [pgvector 文档](https://github.com/pgvector/pgvector)
- [OpenAI API 文档](https://platform.openai.com/docs)

## 🤝 贡献

如果发现问题或有改进建议，请记录在测试日志中。
