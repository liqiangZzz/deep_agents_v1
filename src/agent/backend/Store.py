import atexit

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store.redis import RedisStore

from agent.my_tools import web_search
from model.init_chat_model import glm_llm
from utils.env_utils import REDIS_DATABASE_URL

# =====================================================================
# 1. 创建 Redis Store —— 持久化保存 Agent 的长期 store 数据
# =====================================================================
# RedisStore 是 LangGraph 的长期 store，用来保存可跨进程复用的结构化数据。
# 它应该传给 create_deep_agent 的 store 参数，不是 checkpointer，也不是 backend。
# 注意：store 不等于对话检查点；如果要保存同一个 thread_id 的完整消息历史，应使用 RedisSaver。
DB_URI = REDIS_DATABASE_URL or "redis://localhost:6380"
redis_store_context = RedisStore.from_conn_string(DB_URI)
redis_store = redis_store_context.__enter__()

# 第一次使用 RedisStore 前需要 setup，用来创建 Redis Search 索引等内部结构。
# 如果索引已经存在，setup 通常会安全跳过。
try:
    redis_store.setup()
except Exception as exc:
    raise RuntimeError(
        "RedisStore 初始化失败。当前 Redis 服务可能是普通 Redis，缺少 RediSearch/Redis Stack 的 FT._LIST 命令。"
        "请改用 Redis Stack 服务，或换成不依赖 Redis Search 的 store。"
    ) from exc


def close_redis_store() -> None:
    """进程退出时关闭 Redis 连接。"""
    redis_store_context.__exit__(None, None, None)


# 注册进程退出时关闭 Redis 连接
atexit.register(close_redis_store)

# =====================================================================
# 3. 创建 Redis Store Agent —— 注入 LangGraph RedisStore
# =====================================================================
agent = create_deep_agent(
    model=glm_llm,
    tools=[web_search],
    store=redis_store,
    backend=StoreBackend(),
    system_prompt=(
        "你是一个助手，请根据用户输入的指令，进行相应的操作。"
    )
)
