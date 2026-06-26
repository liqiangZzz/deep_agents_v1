import atexit
import hashlib
import json

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langchain_core.tools import tool
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
# 2. 定义偏好记忆工具 —— 显式把用户偏好写入 / 读出 Redis Store
# =====================================================================
# store=redis_store 只是把 LangGraph store 注入给 Agent 运行时，
# 不代表模型会自动把普通聊天内容保存成长期记忆。
# 因此这里提供明确工具：表达偏好时写入 Redis，询问偏好时读取 Redis。
USER_PREFERENCES_NAMESPACE = ("user_preferences",)


def _normalize_preference_type(preference_type: str) -> str:
    """把模型传入的偏好类型统一成 likes / dislikes / notes。"""
    preference_type = preference_type.lower().strip()
    if preference_type in ("like", "likes", "喜欢"):
        return "likes"
    if preference_type in ("dislike", "dislikes", "不喜欢", "讨厌"):
        return "dislikes"
    return "notes"


def _build_preference_key(preference_type: str, content: str) -> str:
    """构造稳定 key；同类同内容会覆盖同一条记录，避免重复保存。"""
    raw_key = f"{preference_type}:{content}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


@tool("save_user_preference", parse_docstring=True)
def save_user_preference(preference_type: str, content: str) -> str:
    """
    保存用户明确表达的偏好、厌恶或其他长期信息。

    Args:
        preference_type: 偏好类型，可传 likes、dislikes 或 notes。
        content: 需要保存的具体内容，例如“水果”“面条”“饮料”。

    Returns:
        保存结果说明。
    """
    target_type = _normalize_preference_type(preference_type)
    content = content.strip()
    if not content:
        return "没有可保存的偏好内容。"

    redis_store.put(
        USER_PREFERENCES_NAMESPACE,
        _build_preference_key(target_type, content),
        {"type": target_type, "content": content},
    )
    return f"已保存到 Redis Store: {target_type} -> {content}"


@tool("list_user_preferences", parse_docstring=True)
def list_user_preferences() -> str:
    """
    读取已经保存到 Redis Store 的用户偏好列表。

    Returns:
        用户喜欢、不喜欢和其他备注的 JSON 字符串。
    """
    preferences = {"likes": [], "dislikes": [], "notes": []}
    items = redis_store.search(USER_PREFERENCES_NAMESPACE, limit=100)

    for item in items:
        value = item.value
        preference_type = value.get("type", "notes")
        content = value.get("content", "")
        if preference_type not in preferences:
            preference_type = "notes"
        if content:
            preferences[preference_type].append(content)

    # 返回 JSON 字符串
    return json.dumps(preferences, ensure_ascii=False)


# =====================================================================
# 3. 创建 Redis 偏好记忆 Agent —— 通过工具显式读写 LangGraph RedisStore
# =====================================================================
agent = create_deep_agent(
    model=glm_llm,
    tools=[web_search, save_user_preference, list_user_preferences],
    store=redis_store,
    backend=StoreBackend(),
    system_prompt=(
        "你是一个助手，请根据用户输入的指令，进行相应的操作。"
        "当用户明确表达喜欢、偏好、讨厌、不喜欢等长期信息时，"
        "必须调用 save_user_preference 保存，每个偏好点单独保存一次。"
        "当用户询问自己喜欢什么、不喜欢什么、有什么偏好时，"
        "必须先调用 list_user_preferences 读取已保存信息，再根据读取结果回答。"
    )
)
