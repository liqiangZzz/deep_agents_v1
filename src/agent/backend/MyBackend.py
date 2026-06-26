import json
from urllib.parse import parse_qs, unquote, urlparse

import pymysql
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langchain_core.tools import tool

from agent.my_tools import web_search
from model.init_chat_model import glm_llm
from utils.env_utils import MYSQL_DATABASE_URL


# =====================================================================
# 1. 定义 MySQLStoreBackend —— 用 MySQL 实现 StoreBackend 的基础接口
# =====================================================================
# StoreBackend 是 deepagents 的存储后端抽象；自定义 MySQL 版本时，
# 核心就是继承 StoreBackend，并实现 get / put / delete 这几个方法。
class MySQLStoreBackend(StoreBackend):
    """基于 MySQL 的简单键值存储后端。"""

    def __init__(self, database_url: str, table_name: str = "deep_agent_preferences") -> None:
        if not database_url:
            raise RuntimeError(
                "缺少 MYSQL_DATABASE_URL 环境变量，请先在 .env 或系统环境变量中配置 MySQL 连接地址。"
            )

        self.database_url = database_url
        self.table_name = table_name
        self.connection = self._connect()
        self._setup_table()

    @classmethod
    def from_conn_string(
        cls,
        database_url: str,
        table_name: str = "deep_agent_preferences",
    ) -> "MySQLStoreBackend":
        """从 MySQL 连接字符串创建 Store，支持 with ... as store 的写法。"""
        return cls(database_url=database_url, table_name=table_name)

    def __enter__(self) -> "MySQLStoreBackend":
        """进入上下文管理器时返回当前 Store 实例。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出上下文管理器时自动关闭 MySQL 连接。"""
        self.close()

    def _connect(self):
        """根据 MYSQL_DATABASE_URL 创建 MySQL 连接。"""
        parsed = urlparse(self.database_url)
        query = parse_qs(parsed.query)

        # pymysql 是真正负责连接 MySQL 的驱动；StoreBackend 只定义 get / put / delete 抽象接口。
        return pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=(parsed.path or "").lstrip("/"),
            charset=query.get("charset", ["utf8mb4"])[0],
            autocommit=True,
            connect_timeout=int(query.get("connect_timeout", [10])[0]),
            read_timeout=int(query.get("read_timeout", [30])[0]),
            write_timeout=int(query.get("write_timeout", [30])[0]),
        )

    def _ensure_connection(self) -> None:
        """确认 MySQL 连接可用；如果被服务端断开，则自动重连。"""
        try:
            # MySQL 会关闭长时间空闲连接；先 ping 检查，失败后再显式创建新连接。
            self.connection.ping()
        except pymysql.MySQLError:
            if self.connection and self.connection.open:
                self.connection.close()
            self.connection = self._connect()

    def _setup_table(self) -> None:
        """初始化存储表；已存在时不会重复创建。"""
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `deep_agent_preferences` (
                    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    `preference_type` VARCHAR(32) NOT NULL,
                    `content` VARCHAR(512) NOT NULL,
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY `uniq_preference` (`preference_type`, `content`)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """
            )

    def append_user_preference(self, preference_type: str, content: str) -> None:
        """追加一条用户偏好；相同类型和内容已存在时忽略，避免覆盖旧记录。"""
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT IGNORE INTO `deep_agent_preferences` (`preference_type`, `content`)
                VALUES (%s, %s)
                """,
                (preference_type, content),
            )

    def list_user_preferences(self) -> dict[str, list[str]]:
        """按类型读取所有用户偏好记录。"""
        preferences = {"likes": [], "dislikes": [], "notes": []}

        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT `preference_type`, `content`
                FROM `deep_agent_preferences`
                ORDER BY `id` ASC
                """
            )
            rows = cursor.fetchall()

        for preference_type, content in rows:
            if preference_type not in preferences:
                preferences["notes"].append(content)
                continue
            preferences[preference_type].append(content)

        return preferences

    def close(self) -> None:
        """关闭 MySQL 连接。"""
        if self.connection and self.connection.open:
            self.connection.close()


# =====================================================================
# 2. 创建 Store Agent —— 注入自定义 MySQLStoreBackend
# =====================================================================
# 如果只是创建长期运行的全局 agent，可以直接保存 store 实例。
# 如果是在函数里临时创建 agent，也可以使用：
# with MySQLStoreBackend.from_conn_string(MYSQL_DATABASE_URL) as store:
#     agent = create_deep_agent(..., store=store, ...)
mysql_store_backend = MySQLStoreBackend.from_conn_string(MYSQL_DATABASE_URL)


# =====================================================================
# 3. 定义偏好记忆工具 —— 显式把用户偏好写入 / 读出 MySQL Store
# =====================================================================
# 注意：把 store 传给 create_deep_agent 只是提供存储能力，
# 并不代表模型会自动把普通聊天内容写入数据库。
# 因此这里提供明确的工具：表达偏好时写入，询问偏好时读取。
def _load_user_preferences() -> dict[str, list[str]]:
    """读取用户偏好；没有历史记录时返回空结构。"""
    return mysql_store_backend.list_user_preferences()


@tool("save_user_preference", parse_docstring=True)
def save_user_preference(preference_type: str, content: str) -> str:
    """
    保存用户明确表达的偏好、厌恶或其他长期信息。

    Args:
        preference_type: 偏好类型，可传 likes、dislikes 或 notes。
        content: 需要保存的具体内容，例如“水果”“面条”“冰饮料”。

    Returns:
        保存结果说明。
    """
    try:
        preference_type = preference_type.lower().strip()
        if preference_type in ("like", "likes", "喜欢"):
            target_key = "likes"
        elif preference_type in ("dislike", "dislikes", "不喜欢", "讨厌"):
            target_key = "dislikes"
        else:
            target_key = "notes"

        content = content.strip()
        if content:
            mysql_store_backend.append_user_preference(target_key, content)

        return f"已保存到 {target_key}: {content}"
    except Exception as e:
        return f"保存偏好失败: {e}"


@tool("list_user_preferences", parse_docstring=True)
def list_user_preferences() -> str:
    """
    读取已经保存的用户偏好列表。

    Returns:
        用户喜欢、不喜欢和其他备注的 JSON 字符串。
    """
    try:
        preferences = _load_user_preferences()
        return json.dumps(preferences, ensure_ascii=False)
    except Exception as e:
        return f"读取偏好失败: {e}"


agent = create_deep_agent(
    model=glm_llm,
    tools=[web_search, save_user_preference, list_user_preferences],
    system_prompt=(
        "你是一个助手，请根据用户输入的指令，进行相应的操作。"
        "当用户明确表达喜欢、偏好、讨厌、不喜欢等长期信息时，"
        "必须调用 save_user_preference 保存，每个偏好点单独保存一次。"
        "当用户询问自己喜欢什么、不喜欢什么、有什么偏好时，"
        "必须先调用 list_user_preferences 读取已保存信息，再根据读取结果回答。"
    )
)
