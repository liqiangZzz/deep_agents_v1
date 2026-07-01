"""MySQL 虚拟文件系统后端示例

本模块展示如何用 MySQL 作为虚拟文件系统后端，
让 Agent 可以通过 MySQL 进行文件操作（ls/read/write/edit/grep/glob）。

核心思路：把 MySQL 表当作"虚拟磁盘"，每行记录就是一个文件，
file_path 是文件路径，content 是文件内容，encoding 标记编码方式。

架构概览：
    ┌──────────┐    BackendProtocol     ┌───────────────┐     SQL      ┌────────┐
    │  Agent   │ ── ls/read/write/... ──▶│ MySQLBackend  │ ──────────▶ │ MySQL  │
    └──────────┘                        └───────────────┘              └────────┘
                                            │
                                      自动重连 / 编码转换
                                            │
                                      ┌─────┴──────┐
                                      │ virtual_   │
                                      │ filesystem │  ← 一张表存所有文件
                                      └────────────┘

继承自 BackendProtocol，实现了完整的文件系统接口。
"""

import asyncio
import base64
import fnmatch
import re
from typing import AsyncIterator
from urllib.parse import parse_qs, unquote, urlparse

import pymysql
from deepagents import create_deep_agent
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from langgraph.checkpoint.memory import InMemorySaver

from agent.my_tools import web_search
from model.init_chat_model import glm_llm
from utils.env_utils import MYSQL_DATABASE_URL


# =====================================================================
# 1. MySQL 虚拟文件系统后端 —— 实现 BackendProtocol
# =====================================================================
class MySQLBackend(BackendProtocol):
    """基于 MySQL 的虚拟文件系统后端。

    所有文件数据存储在 MySQL 表中，通过 BackendProtocol 接口
    提供给 Agent 文件操作能力。

    表结构：
        - id:           自增主键
        - file_path:    文件路径（唯一键），如 "/src/main.py"
        - content:      文件内容（LONGTEXT，支持大文件）
        - encoding:     编码方式，"utf-8" 或 "base64"（二进制文件用 base64）
        - created_at:   创建时间
        - modified_at:  修改时间（自动更新）
    """

    def __init__(self, database_url: str, table_name: str = "virtual_filesystem") -> None:
        """初始化 MySQL 后端。

        Args:
            database_url: MySQL 连接字符串，格式如 mysql://user:pass@host:3306/dbname?charset=utf8mb4
            table_name:   存储虚拟文件的表名，默认 "virtual_filesystem"
        """
        if not database_url:
            raise RuntimeError(
                "缺少 MYSQL_DATABASE_URL 环境变量，请先在 .env 或系统环境变量中配置 MySQL 连接地址。"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError("table_name 只能包含字母、数字和下划线，且不能以数字开头。")

        self.database_url = database_url
        self.table_name = table_name
        self.connection = self._connect()  # 建立数据库连接
        self._setup_table()  # 确保存储表存在

    @classmethod
    def from_conn_string(
            cls,
            database_url: str,
            table_name: str = "virtual_filesystem",
    ) -> "MySQLBackend":
        """从 MySQL 连接字符串创建后端实例（工厂方法）。

        与直接调用 __init__ 等价，提供更语义化的创建方式。
        """
        return cls(database_url=database_url, table_name=table_name)

    def __enter__(self) -> "MySQLBackend":
        """进入上下文管理器时返回当前实例。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出上下文管理器时自动关闭 MySQL 连接。"""
        self.close()

    # -----------------------------------------------------------------
    # 内部工具方法
    # -----------------------------------------------------------------

    def _connect(self):
        """根据 database_url 创建 MySQL 连接。

        解析 URL 格式：mysql://user:password@host:port/database?charset=utf8mb4&connect_timeout=10
        支持 URL 编码的用户名/密码（如含特殊字符）。
        """
        parsed = urlparse(self.database_url)
        query = parse_qs(parsed.query)

        return pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),       # URL 解码用户名（处理 %40 等编码）
            password=unquote(parsed.password or ""),   # URL 解码密码
            database=(parsed.path or "").lstrip("/"),  # 去掉前导 / 得到数据库名
            charset=query.get("charset", ["utf8mb4"])[0],            # 默认 utf8mb4 支持中文
            autocommit=True,                                         # 自动提交，无需手动管理事务
            connect_timeout=int(query.get("connect_timeout", [10])[0]),   # 连接超时 10s
            read_timeout=int(query.get("read_timeout", [30])[0]),         # 读超时 30s
            write_timeout=int(query.get("write_timeout", [30])[0]),       # 写超时 30s
        )

    def _ensure_connection(self) -> None:
        """确认 MySQL 连接可用；如果被服务端断开（如 wait_timeout），则自动重连。

        MySQL 服务端会主动断开长时间空闲的连接（默认 8 小时），
        调用此方法可在每次操作前确保连接存活，避免 "MySQL server has gone away" 错误。
        """
        if self.connection is None or not getattr(self.connection, "open", False):
            self.connection = self._connect()
            return

        try:
            self.connection.ping(reconnect=True)  # 轻量级心跳检测；必要时由 PyMySQL 重连
        except Exception:
            # PyMySQL 在 socket 已被置空时可能抛 AttributeError，而不是 MySQLError。
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = self._connect()

    def _setup_table(self) -> None:
        """初始化存储表；已存在时不会重复创建（IF NOT EXISTS）。

        表设计说明：
        - file_path 设为唯一键，保证同一路径只有一个文件
        - content 用 LONGTEXT，支持存储大文件（最大 4GB）
        - encoding 区分文本(utf-8)和二进制(base64)，读写时据此做编解码
        """
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{self.table_name}`
                (
                    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    `file_path` VARCHAR(512) NOT NULL COMMENT '文件路径，以 / 开头',
                    `content` LONGTEXT NOT NULL COMMENT '文件内容',
                    `encoding` VARCHAR(16) NOT NULL DEFAULT 'utf-8' COMMENT '编码：utf-8 或 base64',
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    `modified_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY `uniq_path` (`file_path`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

    def _get_file(self, file_path: str) -> dict | None:
        """获取文件记录（内部辅助方法）。

        Args:
            file_path: 文件路径，如 "/src/main.py"

        Returns:
            包含 content / encoding / created_at / modified_at 的字典，
            文件不存在时返回 None。
        """
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `content`, `encoding`, `created_at`, `modified_at` FROM `{self.table_name}` WHERE `file_path` = %s",
                (file_path,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "content": row[0],
                "encoding": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "modified_at": row[3].isoformat() if row[3] else None,
            }

    @staticmethod
    def _normalize_path(path: str) -> str:
        """统一虚拟文件路径格式，保持 BackendProtocol 要求的绝对路径语义。"""
        if not path:
            return "/"
        return path if path.startswith("/") else "/" + path

    @classmethod
    def _directory_prefix(cls, path: str) -> str:
        """返回目录前缀，确保以 / 结尾。"""
        normalized = cls._normalize_path(path)
        return normalized if normalized.endswith("/") else normalized + "/"

    @staticmethod
    def _decode_text(content: str, encoding: str) -> str | None:
        """把存储内容转换为文本；二进制文件无法可靠转文本时返回 None。"""
        if encoding == "base64":
            try:
                return base64.standard_b64decode(content).decode("utf-8")
            except UnicodeDecodeError:
                return None
        return content

    @staticmethod
    def _get_result_field(result, field: str):
        """兼容 dict/TypedDict 与对象属性两种 BackendProtocol 返回类型。"""
        if isinstance(result, dict):
            return result.get(field)
        return getattr(result, field, None)

    @staticmethod
    def _get_file_info_path(file_info: FileInfo) -> str:
        """兼容 dict/TypedDict 与对象属性两种 FileInfo 表示。"""
        if isinstance(file_info, dict):
            return file_info.get("path", "")
        return getattr(file_info, "path", "")

    # -----------------------------------------------------------------
    # BackendProtocol 接口实现 —— 文件操作
    # -----------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """列出目录下的文件和子目录。

        实现逻辑：
        1. 查询所有以 path 为前缀的文件记录
        2. 去掉公共前缀后，按 "/" 分割判断是"当前目录下的文件"还是"子目录"
        3. 子目录只展示目录名（不递归展开），文件则计算大小

        示例：path="/src/" 时，若数据库有 /src/a.py、/src/sub/b.py
        → 返回 FileInfo(path="/src/a.py", is_dir=False) 和 FileInfo(path="/src/sub/", is_dir=True)
        """
        self._ensure_connection()
        normalized_path = self._normalize_path(path)

        if normalized_path != "/" and self._get_file(normalized_path) is not None:
            return LsResult(error=f"Path '{path}': not_a_directory", entries=None)

        # 确保路径以 / 结尾，统一前缀匹配格式
        path = self._directory_prefix(normalized_path)

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `file_path`, `modified_at` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                (path + "%",)
            )
            rows = cursor.fetchall()

        if not rows and normalized_path != "/":
            return LsResult(error=f"Path '{path.rstrip('/')}': path_not_found", entries=None)

        entries: dict[str, FileInfo] = {}  # 用字典去重（同一子目录可能出现多次）

        for file_path, modified_at in rows:
            # 获取相对路径：去掉公共前缀后剩余的部分
            relative = file_path[len(path):]
            if not relative:
                continue  # 跳过路径本身（即 path 恰好是一条记录的情况）

            if "/" in relative:
                # 相对路径中还有 "/"，说明是子目录中的文件
                # 只取第一段作为子目录名，不递归展开
                subdir_name = relative.split("/")[0]
                subdir_path = path + subdir_name + "/"
                if subdir_path not in entries:
                    entries[subdir_path] = FileInfo(
                        path=subdir_path,
                        is_dir=True,
                        size=0,
                        modified_at="",
                    )
            else:
                # 相对路径中没有 "/"，是当前目录下的直接文件
                entry_path = path + relative
                # 查询文件大小（content 字段的字节长度）
                cursor.execute(
                    f"SELECT LENGTH(`content`) FROM `{self.table_name}` WHERE `file_path` = %s",
                    (entry_path,)
                )
                size_row = cursor.fetchone()
                size = size_row[0] if size_row and size_row[0] else 0

                entries[entry_path] = FileInfo(
                    path=entry_path,
                    is_dir=False,
                    size=size,
                    modified_at=modified_at.isoformat() if modified_at else "",
                )

        return LsResult(entries=sorted(entries.values(), key=self._get_file_info_path))

    def ls_info(self, path: str) -> list[FileInfo]:
        """兼容 deepagents 0.5 之前仍调用 ls_info 的旧入口。"""
        result = self.ls(path)
        error = self._get_result_field(result, "error")
        if error is not None:
            raise NotImplementedError(error)
        return self._get_result_field(result, "entries") or []

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """读取文件内容（按行分页）。

        Args:
            file_path: 文件路径
            offset:    起始行号（从 0 开始），用于跳过前 N 行
            limit:     最多返回的行数，默认 2000

        返回原始文件内容片段；行号展示由 deepagents 的工具层负责。

        编码处理：如果文件以 base64 存储且不能解码为 UTF-8，会按 base64 返回。
        """
        file_path = self._normalize_path(file_path)
        file_data = self._get_file(file_path)
        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")

        content = file_data["content"]
        encoding = file_data.get("encoding", "utf-8")

        decoded_content = self._decode_text(content, encoding)
        if decoded_content is None:
            return ReadResult(file_data=FileData(
                content=content,
                encoding="base64",
                created_at=file_data.get("created_at"),
                modified_at=file_data.get("modified_at"),
            ))
        content = decoded_content

        # 按行分割并做切片分页
        lines = content.splitlines(keepends=True)
        if offset >= len(lines) and lines:
            return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")
        result_content = "".join(lines[offset:offset + limit])

        return ReadResult(file_data=FileData(
            content=result_content,
            encoding="utf-8",
            created_at=file_data.get("created_at"),
            modified_at=file_data.get("modified_at"),
        ))

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """写入文件内容；文件不存在时创建，已存在时覆盖。

        这是当前 BackendProtocol 的 write 语义。需要精确替换已有内容时，
        应使用 edit 方法。
        """
        file_path = self._normalize_path(file_path)

        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO `{self.table_name}` (`file_path`, `content`, `encoding`)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    `content` = VALUES(`content`),
                    `encoding` = VALUES(`encoding`)
                """,
                (file_path, content, "utf-8"),
            )

        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """编辑文件内容——精确字符串替换。

        Args:
            file_path:   文件路径
            old_string:  要被替换的原始字符串（必须精确匹配）
            new_string:  替换后的新字符串
            replace_all: 是否替换所有匹配项（默认只替换第一个匹配）

        返回值中包含 occurrences 字段，表示实际替换了几处。

        编码处理：base64 文件按二进制文件处理，不允许文本替换，避免破坏内容。
        """
        file_path = self._normalize_path(file_path)
        file_data = self._get_file(file_path)
        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data["content"]
        encoding = file_data.get("encoding", "utf-8")

        if encoding == "base64":
            return EditResult(error=f"Error: File '{file_path}' is binary and cannot be edited as text")
        if old_string == new_string:
            return EditResult(error="Error: old_string and new_string must be different")

        # 执行替换
        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(error=f"Error: The exact string '{old_string}' was not found in the file.")
        if not replace_all and occurrences > 1:
            return EditResult(
                error=(
                    f"Error: The exact string '{old_string}' appears {occurrences} times. "
                    "Use replace_all=True or provide a more specific old_string."
                )
            )

        if replace_all:
            # 替换所有匹配项
            new_content = content.replace(old_string, new_string)
        else:
            # 只替换第一个匹配项
            new_content = content.replace(old_string, new_string, 1)

        # 将替换后的内容写回数据库
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE `{self.table_name}` SET `content` = %s WHERE `file_path` = %s",
                (new_content, file_path)
            )

        return EditResult(path=file_path, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """搜索文件内容——纯文本子串匹配。

        Args:
            pattern: 要搜索的文本模式（子串匹配，非正则）
            path:    限定搜索的目录前缀，如 "/src/"
            glob:    文件名过滤模式，仅支持简单通配符（* 和 ?）

        实现逻辑：
        1. 从数据库取出文件内容（可按 path 过滤）
        2. 对每行做子串匹配
        3. 可选地用 glob 模式过滤文件名

        注意：当前实现是应用层逐行扫描，文件多时性能有限。
        生产环境可考虑用 MySQL FULLTEXT 索引或外部搜索引擎优化。
        """
        try:
            self._ensure_connection()
            search_prefix = self._directory_prefix(path) if path else None

            # 第一步：从数据库取出候选文件
            with self.connection.cursor() as cursor:
                if search_prefix:
                    # 按 path 前缀过滤，缩小扫描范围
                    cursor.execute(
                        f"SELECT `file_path`, `content`, `encoding` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                        (search_prefix + "%",),
                    )
                else:
                    # 无 path 限定，扫描全部文件
                    cursor.execute(
                        f"SELECT `file_path`, `content`, `encoding` FROM `{self.table_name}`"
                    )
                rows = cursor.fetchall()

            matches: list[GrepMatch] = []

            for file_path, content, encoding in rows:
                # 第二步：glob 模式过滤文件名
                if glob and not fnmatch.fnmatch(file_path.lstrip("/"), glob.lstrip("/")):
                    continue  # 文件名不匹配，跳过

                decoded_content = self._decode_text(content, encoding)
                if decoded_content is None:
                    continue

                # 第三步：逐行搜索匹配内容
                lines = decoded_content.split("\n")
                for line_num, line in enumerate(lines, start=1):
                    if pattern in line:
                        matches.append(GrepMatch(
                            path=file_path,
                            line=line_num,
                            text=line[:500]  # 截断长行，避免返回数据过大
                        ))

            return GrepResult(matches=matches)
        except Exception as exc:
            return GrepResult(error=f"grep failed: {exc}", matches=None)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """按文件路径模式匹配文件。

        Args:
            pattern: glob 模式，如 "**/*.py"、"src/*.txt"
            path:    限定搜索的目录前缀

        实现逻辑：
        先用 SQL LIKE 按目录前缀粗筛，再用 fnmatch 做 glob 精确匹配。
        """
        try:
            self._ensure_connection()
            search_prefix = self._directory_prefix(path) if path else "/"

            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT `file_path`, LENGTH(`content`) as `size`, `modified_at` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                    (search_prefix + "%",),
                )
                rows = cursor.fetchall()

            matches: list[FileInfo] = []
            for file_path, size, modified_at in rows:
                relative_path = file_path[len(search_prefix):] if path else file_path.lstrip("/")
                if not fnmatch.fnmatch(relative_path, pattern.lstrip("/")):
                    continue
                matches.append(FileInfo(
                    path=file_path,
                    is_dir=False,
                    size=size or 0,
                    modified_at=modified_at.isoformat() if modified_at else "",
                ))

            return GlobResult(matches=sorted(matches, key=self._get_file_info_path))
        except Exception as exc:
            return GlobResult(error=f"glob failed: {exc}", matches=None)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """兼容 deepagents 0.5 之前仍调用 glob_info 的旧入口。"""
        result = self.glob(pattern, path)
        error = self._get_result_field(result, "error")
        if error is not None:
            raise NotImplementedError(error)
        return self._get_result_field(result, "matches") or []

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """兼容 deepagents 0.5 之前仍调用 grep_raw 的旧入口。"""
        result = self.grep(pattern, path, glob)
        error = self._get_result_field(result, "error")
        if error is not None:
            return error
        return self._get_result_field(result, "matches") or []

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """批量上传文件（将 bytes 写入虚拟文件系统）。

        Args:
            files: 列表，每项为 (文件路径, 文件二进制内容) 的元组

        编码策略：
        - 优先尝试 UTF-8 解码为文本存储（节省空间，支持后续 edit/read 操作）
        - UTF-8 解码失败（如图片、压缩包等二进制文件），则用 base64 编码存储
        """
        responses: list[FileUploadResponse] = []

        for file_path, content in files:
            try:
                file_path = self._normalize_path(file_path)

                # 优先尝试 UTF-8 文本解码
                try:
                    content_str = content.decode("utf-8")
                    encoding = "utf-8"
                except UnicodeDecodeError:
                    # 二进制文件，用 base64 编码存储
                    content_str = base64.standard_b64encode(content).decode("ascii")
                    encoding = "base64"

                self._ensure_connection()
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        INSERT INTO `{self.table_name}` (`file_path`, `content`, `encoding`)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            `content` = VALUES(`content`),
                            `encoding` = VALUES(`encoding`)
                        """,
                        (file_path, content_str, encoding),
                    )
                responses.append(FileUploadResponse(path=file_path, error=None))

            except Exception as e:
                responses.append(FileUploadResponse(path=file_path, error=str(e)))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """批量下载文件（从虚拟文件系统读取为 bytes）。

        Args:
            paths: 要下载的文件路径列表

        编码策略：
        - utf-8 文件：直接编码为 bytes
        - base64 文件：先解码还原为原始二进制内容
        """
        responses: list[FileDownloadResponse] = []

        for file_path in paths:
            file_path = self._normalize_path(file_path)
            file_data = self._get_file(file_path)
            if file_data is None:
                responses.append(FileDownloadResponse(
                    path=file_path,
                    content=None,
                    error="file_not_found"
                ))
                continue

            content = file_data["content"]
            encoding = file_data.get("encoding", "utf-8")

            # 根据编码方式将内容还原为 bytes
            try:
                if encoding == "base64":
                    content_bytes = base64.standard_b64decode(content)  # 二进制文件：base64 → bytes
                else:
                    content_bytes = content.encode("utf-8")             # 文本文件：str → bytes
                responses.append(FileDownloadResponse(
                    path=file_path,
                    content=content_bytes,
                    error=None
                ))
            except Exception as e:
                responses.append(FileDownloadResponse(
                    path=file_path,
                    content=None,
                    error=str(e)
                ))

        return responses

    def close(self) -> None:
        """关闭 MySQL 连接，释放资源。"""
        if self.connection and self.connection.open:
            self.connection.close()


# =====================================================================
# 2. 创建 MySQL 文件系统 Agent
# =====================================================================
agent = create_deep_agent(
    model=glm_llm,         # 使用 GLM 大模型作为 Agent 的 LLM
    tools=[web_search],    # 赋予 Agent 网页搜索能力
    checkpointer=InMemorySaver(),  # 保持同一 thread_id 下的多轮对话上下文
    backend=MySQLBackend.from_conn_string(MYSQL_DATABASE_URL),  # MySQL 虚拟文件系统后端
    system_prompt=(
        "你是一个助手，请根据用户输入的指令，进行相应的文件操作。"
        "你可以通过读写文件来保存和获取信息。"
        "当用户询问当前对话中已经提到的偏好、事实或上下文时，优先根据对话历史回答；"
        "只有明确需要查询持久文件时，再使用文件工具。"
    )
)


# =====================================================================
# 3. 直接运行入口 —— 便于本地验证 CustomizeBackend
# =====================================================================
async def stream_agent_interaction(thread_id: str) -> AsyncIterator[str]:
    """启动一个简单命令行循环，直接验证 MySQLBackend 注入后的 Agent。"""
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n\n[用户] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n对话结束。")
            break

        if user_input.lower() in ("quit", "exit", "退出", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        print("\n[Agent] >>> ", end="", flush=True)
        inputs = {"messages": [{"role": "user", "content": user_input}]}

        try:
            for chunk in agent.stream(inputs, config=config, stream_mode="messages", subgraphs=False):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    token, _metadata = chunk

                    if hasattr(token, "content") and token.content is not None:
                        content_str = str(token.content)
                        if content_str:
                            yield content_str

                    if hasattr(token, "tool_call_chunks") and token.tool_call_chunks:
                        for tool_chunk in token.tool_call_chunks:
                            if tool_chunk and hasattr(tool_chunk, "get") and tool_chunk.get("name"):
                                yield f"\n[调用工具: {tool_chunk['name']}]\n"
                else:
                    print(f"\n[调试] 意外的 chunk 结构: {type(chunk)}")
        except Exception as exc:
            yield f"\nAgent 执行出错: {exc}\n"


async def main() -> None:
    """运行 CustomizeBackend 的命令行测试入口。"""
    async for response in stream_agent_interaction("customize_backend_demo"):
        print(response, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
