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

import base64
import re  # grep 中 glob 模式匹配需要
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
        try:
            self.connection.ping()  # 轻量级心跳检测
        except pymysql.MySQLError:
            # ping 失败说明连接已断开，先关闭旧连接再重建
            if self.connection and self.connection.open:
                self.connection.close()
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

        # 确保路径以 / 结尾，统一前缀匹配格式
        if not path.endswith("/"):
            path = path + "/"

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `file_path`, `modified_at` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                (path + "%",)
            )
            rows = cursor.fetchall()

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
                        modified_at=""
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
                    modified_at=modified_at.isoformat() if modified_at else ""
                )

        return LsResult(entries=list(entries.values()))

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

        返回格式：每行前带行号和制表符，如 "     1\\tHello World"
        超过 2000 字符的行会被截断并追加 "..."。

        编码处理：如果文件以 base64 存储（如上传的二进制文件），会先解码为 UTF-8 文本。
        """
        file_data = self._get_file(file_path)
        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")

        content = file_data["content"]
        encoding = file_data.get("encoding", "utf-8")

        # 如果是 base64 编码，先解码为文本
        if encoding == "base64":
            try:
                content = base64.standard_b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return ReadResult(error=f"Failed to decode base64 content")

        # 按行分割并做切片分页
        lines = content.split("\n")
        total_lines = len(lines)
        selected_lines = lines[offset:offset + limit]

        # 构建带行号的输出（格式：右对齐 6 位行号 + Tab + 内容）
        result_lines = []
        for i, line in enumerate(selected_lines, start=offset + 1):
            # 截断超长行，避免返回数据过大
            if len(line) > 2000:
                line = line[:2000] + "..."
            result_lines.append(f"{i:6d}\t{line}")

        result_content = "\n".join(result_lines)

        return ReadResult(file_data=FileData(
            content=result_content,
            encoding="utf-8",  # 返回给 Agent 的内容始终是 UTF-8 文本
            created_at=file_data.get("created_at"),
            modified_at=file_data.get("modified_at"),
        ))

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """创建新文件（仅当文件不存在时）。

        注意：如果文件已存在，会返回错误提示。
        这是 BackendProtocol 的语义——"写入新路径"，而非覆盖已有文件。
        覆盖式修改应使用 edit 方法。
        """
        # 检查文件是否已存在
        existing = self._get_file(file_path)
        if existing is not None:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.",
                path=None
            )

        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO `{self.table_name}` (`file_path`, `content`, `encoding`) VALUES (%s, %s, %s)",
                (file_path, content, "utf-8")
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

        编码处理：同 read，base64 文件会先解码再替换，替换后以 UTF-8 存回。
        """
        file_data = self._get_file(file_path)
        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data["content"]
        encoding = file_data.get("encoding", "utf-8")

        # base64 文件先解码为文本再操作
        if encoding == "base64":
            try:
                content = base64.standard_b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return EditResult(error="Failed to decode base64 content")

        # 执行替换
        if replace_all:
            # 替换所有匹配项
            new_content = content.replace(old_string, new_string)
            occurrences = content.count(old_string)
        else:
            # 只替换第一个匹配项
            if old_string not in content:
                return EditResult(error=f"Error: The exact string '{old_string}' was not found in the file.")
            new_content = content.replace(old_string, new_string, 1)
            occurrences = 1

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
        self._ensure_connection()

        # 第一步：从数据库取出候选文件
        with self.connection.cursor() as cursor:
            if path:
                # 按 path 前缀过滤，缩小扫描范围
                cursor.execute(
                    f"SELECT `file_path`, `content` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                    (path + "%",)
                )
            else:
                # 无 path 限定，扫描全部文件
                cursor.execute(
                    f"SELECT `file_path`, `content` FROM `{self.table_name}`"
                )
            rows = cursor.fetchall()

        matches: list[GrepMatch] = []

        for file_path, content in rows:
            # 第二步：glob 模式过滤文件名
            if glob:
                # 简单实现：将 glob 通配符转换为正则表达式
                # * → .* , ? → . , . → \.
                glob_pattern = glob.replace(".", r"\.").replace("*", ".*").replace("?", ".")
                if not re.match(glob_pattern, file_path):
                    continue  # 文件名不匹配，跳过

            # 第三步：逐行搜索匹配内容
            lines = content.split("\n")
            for line_num, line in enumerate(lines, start=1):
                if pattern in line:
                    matches.append(GrepMatch(
                        path=file_path,
                        line=line_num,
                        text=line[:500]  # 截断长行，避免返回数据过大
                    ))

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """按文件路径模式匹配文件。

        Args:
            pattern: glob 模式，如 "**/*.py"、"src/*.txt"
            path:    限定搜索的目录前缀

        实现逻辑：
        将 glob 模式转换为 SQL LIKE 模式，利用数据库索引加速查询。
        转换规则：**/ → %/ , ** → % , * → % , ? → _

        注意：当前 SQL 模式转换逻辑存在边界情况，
        full_pattern 变量构建了完整路径但实际查询未使用，
        生产使用时需重新审视此方法。
        """
        self._ensure_connection()

        # 将 glob 模式转换为 SQL LIKE 模式
        # **/ 匹配任意深度目录 → %/ , ** 匹配任意字符 → % , * 匹配单层 → % , ? 匹配单字符 → _
        sql_pattern = pattern.replace(".", r"\.").replace("**/", "%/").replace("**", "%").replace("*", "%").replace("?", "_")

        if path:
            full_pattern = path.rstrip("/") + "/" + sql_pattern
        else:
            full_pattern = "/" + sql_pattern

        # 转义 % 为字面量，然后追加 % 通配符做前缀匹配
        # TODO: 此处逻辑待确认——先替换 * → % 再转义 %，可能产生非预期结果
        sql_pattern = sql_pattern.replace("%", r"\%") + "%"

        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `file_path`, LENGTH(`content`) as `size`, `modified_at` FROM `{self.table_name}` WHERE `file_path` LIKE %s",
                (sql_pattern,)
            )
            rows = cursor.fetchall()

        matches: list[FileInfo] = []
        for file_path, size, modified_at in rows:
            matches.append(FileInfo(
                path=file_path,
                is_dir=False,
                size=size or 0,
                modified_at=modified_at.isoformat() if modified_at else ""
            ))

        return GlobResult(matches=matches)

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
                # 优先尝试 UTF-8 文本解码
                try:
                    content_str = content.decode("utf-8")
                    encoding = "utf-8"
                except UnicodeDecodeError:
                    # 二进制文件，用 base64 编码存储
                    content_str = base64.standard_b64encode(content).decode("ascii")
                    encoding = "base64"

                # 不覆盖已有文件（与 write 语义一致）
                existing = self._get_file(file_path)
                if existing is not None:
                    responses.append(FileUploadResponse(
                        path=file_path,
                        error=f"File already exists"
                    ))
                    continue

                self._ensure_connection()
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        f"INSERT INTO `{self.table_name}` (`file_path`, `content`, `encoding`) VALUES (%s, %s, %s)",
                        (file_path, content_str, encoding)
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
    backend=MySQLBackend.from_conn_string(MYSQL_DATABASE_URL),  # MySQL 虚拟文件系统后端
    system_prompt=(
        "你是一个助手，请根据用户输入的指令，进行相应的文件操作。"
        "你可以通过读写文件来保存和获取信息。"
    )
)
