"""
MCP 客户端配置模块
==================

本模块负责创建和管理 MCP（Model Context Protocol）多服务客户端实例，
用于连接远端工具服务，供子 Agent 调用。

当前注册的 MCP 服务端：
- data-analyzer  : 数据分析报表服务（ModelScope 托管）
- ticket-booking : 12306 购票搜索服务（ModelScope 托管）

关键设计：
- 使用 MultiServerMCPClient 统一管理多个 MCP 服务端连接；
- 初始化失败时降级为 None，保证本地工具和不依赖 MCP 的子 Agent
  仍可正常工作（容错机制）。
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

# =====================================================================
# 1. MCP 服务端配置 —— 声明远端工具服务的连接地址与传输协议
# =====================================================================

# 数据分析报表的 MCP 服务端配置
# - url       : ModelScope 托管的 MCP 服务地址
# - transport : 使用 streamable_http 传输协议（支持流式响应）
data_analysis_mcp_config = {
    "url": "https://mcp.api-inference.modelscope.net/f55a54972c2040/mcp",
    "transport": "streamable_http",
}

# 搜索 12306 购票的 MCP 服务端配置
# - url       : ModelScope 托管的 MCP 服务地址
# - transport : 使用 streamable_http 传输协议
ticket_booking_mcp_config = {
    "url": "https://mcp.api-inference.modelscope.net/b18d4e4dccdf4f/mcp",
    "transport": "streamable_http",
}


# =====================================================================
# 2. 创建 MCP 客户端实例 —— 容错初始化，失败时降级为 None
# =====================================================================

# ⭐ 容错设计：如果远端服务不可达，打印警告而非崩溃，
#   保证本地工具和不需要 MCP 的子 Agent 仍可正常使用。
#   下游代码通过判断 mcp_client is None 来跳过 MCP 工具加载。
try:
    mcp_client = MultiServerMCPClient({
        "data-analyzer": data_analysis_mcp_config,
        "ticket-booking": ticket_booking_mcp_config,
    })
except Exception as exc:
    print(f"[MCP] 客户端初始化失败，MCP 工具将不可用: {exc}")
    mcp_client = None  # type: ignore[assignment]