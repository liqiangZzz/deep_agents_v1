import os

from dotenv import load_dotenv

# 系统环境变量优先，.env 只补充缺失项。
load_dotenv(override=False)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')

GLM_API_KEY = os.getenv('GLM_API_KEY')
GLM_BASE_URL = os.getenv('GLM_BASE_URL')

MYSQL_DATABASE_URL = os.getenv('MYSQL_DATABASE_URL')
REDIS_DATABASE_URL = os.getenv('REDIS_DATABASE_URL')

# ★ 调试开关：设为 "1" 时在控制台打印 MCP 工具的完整 JSON Schema，
#   默认不输出（仅显示名称、描述和参数名列表）。
#   可在 .env 或系统环境变量中设置。
PRINT_MCP_TOOL_SCHEMA = os.getenv('PRINT_MCP_TOOL_SCHEMA')
