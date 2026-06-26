from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore

from agent.my_tools import web_search
from model.init_chat_model import glm_llm

workspace_root = Path(__file__).resolve().parent / "agent_workspace"

composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/": FilesystemBackend(
            root_dir=str(workspace_root / "memories"),
            virtual_mode=True
        ),
        "/shared_docs/": FilesystemBackend(
            root_dir=str(workspace_root / "shared_docs"),
            virtual_mode=True
        ),
    }
)

agent = create_deep_agent(
    model=glm_llm,
    tools=[web_search],
    store=InMemoryStore(),
    backend=composite_backend,
    system_prompt='你是一个助手，请根据用户输入的指令，进行相应的操作。'
)
