"""Harbor agent that installs this repository's Agent harness in a task container and runs it on the task instruction."""
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import override

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

HOME = "/opt/mg"
PIP = "requests loguru pydantic fastapi pillow numpy beautifulsoup4 lxml"
OUTPUT = EnvironmentPaths.agent_dir / "events.jsonl"
STDERR = EnvironmentPaths.agent_dir / "stderr.log"


class MangaAgent(BaseInstalledAgent):
    capabilities = AgentCapabilities()
    options_model = InstalledAgentOptions

    @staticmethod
    @override
    def name() -> str:
        return "manga-agent"

    @override
    def version(self) -> str | None:
        return os.environ.get("MANGA_AGENT_SHA", "dev")

    @override
    def get_version_command(self) -> str | None:
        return None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=("set -e; if ! command -v curl >/dev/null 2>&1; then "
                     "(apt-get update && apt-get install -y curl ca-certificates) >/dev/null 2>&1 || apk add --no-cache curl ca-certificates >/dev/null 2>&1 || yum install -y curl >/dev/null 2>&1; fi; "
                     f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={HOME}/bin UV_NO_MODIFY_PATH=1 sh"),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await environment.upload_file(Path(os.environ["MANGA_AGENT_TGZ"]), "/tmp/manga-agent.tgz")
        await self.exec_as_root(
            environment,
            command=(f"set -e; mkdir -p {HOME}/src && tar xzf /tmp/manga-agent.tgz -C {HOME}/src && "
                     f"{HOME}/bin/uv venv {HOME}/venv --python 3.12 && "
                     f"{HOME}/bin/uv pip install --python {HOME}/venv/bin/python {PIP} && "
                     f"chmod -R a+rX {HOME}"),
        )

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        base = os.environ["MANGA_AGENT_BASE"]
        model = os.environ["MANGA_AGENT_MODEL"]
        minutes = int(os.environ.get("MANGA_AGENT_MINUTES", "150"))
        profile = json.dumps({"advisor": True, "token_budget": 10 ** 12, "max_steps": 2000})
        var = f"MANGA_INSTRUCTION_{uuid.uuid4().hex}".upper()
        env = {"AGENT_API_KEY": "relay", "PYTHONPATH": f"{HOME}/src", "HOME": f"{HOME}/home", var: instruction}
        await self.exec_as_agent(
            environment,
            command=(f"mkdir -p {HOME}/home/.manga-agent {EnvironmentPaths.agent_dir} && printf '%s' {shlex.quote(profile)} > {HOME}/home/.manga-agent/profile.json; "
                     f'INSTR="${var}"; unset {var}; '
                     f"{HOME}/venv/bin/python -m app.agent.cli exec \"$INSTR\" --base {shlex.quote(base)} --model {shlex.quote(model)} --provider polargrid "
                     f'--workspace "$PWD" --mode auto --sandbox full-access --timeout-min {minutes} --json '
                     f"> {OUTPUT} 2> {STDERR} < /dev/null"),
            env=env,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / "events.jsonl"
        if not path.exists():
            return
        result = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") == "result":
                result = row
        usage = result.get("usage") or {}
        context.n_input_tokens = int(usage.get("prompt_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cached_tokens") or 0)
        context.n_output_tokens = int(usage.get("completion_tokens") or 0)
        context.metadata = {"status": result.get("status"), "stats": result.get("stats")}
