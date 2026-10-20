"""Harbor agent that installs this repository's Chaos harness in a task container and runs it on the task instruction."""
import json
import os
import shlex
import tomllib
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
        return "chaos"

    @override
    def version(self) -> str | None:
        return os.environ.get("MANGA_AGENT_SHA", "dev")

    @override
    def get_version_command(self) -> str | None:
        return None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # The runner's own uv goes in first, so a task image without curl or a package manager can still install the agent.
        await environment.upload_file(Path(os.environ["MANGA_AGENT_UV"]), "/tmp/uv-runner")
        await self.exec_as_root(
            environment,
            command=(f"set -e; mkdir -p {HOME}/bin; if chmod 755 /tmp/uv-runner && /tmp/uv-runner --version >/dev/null 2>&1; then "
                     f"mv /tmp/uv-runner {HOME}/bin/uv; exit 0; fi; "
                     "if ! command -v curl >/dev/null 2>&1; then "
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

    def _minutes(self) -> int:
        """The task's own agent time limit less a minute for start-up, so the agent's clock matches the one that stops it."""
        cap = int(os.environ.get("MANGA_AGENT_MINUTES", "60"))
        task = Path(self.logs_dir).parent.name.rsplit("__", 1)[0]
        try:
            with open(Path(os.environ.get("MANGA_AGENT_TASKS_DIR", "/tmp/tb21/tasks")) / task / "task.toml", "rb") as f:
                seconds = float(tomllib.load(f)["agent"]["timeout_sec"])
        except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
            return cap
        return max(1, min(cap, int(seconds // 60) - 1))

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        base = os.environ["MANGA_AGENT_BASE"]
        model = os.environ["MANGA_AGENT_MODEL"]
        minutes = self._minutes()
        max_output = int(os.environ.get("MANGA_AGENT_MAX_OUTPUT", "16000"))
        request = json.loads(os.environ.get("MANGA_AGENT_REQUEST") or "{}")
        # No token or step ceiling: the task's clock is the only limit.
        # Services the task asks for must still run when the verifier checks them.
        profile = {"token_budget": 10 ** 15, "max_steps": 100_000, "max_output_tokens": max_output, "keep_jobs": True,
                   "models": {".": {"request": request, "max_output": max_output}}}
        echo = os.environ.get("MANGA_AGENT_ECHO_REASONING", "")
        if echo:
            profile["echo_reasoning"] = echo if echo == "all" else echo == "true"
        profile = json.dumps(profile)
        var = f"MANGA_INSTRUCTION_{uuid.uuid4().hex}".upper()
        env = {"PYTHONPATH": f"{HOME}/src", "HOME": f"{HOME}/home", var: instruction}
        if os.environ.get("MANGA_AGENT_SEARCH"):
            # TinyFish through the search relay, which adds the real key.
            search = os.environ["MANGA_AGENT_SEARCH"]
            env |= {"TINYFISH_API_KEY": "relay", "TINYFISH_SEARCH_URL": f"{search}/search", "TINYFISH_FETCH_URL": f"{search}/fetch"}
        await self.exec_as_agent(
            environment,
            command=(f"mkdir -p {HOME}/home/.manga-agent {EnvironmentPaths.agent_dir} && printf '%s' {shlex.quote(profile)} > {HOME}/home/.manga-agent/profile.json; "
                     f'INSTR="${var}"; unset {var}; '
                     f'printf "%s" "$INSTR" | {HOME}/venv/bin/python {HOME}/src/run_agent.py - --base {shlex.quote(base)} --model {shlex.quote(model)} '
                     f'--workspace "$PWD" --minutes {minutes} > {OUTPUT} 2> {STDERR}'),
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
