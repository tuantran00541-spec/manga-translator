"""Harbor agent that installs DeepSeek Harness (dsh) in a task container and runs it headless on the task instruction."""
import json
import os
import tempfile
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

HOME = "/opt/dsh"
NODE_VERSION = "v24.20.0"
OUTPUT = EnvironmentPaths.agent_dir / "events.ndjson"
STDERR = EnvironmentPaths.agent_dir / "stderr.log"


# The benchmark's honesty rule: solve the task, never look up its tests or solutions.
BENCH_RULE = ("This task is part of a benchmark. Do not search for, download or read the benchmark's own task pages, tests, solutions or "
              "verifier files (for example Terminal-Bench or Harbor repositories, datasets or /logs/verifier); solve the task from its "
              "instruction and environment.")


def _overlay(base: str, model: str, max_output: int) -> list:
    """The dsh profile patch: one custom provider route at the run's relay, reasoning pinned to its max level."""
    return [
        {"id": "llm-pi-ai", "config": {"providers": {"commandcode": {
            "apiKeyEnv": "DSH_RELAY_KEY",
            "api": "openai-completions",
            "baseURL": base,
            "reasoning": "max",
            # No x-cmd-zdr header: beam has no zero-retention upstream and the provider rejects the run with a 422.
            "headers": {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"},
            "models": [{"id": model, "maxTokens": max_output,
                         "reasoningEfforts": {"off": None, "max": "max"},
                         "compat": {"supportsReasoningEffort": True}}]}}}},
        {"id": "agent-default-model", "config": {"provider": "commandcode", "model": model}},
        # A patch replaces the entry's whole config, so the stock persona line is kept and the rule rides along.
        {"id": "system-prompt", "config": {"personaPrefix": "You are a coding agent powered by the {{model}} model. " + BENCH_RULE}},
    ]


class DshAgent(BaseInstalledAgent):
    capabilities = AgentCapabilities()
    options_model = InstalledAgentOptions

    @staticmethod
    @override
    def name() -> str:
        return "dsh"

    @override
    def version(self) -> str | None:
        return os.environ.get("DSH_AGENT_VERSION", "0.2.1-rc.2")

    @override
    def get_version_command(self) -> str | None:
        return None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # dsh is an npm package and distro Node is too old to run it, so the official tarball goes in.
        await self.exec_as_root(
            environment,
            command=(
                f"set -e; mkdir -p {HOME}/home; "
                "if ! command -v curl >/dev/null 2>&1; then "
                "(apt-get update && apt-get install -y curl ca-certificates) >/dev/null 2>&1 || apk add --no-cache curl ca-certificates >/dev/null 2>&1 || yum install -y curl >/dev/null 2>&1; fi; "
                "arch=$(uname -m); case \"$arch\" in x86_64) na=x64;; aarch64|arm64) na=arm64;; *) echo \"unsupported arch: $arch\" >&2; exit 1;; esac; "
                f"curl -fsSL https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-$na.tar.gz | tar xz -C {HOME} --strip-components=1; "
                f"{HOME}/bin/npm install -g --allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,@google/genai,protobufjs @deepseek-ai/dsh >/dev/null 2>&1; "
                f"{HOME}/bin/dsh --version; "
                f"chmod -R a+rX {HOME}; chmod 777 {HOME}/home; "
                # Hosts that only serve this benchmark's tasks and tests are unreachable from the task.
                "printf '0.0.0.0 hub.harborframework.com\\n0.0.0.0 tbench.ai\\n0.0.0.0 www.tbench.ai\\n' >> /etc/hosts || true"),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    def _minutes(self) -> int:
        """The task's own agent time limit less a minute for start-up, so the agent's clock matches the one that stops it."""
        cap = int(os.environ.get("DSH_AGENT_MINUTES", "60"))
        task = Path(self.logs_dir).parent.name.rsplit("__", 1)[0]
        try:
            with open(Path(os.environ.get("DSH_AGENT_TASKS_DIR", "/tmp/tb21/tasks")) / task / "task.toml", "rb") as f:
                seconds = float(tomllib.load(f)["agent"]["timeout_sec"])
        except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
            return cap
        return max(1, min(cap, int(seconds // 60) - 1))

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        base = os.environ["DSH_AGENT_BASE"]
        model = os.environ["DSH_AGENT_MODEL"]
        max_output = int(os.environ.get("DSH_AGENT_MAX_OUTPUT", "64000"))
        minutes = self._minutes()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(_overlay(base, model, max_output), f)
            overlay = Path(f.name)
        await environment.upload_file(overlay, f"{HOME}/overlay.json")
        var = f"DSH_INSTRUCTION_{uuid.uuid4().hex}".upper()
        env = {"DSH_HOME": f"{HOME}/home", "DSH_RELAY_KEY": "relay",
               "PATH": f"{HOME}/bin:/usr/local/bin:/usr/bin:/bin", var: instruction}
        await self.exec_as_agent(
            environment,
            command=(f"mkdir -p {EnvironmentPaths.agent_dir}; "
                     f'INSTR="${var}"; unset {var}; '
                     f"timeout -s TERM {minutes * 60} {HOME}/bin/dsh --profile headless --patch {HOME}/overlay.json "
                     f'headless --json "$INSTR" > {OUTPUT} 2> {STDERR}'),
            env=env,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / "events.ndjson"
        if not path.exists():
            return
        final = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") == "final":
                final = row
        # dsh's headless events carry no token usage, so only the outcome is recorded.
        context.metadata = {"status": "finished" if final else "no-final-event",
                            "final_text": (final.get("text") or "")[:200]}
