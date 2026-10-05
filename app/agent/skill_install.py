"""Install Agent Skills from a public GitHub repository into the user's skills folder."""
from __future__ import annotations

import io
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from app.agent.skills import NAME_RE, frontmatter
from app.downloader.http import read_response_limited, safe_get

SPEC_RE = re.compile(r"^([\w.-]+)/([\w.-]+)(?:@([\w.-]+))?(?:/([\w./-]+))?$")
MAX_ZIP_BYTES = 40_000_000
MAX_FILE_BYTES = 2_000_000
MAX_FILES = 400
MAX_SKILLS = 40


def parse(spec: str) -> tuple[str, str, str, str]:
    """owner/repo, optionally @ref and /folder inside it, such as obra/superpowers@main/skills."""
    match = SPEC_RE.match(spec.strip().removeprefix("https://github.com/").strip("/"))
    if not match:
        raise ValueError("dùng dạng CHỦ/REPO, CHỦ/REPO@nhánh hoặc CHỦ/REPO/thư-mục")
    owner, repo, ref, sub = match.groups()
    return owner, repo, ref or "HEAD", (sub or "").strip("/")


def _fetch(owner: str, repo: str, ref: str) -> zipfile.ZipFile:
    response = safe_get(f"https://codeload.github.com/{owner}/{repo}/zip/{ref}", timeout=(10, 60))
    try:
        if response.status_code != 200:
            raise ValueError(f"GitHub trả về HTTP {response.status_code} cho {owner}/{repo}")
        return zipfile.ZipFile(io.BytesIO(read_response_limited(response, limit_bytes=MAX_ZIP_BYTES)))
    finally:
        response.close()


def install(spec: str, home: Path) -> list[str]:
    """Copy every skill folder found under the repo path; returns their names."""
    owner, repo, ref, sub = parse(spec)
    archive = _fetch(owner, repo, ref)
    entries = [i for i in archive.infolist() if not i.is_dir()]
    root = entries[0].filename.split("/", 1)[0] if entries else ""
    prefix = f"{root}/{sub}/" if sub else f"{root}/"
    folders = sorted({i.filename.rsplit("/", 1)[0] for i in entries if i.filename.startswith(prefix) and i.filename.endswith("/SKILL.md")})
    if not folders:
        raise ValueError(f"không thấy SKILL.md nào trong {spec}")
    target = home / ".manga-agent" / "skills"
    done = []
    for folder in folders[:MAX_SKILLS]:
        files = [i for i in entries if i.filename.startswith(folder + "/")][:MAX_FILES]
        head, _ = frontmatter(archive.read(f"{folder}/SKILL.md")[:20_000].decode("utf-8", errors="replace"))
        name = head.get("name") or folder.rsplit("/", 1)[-1]
        if not NAME_RE.match(name):
            continue
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / name
            for info in files:
                relative = Path(info.filename[len(folder) + 1:])
                if relative.is_absolute() or ".." in relative.parts or info.file_size > MAX_FILE_BYTES:
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
            target.mkdir(parents=True, exist_ok=True)
            final = target / name
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(staging), str(final))
        done.append(name)
    if not done:
        raise ValueError("các skill tìm thấy có tên không hợp lệ")
    return done
