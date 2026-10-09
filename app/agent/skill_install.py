"""Install Agent Skills from GitHub, a zip or a SKILL.md into the user's skills folder, and write new ones."""
from __future__ import annotations

import io
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from app.agent.skills import NAME_RE, frontmatter
from app.downloader.http import read_response_limited, safe_get
from loguru import logger

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


def _github_json(owner: str, repo: str, path: str) -> dict:
    """One GitHub API call, with a plain error when it fails."""
    try:
        response = safe_get(f"https://api.github.com/repos/{owner}/{repo}/{path}", timeout=(10, 30),
                            headers={"Accept": "application/vnd.github+json"})
    except Exception as exc:
        raise ValueError(f"không gọi được GitHub API cho {owner}/{repo}: {exc}") from exc
    try:
        data = response.json()
    except Exception as exc:
        raise ValueError(f"GitHub trả về dữ liệu không đọc được cho {owner}/{repo}/{path}") from exc
    finally:
        response.close()
    if not isinstance(data, dict):
        raise ValueError(f"GitHub trả về dữ liệu không đọc được cho {owner}/{repo}/{path}")
    return data


def _resolve_ref(owner: str, repo: str, ref: str) -> str:
    """Pin a branch, tag or HEAD to the commit SHA it points at, so the zip fetched
    below cannot change between inspection and install. Fails closed: a ref that
    cannot be resolved is never downloaded as a mutable name."""
    if ref == "HEAD":
        ref = _github_json(owner, repo, "").get("default_branch") or "main"
    sha = _github_json(owner, repo, f"commits/{ref}").get("sha") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"GitHub không trả về commit hợp lệ cho {owner}/{repo}@{ref}")
    return sha


def _fetch(owner: str, repo: str, ref: str) -> zipfile.ZipFile:
    response = safe_get(f"https://codeload.github.com/{owner}/{repo}/zip/{ref}", timeout=(10, 60))
    try:
        if response.status_code != 200:
            raise ValueError(f"GitHub trả về HTTP {response.status_code} cho {owner}/{repo}")
        return zipfile.ZipFile(io.BytesIO(read_response_limited(response, limit_bytes=MAX_ZIP_BYTES)))
    finally:
        response.close()


MAX_SKILL_BYTES = 200_000


def user_dir(home: Path) -> Path:
    return home / ".manga-agent" / "skills"


def install(spec: str, home: Path) -> list[str]:
    """Copy every skill folder found under the repo path; returns their names.

    The ref is pinned to a commit SHA first, so the zip installed is exactly
    the commit resolved here; the SHA is recorded next to each installed skill.
    """
    owner, repo, ref, sub = parse(spec)
    sha = _resolve_ref(owner, repo, ref)
    names = _install_archive(_fetch(owner, repo, sha), sub, home, f"{spec} (commit {sha[:12]})")
    for name in names:
        try:
            (user_dir(home) / name / ".installed-from").write_text(f"{owner}/{repo}@{sha}\n", encoding="utf-8")
        except OSError:
            pass
    logger.info("Installed skills {} from {}/{}@{}", ", ".join(names), owner, repo, sha)
    return names


def install_zip(data: bytes, home: Path) -> list[str]:
    """Every skill folder inside an uploaded zip, wherever it sits."""
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError("file zip quá lớn (tối đa 40 MB)")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("không đọc được file zip") from exc
    return _install_archive(archive, "", home, "file zip", wrapped=False)


def install_file(text: str, home: Path) -> str:
    """A lone SKILL.md becomes a skill folder named by its front matter."""
    head, _ = frontmatter(text[:20_000])
    name = head.get("name", "")
    if not NAME_RE.match(name) or not head.get("description"):
        raise ValueError("SKILL.md cần phần đầu có name (chữ thường, số, gạch ngang) và description")
    _write(home, name, text)
    return name


def create(home: Path, name: str, description: str, body: str, replace: bool = False) -> str:
    """Write a new SKILL.md from a name, a one-paragraph description and its instructions."""
    name, description, body = name.strip(), " ".join(description.split()), body.strip()
    if not NAME_RE.match(name):
        raise ValueError("tên skill chỉ gồm chữ thường, số và gạch ngang, tối đa 64 ký tự")
    if not description or len(description) > 1024:
        raise ValueError("cần mô tả 1–1024 ký tự: skill làm gì và khi nào dùng")
    if not body:
        raise ValueError("cần nội dung hướng dẫn")
    if not replace and (user_dir(home) / name).exists():
        raise ValueError(f"đã có skill {name}")
    _write(home, name, f"---\nname: {name}\ndescription: >-\n  {description}\n---\n\n{body}\n")
    return name


def read(home: Path, name: str) -> str:
    path = user_dir(home) / name / "SKILL.md"
    if not NAME_RE.match(name) or not path.is_file():
        raise ValueError(f"không có skill {name} của bạn")
    return path.read_text(encoding="utf-8", errors="replace")


def remove(home: Path, name: str) -> None:
    folder = user_dir(home) / name
    if not NAME_RE.match(name) or not (folder / "SKILL.md").is_file():
        raise ValueError(f"không có skill {name} của bạn")
    shutil.rmtree(folder)


def _write(home: Path, name: str, text: str) -> None:
    if len(text.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError("SKILL.md quá dài (tối đa 200 KB)")
    folder = user_dir(home) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")


def _install_archive(archive: zipfile.ZipFile, sub: str, home: Path, source: str, wrapped: bool = True) -> list[str]:
    """Copy the skill folders of an archive; GitHub zips wrap everything in one top folder."""
    entries = [i for i in archive.infolist() if not i.is_dir()]
    root = entries[0].filename.split("/", 1)[0] if entries and wrapped else ""
    prefix = "/".join(p for p in (root, sub) if p) + "/" if root or sub else ""
    found = {i.filename.rsplit("/", 1)[0] if "/" in i.filename else "" for i in entries
             if i.filename.startswith(prefix) and (i.filename == "SKILL.md" or i.filename.endswith("/SKILL.md"))}
    folders = sorted(found)
    if not folders:
        raise ValueError(f"không thấy SKILL.md nào trong {source}")
    target = user_dir(home)
    done = []
    for folder in folders[:MAX_SKILLS]:
        lead = folder + "/" if folder else ""
        files = [i for i in entries if i.filename.startswith(lead)][:MAX_FILES]
        head, _ = frontmatter(archive.read(f"{lead}SKILL.md")[:20_000].decode("utf-8", errors="replace"))
        name = head.get("name") or folder.rsplit("/", 1)[-1]
        if not NAME_RE.match(name):
            continue
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / name
            for info in files:
                relative = Path(info.filename[len(lead):])
                if relative.is_absolute() or ".." in relative.parts or info.file_size > MAX_FILE_BYTES:
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # L2: the per-file cap used to trust info.file_size, which is the DECLARED size in
                # the zip header. A 1MB-declared entry can expand to 1GB; check the actual bytes.
                data = archive.read(info)
                if len(data) > MAX_FILE_BYTES:
                    continue
                destination.write_bytes(data)
            target.mkdir(parents=True, exist_ok=True)
            final = target / name
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(staging), str(final))
        done.append(name)
    if not done:
        raise ValueError("các skill tìm thấy có tên không hợp lệ")
    return done
