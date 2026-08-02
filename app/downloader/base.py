from abc import ABC, abstractmethod
from pathlib import Path
import requests
from app.security import validate_url


class BaseAdapter(ABC):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        ...

    @abstractmethod
    def extract_image_urls(self, chapter_url: str) -> list[str]:
        ...

    def download(self, chapter_url: str, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_urls = self.extract_image_urls(chapter_url)
        saved_paths = []
        for i, img_url in enumerate(image_urls):
            ext = self._guess_ext(img_url)
            out_path = output_dir / f"{i:03d}{ext}"
            self._download_file(img_url, out_path, referer=chapter_url)
            saved_paths.append(out_path)
        return saved_paths

    def _download_file(self, url: str, out_path: Path, referer: str) -> None:
        validate_url(url)
        headers = dict(self.headers)
        headers["Referer"] = referer
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

    @staticmethod
    def _guess_ext(url: str) -> str:
        clean_url = url.lower().split("?")[0].split("#")[0]
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            if clean_url.endswith(ext):
                return ext
        return ".jpg"
