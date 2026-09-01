from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.downloader.http import safe_download_file
from app.parameters import DOWNLOAD_WORKER_LIMIT


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
        image_urls = self.extract_image_urls(chapter_url)
        return self.download_urls(image_urls, output_dir, referer=chapter_url)

    def download_urls(
        self,
        image_urls: list[str],
        output_dir: Path,
        *,
        referer: str,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        urls = self._dedupe(image_urls)
        if not urls:
            return []

        work: list[tuple[str, Path]] = []
        for i, img_url in enumerate(urls):
            ext = self._guess_ext(img_url)
            work.append((img_url, output_dir / f"{i:03d}{ext}"))

        max_workers = min(DOWNLOAD_WORKER_LIMIT, len(work))
        if max_workers <= 1:
            for img_url, out_path in work:
                self._download_file(img_url, out_path, referer=referer)
            return [path for _url, path in work]

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chapter-download") as pool:
            futures = {
                pool.submit(self._download_file, img_url, out_path, referer): out_path
                for img_url, out_path in work
            }
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise

        return [path for _url, path in work]

    def _download_file(self, url: str, out_path: Path, referer: str) -> None:
        headers = dict(self.headers)
        headers["Referer"] = referer
        safe_download_file(url, out_path, headers=headers)

    @staticmethod
    def _dedupe(urls: list[str]) -> list[str]:
        seen = set()
        result = []
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                result.append(url)
        return result

    @staticmethod
    def _guess_ext(url: str) -> str:
        clean_url = url.lower().split("?")[0].split("#")[0]
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            if clean_url.endswith(ext):
                return ext
        return ".jpg"
