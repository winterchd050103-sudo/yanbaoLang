"""采集器基类：限速、重试、礼貌抓取（robots.txt / UA 声明）。

合规与稳定性设计（见方案 3.2 / 5.2 / 11）：
- RateLimiter：单源请求间隔 >= CRAWL_DELAY_SECONDS（默认 3s），并发 1
- robots.txt 检查：爬取前校验路径是否允许（urllib 内置 robotparser）
- 重试：指数退避，网络错误不立即放弃；超过上限记录失败并继续后续任务
- 断点续采靠 crawl_jobs 表（scheduler 层实现），基类只管「单次请求」
"""

from __future__ import annotations

import logging
import time
import urllib.robotparser
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import requests

from autoreport.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """令牌间隔限速器：保证同一站点相邻请求的最小间隔。"""

    def __init__(self, delay_seconds: float) -> None:
        self.delay = max(delay_seconds, 0.5)
        self._last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        remaining = self.delay - (now - self._last)
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


class RobotsPolicy:
    """robots.txt 缓存与校验（只读 robots.txt 本身，不做任何绕过）。"""

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if host not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(f"{host}/robots.txt")
                rp.read()
                self._cache[host] = rp
            except Exception as e:
                logger.warning("robots.txt 获取失败 %s（默认允许，人工确认用途合规）: %s", host, e)
                self._cache[host] = None
        rp = self._cache[host]
        return True if rp is None else rp.can_fetch(self.user_agent, url)


class BaseCrawler(ABC):
    """采集器基类：子类实现 fetch_list / download_item。"""

    source_name: str = "base"

    def __init__(self) -> None:
        s = get_settings()
        self.delay = s.crawl_delay_seconds
        self.max_retries = s.crawl_max_retries
        self.timeout = s.crawl_timeout_seconds
        self.limiter = RateLimiter(s.crawl_delay_seconds)
        self.robots = RobotsPolicy(s.user_agent)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": s.user_agent})

    def get_json(self, url: str, **kwargs) -> dict:
        """带限速与重试的 GET（JSON）。"""
        return self._request(url, parse_json=True, **kwargs)

    def get_text(self, url: str, **kwargs) -> str:
        return self._request(url, parse_json=False, **kwargs)

    def _request(self, url: str, parse_json: bool, **kwargs):
        if not self.robots.allowed(url):
            raise PermissionError(f"robots.txt 不允许抓取: {url}")
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp.json() if parse_json else resp.text
            except Exception as e:
                last_err = e
                backoff = self.delay * (2 ** (attempt - 1))  # 指数退避
                logger.warning("请求失败(%d/%d) %s: %s，%.0fs 后重试",
                               attempt, self.max_retries, url, e, backoff)
                time.sleep(backoff)
        raise ConnectionError(f"重试 {self.max_retries} 次仍失败: {url} ({last_err})")

    def download_pdf(self, url: str, dest_dir, filename: str | None = None) -> str:
        """限速下载 PDF，返回本地路径。"""
        from pathlib import Path

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = filename or url.rstrip("/").split("/")[-1] or "report.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        dest = dest_dir / name

        if not self.robots.allowed(url):
            raise PermissionError(f"robots.txt 不允许抓取: {url}")
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True)
                resp.raise_for_status()
                head = resp.raw.read(5, decode_content=True)
                if head and not head.startswith(b"%PDF"):
                    raise ValueError("响应不是 PDF 内容（可能被反爬页拦截）")
                with open(dest, "wb") as f:
                    f.write(head)
                    for chunk in resp.iter_content(64 * 1024):
                        f.write(chunk)
                return str(dest)
            except Exception as e:
                last_err = e
                dest.unlink(missing_ok=True)
                backoff = self.delay * (2 ** (attempt - 1))
                logger.warning("PDF 下载失败(%d/%d) %s: %s", attempt, self.max_retries, url, e)
                time.sleep(backoff)
        raise ConnectionError(f"PDF 下载失败: {url} ({last_err})")

    @abstractmethod
    def fetch_list(self, **kwargs) -> list[dict]:
        """抓取列表页，返回 [{url, title, ...}] 供下载。"""

    @abstractmethod
    def download_item(self, item: dict, dest_dir) -> dict:
        """下载单条目，返回 {local_path, source_url, ...}。"""
