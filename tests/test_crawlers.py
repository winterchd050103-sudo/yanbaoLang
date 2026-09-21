"""采集器基类测试：限速、重试、robots 检查（全部 mock 网络，不真实访问）。"""

import time
from unittest.mock import MagicMock, patch

import pytest

from autoreport.data_ingestion.crawlers.base import BaseCrawler, RateLimiter


class DummyCrawler(BaseCrawler):
    source_name = "dummy"

    def fetch_list(self, **kwargs):
        return []

    def download_item(self, item, dest_dir):
        return {}


def test_rate_limiter_enforces_min_interval():
    rl = RateLimiter(delay_seconds=0.3)
    t0 = time.monotonic()
    rl.wait()
    rl.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.3  # 第二次请求必须等够间隔


def test_rate_limiter_no_wait_first_call():
    rl = RateLimiter(delay_seconds=5)
    t0 = time.monotonic()
    rl.wait()
    assert time.monotonic() - t0 < 1  # 首次请求不应等待


def test_request_retry_with_backoff():
    crawler = DummyCrawler()
    crawler.max_retries = 2
    crawler.delay = 0.05

    responses = [MagicMock(raise_for_status=MagicMock(side_effect=Exception("boom")))] * 3
    with patch.object(crawler.session, "get", side_effect=responses) as mock_get:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ConnectionError):
                crawler.get_text("http://example.com/x")
    assert mock_get.call_count == 2  # 重试上限生效


def test_request_success_after_retry():
    crawler = DummyCrawler()
    crawler.max_retries = 3
    crawler.delay = 0.01
    ok = MagicMock()
    ok.raise_for_status = MagicMock()
    ok.json.return_value = {"ok": 1}
    with patch.object(crawler.session, "get",
                      side_effect=[Exception("net"), ok]) as mock_get:
        with patch("time.sleep"):
            data = crawler.get_json("http://example.com/api")
    assert data == {"ok": 1}
    assert mock_get.call_count == 2


def test_robots_blocked(monkeypatch):
    crawler = DummyCrawler()
    fake_rp = MagicMock()
    fake_rp.can_fetch.return_value = False
    crawler.robots._cache["http://blocked.test"] = fake_rp
    with pytest.raises(PermissionError):
        crawler.get_text("http://blocked.test/page")
