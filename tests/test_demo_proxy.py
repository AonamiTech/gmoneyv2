from __future__ import annotations

from pathlib import Path

PROXY_CONFIG = Path(__file__).parents[1] / "infra" / "nginx" / "demo.conf"


def test_demo_proxy_has_no_request_rate_limit() -> None:
    config = PROXY_CONFIG.read_text()
    assert "limit_req" not in config


def test_demo_proxy_allows_multipart_overhead_above_gpu_file_limit() -> None:
    config = PROXY_CONFIG.read_text()
    assert "client_max_body_size 0;" in config
    assert "proxy_request_buffering off;" in config
