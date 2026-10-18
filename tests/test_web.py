import socket

import pytest

from tac.web import PublicFetcher, public_url, scope_contains


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://[::1]/", "http://192.168.1.1/", "http://10.0.0.1/", "http://169.254.169.254/", "http://0.0.0.0/", "http://224.0.0.1/", "http://[::ffff:127.0.0.1]/", "http://[64:ff9b::c0a8:101]/", "http://[2002:7f00:1::]/"])
def test_private_addresses_blocked(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_dns_mixed_public_private_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("10.1.1.1", 443))])
    with pytest.raises(ValueError):
        public_url("https://example.test/")


def test_scope_path_escape():
    assert scope_contains("https://example.org/docs/", "https://example.org/docs/python.html")
    assert not scope_contains("https://example.org/docs/", "https://example.org/docs2/a")
    assert not scope_contains("https://example.org/docs/", "https://other.org/docs/a")
    assert not scope_contains("https://example.org/docs/", "https://example.org/docs/%2e%2e/private")


def test_redirect_private_destination_blocked(core, monkeypatch):
    import tac.web as web
    original = web.public_url
    def resolve(url):
        if url.startswith("https://public.test"):
            return url, "public.test", 443, ["8.8.8.8"]
        return original(url)
    monkeypatch.setattr(web, "public_url", resolve)
    core.gate.set_enabled(True)
    fetcher = PublicFetcher(core.gate, core.config)
    class Allow:
        def can_fetch(self, *args):
            return True
    monkeypatch.setattr(fetcher, "robot_policy", lambda url: (Allow(), 0))
    monkeypatch.setattr(fetcher, "raw", lambda url, *a: (302, {"Location": "http://127.0.0.1/admin"}, b"", url))
    with pytest.raises(ValueError, match="bloqueados"):
        fetcher.fetch("https://public.test/")


def test_connection_pins_validated_ip_and_verifies_tls(core, monkeypatch):
    import tac.web as web
    core.gate.set_enabled(True)
    monkeypatch.setattr(web, "public_url", lambda url: (url, "example.org", 443, ["93.184.216.34"]))
    captured = {}
    class Response:
        status = 200
        headers = {"Content-Type": "text/html"}
        def read(self, *a, **k):
            return b""
        def close(self):
            pass
    class Pool:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def urlopen(self, *args, **kwargs):
            captured["request"] = kwargs
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(web.urllib3, "HTTPSConnectionPool", Pool)
    PublicFetcher(core.gate, core.config).raw("https://example.org/", 1000)
    assert captured["host"] == "93.184.216.34"
    assert captured["server_hostname"] == "example.org"
    assert captured["assert_hostname"] == "example.org"
    assert captured["request"]["redirect"] is False
    assert captured["request"]["headers"]["Host"] == "example.org"


def test_lowercase_http_headers_supported(core, monkeypatch):
    import tac.web as web
    from urllib3._collections import HTTPHeaderDict
    core.gate.set_enabled(True)
    monkeypatch.setattr(web, "public_url", lambda url: (url, "example.org", 443, ["8.8.8.8"]))
    fetcher = PublicFetcher(core.gate, core.config)
    def raw(url, *args):
        if url.endswith("robots.txt"):
            return 200, HTTPHeaderDict({"content-type": "text/plain"}), b"User-agent: *\nAllow: /", url
        return 200, HTTPHeaderDict({"content-type": "text/html"}), b"<title>Teste</title><p>Texto publico.</p>", url
    monkeypatch.setattr(fetcher, "raw", raw)
    result = fetcher.fetch("https://example.org/docs/")
    assert result[1] == "Teste"
    assert "Texto publico." in result[2]


def test_robots_denies_access(core, monkeypatch):
    import tac.web as web
    core.gate.set_enabled(True)
    monkeypatch.setattr(web, "public_url", lambda url: (url, "example.org", 443, ["8.8.8.8"]))
    fetcher = PublicFetcher(core.gate, core.config)
    requests = []
    def raw(url, *args):
        requests.append(url)
        return 200, {}, b"User-agent: *\nDisallow: /private\n", url
    monkeypatch.setattr(fetcher, "raw", raw)
    with pytest.raises(ValueError, match="robots"):
        fetcher.fetch("https://example.org/private")
    assert requests == ["https://example.org/robots.txt"]


def test_enqueue_without_network_or_learning(core, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DNS não autorizado")))
    core.enqueue_url("https://example.org/docs/", True, terms=True)
    assert core.db.query("SELECT * FROM queue")[0]["status"] == "pendente"


def test_network_failure_recoverable(core, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(socket.gaierror("falha DNS de teste")))
    core.enqueue_url("https://example.org/docs/", terms=True)
    core.gate.set_enabled(True)
    result = core.perform("Site com erro", core.run_queue, learning=True)
    assert result["processed"] == 0
    assert core.db.query("SELECT status FROM queue")[0]["status"] == "erro"
    assert core.chat("Python")["mode"] == "documental"


def test_documentation_crawl_scope_depth_and_page_limit(core, monkeypatch):
    import tac.web as web
    core.gate.set_enabled(True)
    core.config.update(web_pages=2, web_depth=1)
    monkeypatch.setattr(web, "public_url", lambda url: (url, "example.org", 443, ["8.8.8.8"]))
    fetcher = PublicFetcher(core.gate, core.config)
    requested = []
    def fetch(url, scope):
        requested.append(url)
        return url, "Título", "Texto de documentação.", ["/docs/next", "/outside", "https://other.org/docs/", "/docs/third"]
    monkeypatch.setattr(fetcher, "fetch", fetch)
    pages = list(fetcher.crawl("https://example.org/docs/", documentation=True))
    assert len(pages) == 2
    assert requested == ["https://example.org/docs/", "https://example.org/docs/next"]
