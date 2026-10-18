"""HTTP público com DNS fixado por conexão, TLS verificado e redirecionamentos explícitos."""
from __future__ import annotations

import ipaddress
import posixpath
import socket
import threading
import time
from collections import deque
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import certifi
import urllib3

from .documents import html_text, html_references

USER_AGENT = "TAC/0.1 (local document study; user-authorized)"


def public_url(url):
    if len(url) > 4096 or any(ord(c) < 33 for c in url) or "\\" in url:
        raise ValueError("URL inválida.")
    p = urlsplit(url)
    if p.scheme.lower() not in ("http", "https") or not p.hostname or p.username or p.password:
        raise ValueError("Use uma URL pública HTTP/HTTPS, sem credenciais.")
    host = p.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    port = p.port or (443 if p.scheme == "https" else 80)
    if port != (443 if p.scheme == "https" else 80):
        raise ValueError("Somente as portas padrão HTTP/HTTPS são permitidas.")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = sorted({a[4][0] for a in addresses})
    if not ips:
        raise ValueError("O domínio não possui endereço público.")
    for value in ips:
        ip = ipaddress.ip_address(value)
        transition = ip.version == 6 and any(ip in ipaddress.ip_network(net) for net in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))
        if not ip.is_global or ip.is_multicast or ip.is_reserved or transition or (ip.version == 6 and ip.ipv4_mapped and not ip.ipv4_mapped.is_global):
            raise ValueError("Endereços locais, privados e reservados são bloqueados.")
    netloc = f"[{host}]" if ":" in host else host
    path = quote(unquote(p.path or "/"), safe="/%:@!$&'()*+,;=-._~")
    canonical = urlunsplit((p.scheme.lower(), netloc, path, p.query, ""))
    return canonical, host, port, ips


def scope_contains(root, candidate):
    a, b = urlsplit(root), urlsplit(candidate)
    if (a.scheme.lower(), a.netloc.lower()) != (b.scheme.lower(), b.netloc.lower()):
        return False
    # Decodificação repetida impede escapes de caminho por redirecionamentos.
    def clean(path):
        for _ in range(4):
            path = unquote(path)
        return posixpath.normpath(path.replace("\\", "/"))
    prefix = clean(a.path)
    path = clean(b.path)
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


class PublicFetcher:
    def __init__(self, gate, config, progress=lambda *_: None):
        self.gate, self.config = gate, config
        self.last = {}
        self.robots = {}
        self.progress = progress

    def delay(self, origin, extra=0):
        interval = max(self.config.get("web_interval"), extra)
        wait = interval - (time.monotonic() - self.last.get(origin, 0))
        if wait > 0:
            self.progress(0, f"Respeitando intervalo do site: aguardando {wait:.1f} s…")
            if self.gate.cancel.wait(wait):
                self.gate.check()
        self.gate.check()
        self.last[origin] = time.monotonic()

    def raw(self, url, max_bytes, delay=0):
        self.gate.check()
        self.progress(0, f"Verificando endereço público: {url}")
        canonical, host, port, ips = public_url(url)
        p = urlsplit(canonical)
        self.delay((p.scheme, p.netloc), delay)
        # A conexão usa o IP já validado; o hostname é usado apenas em SNI/certificado.
        # Não há segunda resolução DNS nem uso de proxy de ambiente.
        pool_cls = urllib3.HTTPSConnectionPool if p.scheme == "https" else urllib3.HTTPConnectionPool
        kwargs = dict(host=ips[0], port=port, timeout=urllib3.Timeout(connect=8, read=8), maxsize=1)
        if p.scheme == "https":
            kwargs.update(server_hostname=host, assert_hostname=host, cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
        pool = pool_cls(**kwargs)
        response = None
        try:
            self.progress(0, f"Conectando ao site: {canonical}")
            response = pool.urlopen("GET", p.path + ("?" + p.query if p.query else ""),
                                    headers={"Host": p.netloc, "User-Agent": USER_AGENT, "Accept-Encoding": "identity", "Accept": "text/html,text/plain;q=0.9"},
                                    redirect=False, retries=False, preload_content=False)
            # Preserva HTTPHeaderDict: nomes HTTP não distinguem maiúsculas de minúsculas.
            headers = response.headers.copy()
            if response.status in (301, 302, 303, 307, 308):
                return response.status, headers, b"", canonical
            if headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
                raise ValueError("Servidor enviou compressão não solicitada; resposta rejeitada para limitar memória.")
            if int(headers.get("Content-Length", "0")) > max_bytes:
                raise ValueError("Página maior que o limite configurado.")
            body = bytearray()
            deadline = time.monotonic() + 30
            while True:
                self.gate.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("Tempo máximo por página excedido.")
                block = response.read(min(32768, max_bytes + 1 - len(body)), decode_content=False)
                if not block:
                    break
                body.extend(block)
                if len(body) > max_bytes:
                    raise ValueError("Página maior que o limite configurado.")
                if len(body) % (256 * 1024) < 32768:
                    self.progress(0, f"Recebendo página: {len(body) / 1024:.0f} KB de no máximo {max_bytes / 1024:.0f} KB")
            return response.status, headers, bytes(body), canonical
        finally:
            if response is not None:
                response.close()
            pool.close()

    def robot_policy(self, url):
        p = urlsplit(url)
        origin = urlunsplit((p.scheme, p.netloc, "", "", ""))
        if origin not in self.robots:
            self.progress(0, f"Verificando permissão robots.txt: {origin}")
            robot_url = origin + "/robots.txt"
            for _ in range(5):
                status, headers, body, final = self.raw(robot_url, 512000)
                if status in (301, 302, 303, 307, 308):
                    robot_url = urljoin(final, headers.get("Location", ""))
                    if urlsplit(robot_url).netloc != p.netloc or urlsplit(robot_url).scheme != p.scheme:
                        raise ValueError("robots.txt redirecionou para fora da origem autorizada.")
                    continue
                policy = RobotFileParser()
                policy.set_url(origin + "/robots.txt")
                if status in (404, 410):
                    policy.parse(["User-agent: *", "Allow: /"])
                elif status == 200:
                    policy.parse(body.decode("utf-8", errors="replace").splitlines())
                else:
                    raise ValueError(f"robots.txt indisponível/bloqueado (HTTP {status}); acesso cancelado.")
                rate = policy.request_rate(USER_AGENT)
                gap = max(policy.crawl_delay(USER_AGENT) or 0, rate.seconds / rate.requests if rate and rate.requests else 0)
                self.robots[origin] = policy, gap
                break
            else:
                raise ValueError("Muitos redirecionamentos em robots.txt.")
        return self.robots[origin]

    def fetch(self, url, scope=None):
        self.reference_links = []
        for _ in range(6):
            canonical, _, _, _ = public_url(url)
            if scope and not scope_contains(scope, canonical):
                raise ValueError("Página fora do domínio/caminho autorizado.")
            policy, gap = self.robot_policy(canonical)
            if not policy.can_fetch(USER_AGENT, canonical):
                raise ValueError("Acesso não permitido por robots.txt.")
            status, headers, body, final = self.raw(canonical, self.config.get("web_page_mb") * 2**20, gap)
            if status in (301, 302, 303, 307, 308):
                if not headers.get("Location"):
                    raise ValueError("Redirecionamento sem destino.")
                url = urljoin(final, headers["Location"])
                continue
            if status != 200:
                raise ValueError(f"Site retornou HTTP {status}. Nenhuma proteção será contornada.")
            content_type = headers.get("Content-Type", "").lower()
            if not any(t in content_type for t in ("text/html", "application/xhtml+xml", "text/plain")):
                raise ValueError("A URL não retornou uma página textual suportada.")
            if "text/plain" in content_type:
                return final, final, body.decode("utf-8", errors="replace"), []
            self.progress(0, f"Extraindo texto da página: {final}")
            title, text, links = html_text(body)
            self.reference_links = html_references(body)
            if not text:
                raise ValueError("Página sem texto extraível (JavaScript não é executado).")
            if len(text) > self.config.get("max_text_chars"):
                raise ValueError("Texto do site excede o limite.")
            return final, title or final, text, links
        raise ValueError("Muitos redirecionamentos.")

    def crawl(self, root, documentation=False, progress=lambda *_: None):
        root, _, _, _ = public_url(root)
        queue = deque([(root, 0)])
        visited = set()
        max_pages = self.config.get("web_pages") if documentation else 1
        attempted = 0
        while queue and attempted < max_pages:
            self.gate.check()
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            attempted += 1
            progress(attempted, f"Página {attempted}/{max_pages}: buscando {url}")
            try:
                final, title, text, links = self.fetch(url, root if documentation else None)
                visited.add(final)
                yield {"origin": final, "title": title, "text": text, "depth": depth}
                if documentation and depth < self.config.get("web_depth"):
                    for href in links[:2000]:
                        candidate = urljoin(final, href).split("#", 1)[0]
                        if len(queue) < max_pages * 20 and scope_contains(root, candidate) and candidate not in visited:
                            queue.append((candidate, depth + 1))
            except (ValueError, OSError, urllib3.exceptions.HTTPError) as exc:
                self.gate.check()
                progress(attempted, f"Erro em {url}: {exc}")
                if attempted == 1:
                    raise
