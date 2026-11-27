import httpx

from hotel_agent.tools.predict_tool import PredictClient

VALID = dict(hotel="City Hotel", lead_time=45, arrival_date_year=2017, arrival_date_month="July",
             arrival_date_day_of_month=15, stays_in_weekend_nights=1, stays_in_week_nights=2,
             adults=2, adr=110.0)


def _client(handler):
    return PredictClient("http://api.test", httpx.Client(transport=httpx.MockTransport(handler)))


def test_success_drops_unknown_and_none_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"cancel_probability": 0.4, "will_cancel": True, "threshold_used": 0.21})

    import json
    out = _client(handler).predict(**VALID, agent=None, nonsense=1)
    assert out["ok"] and out["cancel_probability"] == 0.4
    assert "agent" not in seen and "nonsense" not in seen


def test_missing_required_fields_never_hit_the_network():
    def handler(request):
        raise AssertionError("must not be called")

    out = _client(handler).predict(hotel="City Hotel")
    assert not out["ok"] and "lead_time" in out["error"] and "Never guess" in out["error"]
    assert out["retryable"] is True


def test_validation_error_is_readable():
    detail = [{"loc": ["body", "lead_time"], "msg": "Input should be greater than or equal to 0"}]
    out = _client(lambda r: httpx.Response(422, json={"detail": detail})).predict(**VALID)
    assert not out["ok"] and "lead_time" in out["error"]


def test_unreachable_api():
    def handler(request):
        raise httpx.ConnectError("refused")

    out = _client(handler).predict(**VALID)
    assert not out["ok"] and "unreachable" in out["error"]


def test_default_client_ignores_system_proxy_settings(monkeypatch):
    """Regression: with HTTP_PROXY set (VPN/proxy), calls to a local API must not go through it."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"cancel_probability": 0.5, "will_cancel": True, "threshold_used": 0.2}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:1")      # a proxy that cannot serve anything
    try:
        out = PredictClient(f"http://127.0.0.1:{server.server_port}").predict(**VALID)
    finally:
        server.shutdown()
    assert out["ok"], out


def test_misnamed_fields_are_reported_so_the_model_can_retry():
    """A model that sends `weekend_nights` instead of `stays_in_weekend_nights` must be told so."""
    def handler(request):
        raise AssertionError("must not be called")

    fields = {k: v for k, v in VALID.items() if k != "stays_in_weekend_nights"}
    out = _client(handler).predict(**fields, weekend_nights=2)
    assert not out["ok"]
    assert "stays_in_weekend_nights" in out["error"] and "weekend_nights" in out["error"]
    assert "not valid" in out["error"] and "call the tool again" in out["error"]
