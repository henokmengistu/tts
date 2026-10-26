"""The studio page depends on these routes; creating the app must not load the model."""
from amharic_tts.server import create_app


def test_routes_present():
    app = create_app()
    paths = {r.path for r in app.routes}
    for p in ["/", "/health", "/v1/voices", "/v1/voices/{name}/reference", "/v1/models", "/v1/normalize",
              "/v1/audio/speech", "/v1/external", "/v1/external/google", "/v1/external/gemini"]:
        assert p in paths, p
