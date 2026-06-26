# Deployment Guide

## One-Command Startup

1. Copy environment defaults:

```bash
cp .env.example .env
```

2. Add your provider keys in `.env`:

- `FINNHUB_API_KEY`
- `ALPHA_VANTAGE_API_KEY`
- `SEC_USER_AGENT`

3. Start the platform:

```bash
docker compose up --build
```

## Services

- App API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Ollama: `http://localhost:11434`

## Notes

- `ollama-init` pulls `llama3.1:8b` automatically.
- If Ollama is not ready yet, the platform still boots and falls back safely.
- Redis is used for cache acceleration but the app still degrades gracefully if unavailable.

## Useful Commands

```bash
docker compose logs -f app
docker compose logs -f ollama
docker compose restart app
docker compose down
```
