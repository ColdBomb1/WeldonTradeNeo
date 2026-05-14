# Local AI Trading Integration

This app now routes trading model work through `services.ai_service` instead of
calling Claude directly from signal, ruleset, sentiment, training, and
post-trade analysis paths.

## Decision Structure

The model is an analyst, not the final executor:

1. Deterministic code builds candidates from rules, indicators, candles, news,
   sentiment, and recent trade lessons.
2. The configured AI provider scores or evolves those candidates.
3. Deterministic risk checks block trades that breach pause, blackout, daily
   loss, relative drawdown, aggregate open risk, max-position, and correlation
   limits.
4. Broker execution and post-trade analysis remain audited in the database.

Keep the database field `claude_analysis` for now. It is a compatibility field
that stores provider-neutral model analysis.

## Providers

Configure providers on `/settings`.

- `claude`: legacy Anthropic Claude API path.
- `ollama`: local Ollama chat API, usually `http://127.0.0.1:11434`.
- `openai_compatible`: local servers that expose `/v1/chat/completions`.

For the ForexAI laptop with the RTX 4090 Laptop GPU, the first local model to
try is still a 14B-class instruct model such as Qwen3 14B or a comparable
quantized model. Start with conservative temperature (`0.1` to `0.3`) and JSON
output prompts.

## Recommended Remote Topology

The trading server should not need public access to the local AI machine. Use a
reverse SSH tunnel from `forexai` to `weldontrade` so the app can call a local
loopback URL on the remote server:

```powershell
ssh -N -R 127.0.0.1:11434:127.0.0.1:11434 weldontrade
```

Then configure the app on `weldontrade`:

- Provider: `ollama`
- Base URL: `http://127.0.0.1:11434`
- Model: the pulled local model name

Run that tunnel as a scheduled task on the ForexAI machine after the model
server is installed and tested.

## Live Safety

Live entries should remain paused while changing providers, risk settings, or
rule text. The execution pause lives in `data/config.json` as:

```json
"execution": {
  "paused": true
}
```

Unpause only after the local model endpoint is reachable from `weldontrade`, the
risk dashboard shows drawdown below the hard stop, and paper/backtest behavior
looks sane.
