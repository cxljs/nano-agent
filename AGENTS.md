# AGENTS.md

## Style

- Always run `gofmt` and `make test` before committing.
- Comments should explain why we are doing something, not just what we are doing. what commands are almost never usful.

## Testing

- `make test` — fast suite (tools + agent against a mock LLM). No tokens spent. Run this before every commit.
- `make test-live` — end-to-end against a real Anthropic API. Run on demand only.

## Commit Style

- Body should explain the "why" not just the "what".
