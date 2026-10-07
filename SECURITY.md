# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory feature instead of opening a public issue.
Include the affected file or component, reproduction steps, and the potential impact.

## Secrets and local artifacts

Do not commit API keys, Hugging Face tokens, GitHub tokens, private datasets, model checkpoints, optimizer state, or
machine-specific cache paths. The repository intentionally publishes only tiny demonstration records; full training data,
generated predictions, review queues, checkpoints, and model weights remain ignored and are distributed separately when
their licenses permit it.
