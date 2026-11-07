# Third-Party Licenses — Llama RPC Farm

Llama RPC Farm itself is your own original code (no third-party code is
copied into this repo). It **uses** the following third-party projects at
runtime via standard package installs / downloads / CDN links. All of them
permit commercial use, including in a paid closed-source wrapper product,
provided you keep the notices below.

## Runtime components (downloaded, not vendored)

| Component | License | Source | What we do |
|-----------|---------|--------|------------|
| llama.cpp (llama-server, rpc-server) | MIT License — Copyright (c) 2023-2026 The ggml authors | https://github.com/ggml-org/llama.cpp | Referenced/launched via generated scripts and `llamacpp_installer.py`. Never copied into this repo. |
| llama-swap (mostlygeek/llama-swap) | MIT License | https://github.com/mostlygeek/llama-swap | Managed via config + HTTP API. Binary downloaded by the user, never copied into this repo. |
| Chainlit (`chainlit` PyPI package, used in `chat/app.py`) | Apache License 2.0 — Copyright Chainlit contributors | https://github.com/Chainlit/chainlit/blob/main/LICENSE | Used as a normal pip dependency (`requirements.txt`). No code copied. |
| openai (`openai` PyPI package) | Apache License 2.0 — Copyright 2024 OpenAI | https://github.com/openai/openai-python/blob/main/LICENSE | Normal pip dependency. Only speaks to OpenAI-compatible local endpoints. |
| huggingface_hub (`huggingface_hub` PyPI package) | Apache License 2.0 | https://github.com/huggingface/huggingface_hub | Normal pip dependency for downloads/metadata. |
| fastapi, pydantic, pyyaml | MIT | https://github.com/tiangolo/fastapi, https://github.com/pydantic/pydantic, https://github.com/yaml/pyyaml | Normal pip dependencies. |
| uvicorn, jinja2, httpx, websockets, starlette | BSD-3-Clause (jinja2: BSD-3-Clause; others BSD/MIT-family) | respective PyPI/GitHub pages | Normal pip dependencies. |
| uv installer (astral-sh/uv, fetched by `install.sh` if needed) | MIT / Apache-2.0 | https://github.com/astral-sh/uv | Downloaded at install time only. |
| Google Fonts: Inter + JetBrains Mono (loaded via fonts.googleapis.com in `ui/templates/index.html`) | SIL Open Font License 1.1 | https://fonts.google.com | Remote stylesheet link only. Commercial use allowed. |
| Font Awesome Free 6.5.2 (loaded via cdnjs in `ui/templates/index.html`) | Icons: CC BY 4.0 · Fonts: SIL OFL 1.1 · Code: MIT — Copyright FortAwesome | https://fontawesome.com/license/free | Remote CSS link only. Commercial use allowed **with attribution** (see below). |

## No copyleft runtime dependency

Remote shell access (paramiko/sshpass) was removed. Peers are reached only
through the Apache-2.0 peer agent, so this stack has no LGPL runtime
dependency.

## Attribution notices (keep these)

- llama.cpp — MIT: `Copyright (c) 2023-2026 The ggml authors. Permission is
  hereby granted ... to use, copy, modify, merge, publish, distribute,
  sublicense, and/or sell copies of the Software ...` Full copy bundled only
  with llama.cpp distributions; see https://github.com/ggml-org/llama.cpp
- llama-swap — MIT (mostlygeek): commercial use/modification/distribution/sale
  allowed with copyright + permission notice preserved. See
  https://github.com/mostlygeek/llama-swap
- Chainlit / openai / huggingface_hub — Apache-2.0: you may reproduce,
  distribute, and sell works using these libraries; keep their copyright +
  license notices, state significant changes, and retain any NOTICE files.
  Do not use their trademarks to endorse your product.
- Font Awesome Free — attribution is required by CC BY 4.0 / SIL OFL / MIT.
  The CDN files already carry attribution comments — do not strip them — and
  keep a visible credit such as "Icons: Font Awesome Free (CC BY 4.0)".
- This product does not claim endorsement by llama.cpp, llama-swap,
  Chainlit, OpenAI, Hugging Face, FortAwesome, or Google. Use "compatible
  with" / "manager for" wording, not their names as your product name.

## Model-license disclaimer (important)

Model **weights** (Llama, Qwen, Mistral, DeepSeek, etc.) are NOT covered by
the tool licenses above. Each model ships under its own terms (e.g. Meta
Llama Community License, Qwen License, Apache-2.0, MIT). Some restrict
commercial use, redistribution, or require registration.

- Llama RPC Farm grants you no rights to any model.
- Customers must bring their own models and comply with each model's license.
- If you bundle or pre-load weights for sale, check that model's license
  first — that is the most likely thing that could block a commercial deal.

## Secrets hygiene

`config/settings.yaml` holds access keys and may hold a Hugging Face
token; agent tokens live encrypted in `.env`. Before distributing or
selling:

- remove / rotate real credentials and tokens,
- ship configs with placeholders,
- tell buyers to set their own credentials.

---
*This file is informational, not legal advice. Have counsel review before launch.*
