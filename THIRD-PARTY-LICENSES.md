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

## 1. paramiko — LGPL-2.1 (the only copyleft in this stack)

`src/ssh_manager.py` talks to peer machines over SSH via **paramiko**
(LGPL-2.1, https://github.com/paramiko/paramiko/blob/main/LICENSE).
LGPL expressly permits linking into non-free / commercial programs, and a
plain `import paramiko` installed from PyPI (`requirements.txt`) counts as
dynamic linking — so selling this wrapper is allowed.

To stay compliant you MUST:

1. **Give notice** that paramiko is used and that it is covered by the GNU
   Lesser General Public License v2.1. Keep this file plus the reference
   below, and ship a copy of the LGPL-2.1 text with any distribution
   (see `licenses/LGPL-2.1.txt`, or fetch it from
   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt).
2. **Keep paramiko replaceable.** Ship it as a normal pip dependency
   (`pip install -r requirements.txt`) so a customer can upgrade/replace
   the library. Do NOT freeze it invisibly inside a single binary with no
   way to swap it, and do not forbid reverse engineering for debugging
   such a replacement (LGPL §6 requires this permission).
3. **Do not copy paramiko code (including `demos/`) into this repo.**
   Use it only via `import`. If you ever patch paramiko itself, publish
   those patches under LGPL-2.1.
4. Recipients automatically receive their paramiko rights from the
   original licensors — do not add further restrictions on the library.

Full license text: `licenses/LGPL-2.1.txt`
Upstream license page: https://github.com/paramiko/paramiko/blob/main/LICENSE

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

`config/machines.yaml` stores SSH passwords in plaintext and
`config/settings.yaml` may hold a Hugging Face token — by design for local
use (mirrors existing rpc scripts). Before distributing or selling:

- remove / rotate real passwords and tokens,
- ship configs with placeholders,
- tell buyers to set their own credentials.

---
*This file is informational, not legal advice. Have counsel review before launch.*
