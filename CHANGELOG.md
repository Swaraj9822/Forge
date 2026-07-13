# Changelog

All notable changes to Forge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Gemini 3 thinking level.** New `provider.thinking_level` config key
  (`minimal` / `low` / `medium` / `high`) controls how much internal reasoning
  Gemini 3 models perform. Wired into the Vertex request as `thinking_level`;
  when absent, the model's own default is used. Older SDKs that don't accept the
  field degrade gracefully. See the manual §4 and §3.4.
- **Interactive UI: startup banner** showing the model, provider, thinking
  level, autonomy mode, exposed tool count, and workspace.
- **Interactive UI: "thinking" spinner** while awaiting the model, plus the
  turn's elapsed time on the end-of-response marker.
- **Tool visibility.** Tool calls now show their target (e.g.
  `[tool: read] src/app.py`, `[tool: shell] $ pytest -q`) and each result shows
  a concise outcome (`-> 42 lines`, `-> wrote 1200 bytes`, `-> error: …`).
  Failed tool calls are now surfaced instead of being silent.
- **New informational slash commands:** `/cost` (cumulative session usage and
  cost), `/tools` (tools currently exposed to the model), `/model` (active
  model, provider, thinking level), and `/clear` (clear the screen).

### Changed

- **Accurate token accounting for thinking models.** Reasoning ("thoughts")
  tokens reported by Gemini are now folded into the output token tally (they are
  billed at the output rate), so usage counts and estimated cost are accurate
  rather than under-reported.
- **Humanized usage line.** Token counts in the `[usage]` line are now formatted
  for readability (e.g. `8.4k`).

### Notes

- The above UI enhancements activate under `ui.color` / `ui.spinner` on a real
  terminal; they degrade gracefully to plain ASCII when redirected or when the
  `rich` library is unavailable.
