# Making opencode v2 behave like pi

Target: opencode `v2.0.21` (`~/.opencode/bin/opencode`) behaving like this repo's pi setup.

## What is already free

opencode reads `AGENTS.md` from the project root with no config. Every convention in
this repo's `AGENTS.md` carries over unchanged on step 1:

- `./tmp` for all scratch files
- `WANDB_MODE=offline` for smoke runs
- `git checkout -- wandb/` after `pytest`
- torch/JAX builtins-first rules, diff discipline, naming

Do not add `AGENTS.md` to `instructions` in `opencode.json`. It is auto-loaded, and
listing it duplicates the whole file into context.

## Deltas to close

| Concern | pi | opencode v2 | Step |
|---|---|---|---|
| Project instructions | `AGENTS.md` | same file, auto-read | 1 |
| Personal instructions | `~/.pi/agent/AGENTS.md` | `~/.config/opencode/AGENTS.md` | 2 |
| Approval prompts | none | `external_directory` + `*.env` read ask | 3 |
| Watcher noise | n/a | add `wandb/`, `tensorboard/` ignores | 3 |
| Skills | `SKILL.md` spec | same spec, reads `.agents/skills/` | 5 |
| Prompt templates | `~/.pi/agent/prompts/` | `.opencode/commands/` | 7 |
| MCP | `mcp.json`, `exposure` levels | `opencode.json` `mcp`, no exposure | 6 |
| Extensions | `.pi/extensions/*.ts` | `.opencode/plugins/*.ts` | 8 |
| Model | `openrouter/thinkingmachines/inkling:free` | `openrouter/thinkingmachines/inkling` | 4 |

## 1. Project instructions

Nothing to do. Confirm opencode picked it up:

```bash
opencode debug config | grep -A 5 instructions
```

## 2. Personal instructions

pi reads `~/.pi/agent/AGENTS.md`. opencode reads `~/.config/opencode/AGENTS.md`.
Symlink so one edit covers both:

```bash
mkdir -p ~/.config/opencode
ln -sf ~/.pi/agent/AGENTS.md ~/.config/opencode/AGENTS.md
```

## 3. Project config

Create `opencode.json` at the repo root. Project config beats global config, so this
overrides anything in `~/.config/opencode/opencode.json`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "default_agent": "build",
  "permission": {
    "*": "allow",
    "external_directory": "allow",
    "read": { "*.env": "allow", "*.env.*": "allow" },
    "browser": "deny"
  },
  "compaction": { "auto": true, "prune": true, "reserved": 10000 },
  "watcher": {
    "ignore": [
      "wandb/**",
      "tensorboard/**",
      ".venv/**",
      "tmp/**",
      "outputs/**",
      "experiments/**",
      ".ruff_cache/**",
      ".pytest_cache/**",
      "**/__pycache__/**"
    ]
  }
}
```

Why each key:

- `permission` — opencode's `build` agent is `*` allow with two exceptions:
  `external_directory` and `read` on `*.env`. Verified via `opencode debug agents`.
  pi asks for neither, so both are flipped to `allow`. `browser` stays denied;
  pi has no browser tool at all.
- `compaction.prune` — pi prunes old tool output. opencode defaults this to `false`.
- `watcher.ignore` — `wandb/` and `tensorboard/` are tracked in git here and get
  rewritten on every run. Watching them triggers reloads and dirties `git status`.

Skip `lsp` and `formatter`: both are off unless explicitly enabled, and pi has
neither.

Verify:

```bash
opencode debug config
opencode debug agents | python -c "import json,sys; print(json.load(sys.stdin)[0]['permissions'])"
```

## 4. Model and provider

pi's `~/.pi/agent/settings.json` pins `openrouter` +
`thinkingmachines/inkling:free`. opencode model ids are `provider/model-id`, and
`:free` is part of the upstream id, not opencode's syntax.

1. Authenticate:

```bash
opencode auth login
```

Pick OpenRouter, paste the key from `~/.config/opencode/auth.json` or
https://openrouter.ai/keys.

2. Confirm the exact id opencode knows:

```bash
opencode models openrouter | grep -i inkling
```

3. Add to `opencode.json`:

```json
{ "model": "openrouter/<id-from-step-2>" }
```

Use `small_model` for the same provider's cheapest model; opencode uses it for
session titles, matching pi's title generation.

Alternative: launch with `opencode --auto`, which auto-approves anything not
explicitly denied. Useful when a remote or managed config injects `ask` rules
that your project config does not override.

## 5. Skills

Both harnesses implement the Agent Skills spec and both read `.agents/skills/`,
so that one directory serves pi and opencode:

- project: `.agents/skills/<name>/SKILL.md`
- global: `~/.agents/skills/<name>/SKILL.md`

The only skill installed is Context7, currently owned by a pi package. Link it:

```bash
mkdir -p ~/.agents/skills
ln -sfn ~/.pi/agent/npm/node_modules/@upstash/context7-pi/skills/context7-docs \
  ~/.agents/skills/context7-docs
```

opencode only recognises `name`, `description`, `license`, `compatibility`, and
`metadata` in frontmatter. pi's extra fields (`allowed-tools`,
`disable-model-invocation`) are ignored silently, so a skill relying on them
degrades to always-loadable.

Verify:

```bash
ls ~/.agents/skills/*/SKILL.md
```

In-session, ask opencode to load it by name, or check the `skill` tool description.

## 6. MCP servers

pi config lives in `~/.pi/agent/mcp.json` and `.pi/mcp.json`. Neither exists yet,
so start from the recommended set. Convert with `opencode mcp add`:

```bash
opencode mcp add wandb --url https://mcp.wandb.ai/mcp
opencode mcp add sentry --url https://mcp.sentry.dev/mcp
opencode mcp auth sentry
opencode mcp list
```

Format differences when hand-writing entries in `opencode.json`:

| pi | opencode |
|---|---|
| `{ "url": ... }` | `{ "type": "remote", "url": ... }` |
| `{ "command": "npx", "args": [...] }` | `{ "type": "local", "command": ["npx", ...] }` |
| `env` | `environment` |
| `exposure: "deferred"` | no equivalent |
| `${TOKEN}` | `{env:TOKEN}` |

Read the gap before adding more servers: pi defers large MCP servers out of the
system prompt with `exposure: "deferred"` or `codemode`. opencode always injects
every MCP tool into context. The GitHub server alone can exceed the window. Keep
it to two servers, and pin narrow ones per agent instead of globally:

```json
{
  "tools": { "github*": false },
  "agent": { "build": { "tools": { "github*": true } } }
}
```

## 7. Prompt templates

`~/.pi/agent/prompts/` does not exist, so there is nothing to convert today. When
you add one, pi's syntax does not carry over:

| pi template | opencode command |
|---|---|
| `${1:-default}` | no default syntax; inline the fallback |
| `argument-hint: "[focus]"` | no equivalent |
| `$@`, `$ARGUMENTS` | `$@`, `$ARGUMENTS` |
| no shell injection | `` !`command` `` injects output |
| no file inclusion | `@path` includes the file |

Place in `.opencode/commands/<name>.md`. Filename becomes the `/command` name.

## 8. The typecheck extension

`.pi/extensions/typecheck.ts` registers a `typecheck` tool that shells out to
ruff. pi extensions have no opencode equivalent; opencode's plugin API is
different code.

Cheapest path: drop the tool and make it a command instead, so the agent runs ruff
via the `bash` tool it already has.

```bash
mkdir -p .opencode/commands
```

`.opencode/commands/typecheck.md`:

```markdown
---
description: Run ruff over the project and return findings grouped by file
---

Run ruff over the project and return findings grouped by file.
Use `.venv/bin/ruff check <paths>`; default to `src`, `scripts`, `tests`.
```

Invoke with `/typecheck paths`. If you want the tool back as a first-class
callable, rewrite it against opencode's `tool` plugin API in
`.opencode/plugins/typecheck.ts`.

## 9. Verify the whole thing

```bash
opencode debug config      # merged config, project wins
opencode debug agents      # build permissions, expect external_directory allow
opencode debug paths       # data, config, cache, state
opencode mcp list          # every server connected
opencode models openrouter # model id exists
```

Then in a session: `/init` is not needed, `AGENTS.md` is already loaded. Confirm
the agent applies the `./tmp` rule by asking it to run `ls ./tmp` before writing
any scratch file.

## What cannot be replicated

| pi feature | Why | Workaround |
|---|---|---|
| MCP `exposure` | opencode injects all MCP tools into context | Restrict servers, or per-agent `tools` |
| `codemode` | no opencode equivalent | Call MCP tools directly |
| `tool_search` | no deferred tool loading | Fewer servers |
| `source_check` | opencode has `websearch` + `webfetch` only | Cite URLs from `webfetch` output |
| `/tree`, session fork | opencode has child sessions via subagents | `@general` in a child session |
| pi packages (`pi-web-access`, `opencode-pi`) | pi-specific | Port what you use; ignore the rest |
| Typed extension tools | plugin API differs | Commands (step 8) |

## Notes

- `opencode agent create` appears in the agents docs but is **not** a subcommand in
  `v2.0.21`. Verified against `opencode --help`. Write `.opencode/agents/*.md` by
  hand.
- pi's `codemode` and `tool-search` extensions activate only when a matching MCP
  server connects. With no MCP servers, pi runs without either, which is the mode
  step 6 reproduces.
- `wandb/` and `tensorboard/` are tracked in git. opencode writes nothing there on
  its own, but `bash` commands it runs will. The `AGENTS.md` rule still applies.