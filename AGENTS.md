```markdown
# AGENTS.md

## Core Principle

Write the least code that makes the idea correct, readable, reproducible, and easy to experiment with. This is research code, not production infrastructure. Priorities: correctness, clarity, reproducibility, ease of modification, minimal code. Never optimize for hypothetical scale, enterprise architecture, or defensive engineering. When two designs work equally well, choose the one with fewer lines, abstractions, files, and moving parts.

## Minimal Implementation

Identify the smallest change that solves the task completely. Prefer existing functions over new abstractions, direct code over wrappers, stdlib over new dependencies, local changes over repo-wide refactors. Never add code for hypothetical future requirements: no plugin systems, factories, registries, service layers, config frameworks, speculative caching, retry logic, error hierarchies, telemetry, or feature flags unless explicitly required.

**Rule:** if code isn't needed for the requested behavior, don't add it.

Reuse before creating: search the repo for existing equivalents, and don't abstract two-line expressions. Prefer fewer files — don't create modules for functions that can live next to their callers.

## Research Code Style

Make it easy to change experiments, inspect intermediates, compare methods, and reproduce results. Prefer:

```python
def train(model, data, epochs):
    ...
```

over:

```python
class TrainingService:
    def __init__(self, trainer_factory, optimizer_factory, config_provider):
    ...
```

Use abstraction only when it materially reduces duplication or improves understanding.

**Math code:** preserve the algorithm's structure so it's recognizable against the paper:

```python
ratio = (new_log_prob - old_log_prob).exp()
objective = torch.min(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
```

**Tensor code:** keep operations explicit; avoid unnecessary copies/conversions and wrappers around trivial ops:

```python
logits = model(obs)
log_probs = logits.log_softmax(-1)
```

When performance matters, measure first — don't optimize on speculation.

## Standard ML/RL Naming

Always name classes, functions, and variables using conventions from standard libraries (HuggingFace, Stable Baselines 3, CleanRL, skrl). Do not invent new names for existing concepts.

- **Classes:** `Agent`, `Policy`, `ActorCriticPolicy`, `ReplayBuffer`, `RolloutBuffer`, `Environment`, `Trainer`
- **Functions:** `collect_rollouts`, `predict`, `learn`, `update`, `get_action`, `evaluate`
- **Variables:** `obs`, `next_obs`, `actions`, `rewards`, `dones`, `log_probs`, `values`, `advantages`, `returns`, `loss`, `optimizer`, `q_values`

## Python Style

- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- Names that convey mathematical meaning: `advantages`, `returns`, `log_probs`, `latent_dim` — not `a`, `r`, `lp` (conventional abbreviations like `lr` are fine)
- Type hints when they improve readability

## Match the Repository

Inspect nearby code before introducing new patterns. Follow existing conventions for naming, structure, config, logging, and imports. Consistency beats a theoretically "better" architecture. Don't refactor unrelated code for style.

## Dependencies and Configuration

Avoid new dependencies unless the stdlib or an existing dependency can't solve it — each one costs maintenance and reproducibility. Keep config simple; plain constants are fine for experiment-local values:

```python
learning_rate = 3e-4
batch_size = 256
```

Don't build config frameworks. Use the repo's existing mechanism if one exists.

## Error Handling

Don't add defensive checks for impossible conditions or broad `except Exception` blocks that hide research bugs. Prefer obvious, debuggable failures. Validate only when an invalid value would produce a confusing result:

```python
if batch_size <= 0:
    raise ValueError("batch_size must be positive")
```

## Comments

Comment **why**, not **what**. Capture mathematical intent, experimental assumptions, numerical tricks, and deviations from the paper:

```python
# Stop-gradient so the target network does not receive actor gradients.
target = target.detach()
```

No large doc blocks on trivial functions.

## Testing

Write the smallest test that catches the relevant failure: shapes, numerics, algorithmic invariants, reproducibility, and regressions for real bugs. No test suites for code with no failure surface.

## Scripts

Simple executable scripts suffice:

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

No CLI frameworks unless the repo uses one or the task requires it.

## Diff Discipline

Minimize the diff. Don't turn "Add X" into "rewrite the architecture." Before finishing: inspect the diff; remove unused imports, dead code, unnecessary comments/abstractions, and unrelated changes.

## Decision Rule

When uncertain, prefer in order:

1. Existing repository pattern
2. Simple direct implementation
3. Standard library solution
4. Small helper
5. Abstraction

Move down only when the simpler option is genuinely insufficient.

## Anti-Patterns

Avoid `SomethingManager`, `SomethingFactory`, `SomethingService`, wrapper chains, multi-layer config loading, broad `except Exception`, and long `isinstance` chains. Do not confuse abstraction with quality.

## Workflow

1. Inspect relevant code and existing patterns.
2. Make the smallest correct change.
3. Run relevant checks/tests.
4. Inspect the diff; remove anything unnecessary.

Don't ask architectural permission for ordinary decisions — use judgment and follow existing conventions. When multiple implementations work, choose the simplest one that's correct and easy for a researcher to modify later.

## Pi Harness System Prompt (Reference)

Defaults of the Pi coding-agent harness (`@earendil-works/pi-coding-agent`,
`packages/coding-agent/src/core/system-prompt.ts`). Pi assembles its system
prompt from ordered sections: preamble, tools, rules, docs, addendum, project
context, skills, cwd.

Preamble (verbatim default): "You are an expert coding assistant operating
inside pi, a coding agent harness. You help users by reading files, executing
commands, editing code, and writing new files."

Default tools (`read`, `bash`, `edit`, `write`), each with a one-line snippet
plus rules, e.g. `read`: "Read file contents" — "Use read to examine files
instead of cat or sed." `bash`: "Execute bash commands (ls, grep, find, etc.)"
Always-on rules: "Be concise in your responses"; "Show file paths clearly when
working with files." Project context files and cwd are appended last.

This repo already matches that shape: short role preamble, tool-specific rules,
conciseness + explicit file paths, project context via `AGENTS.md` itself.
```
