# AGENTS.md — Minimal Research Code Standards

Write ML/RL research code in the style of **CleanRL**, **RSL-RL**, and **SKRL**: flat, direct, high-performance, and readable top-to-bottom in one sitting. Every extra abstraction adds cognitive load—treat code size and indirection as costs you must justify.

---

## 1. Core Principles

1. **Flat over clever.** Prefer one clear 60-line function over 5 small functions passing a state object back and forth.
2. **No single-use micro-helpers.** Do NOT extract 1–4 line operations into named functions unless reused in multiple places *and* semantically non-trivial (e.g., `F.smooth_l1_loss`, `v.clamp(-b, b)`, `actions / scale`, `torch.randn`, `(a - b)` belong inline).
3. **No speculative abstractions.** Do NOT create base classes, backend registries, factory functions, or adapter wrappers when only one implementation exists (e.g., call `wandb` directly instead of `Backend -> WandbBackend -> make_backends -> log_metrics`).
4. **No dead code or compatibility shims.** Delete unused fields, unused attributes, redundant lists, and dead helpers outright. Do not deprecate or keep fallback paths for deleted features.
5. **Preserve public contracts.** When simplifying internals, keep public APIs, `Config` fields, `nn.Module` attribute/buffer names (`state_dict` compatibility), and logged metric/CSV keys intact unless explicitly asked to change them.

---

## 2. File & Function Layout

* **Single import block at the top.** Group all standard-library and core third-party imports (`numpy`, `torch`, `torch.nn.functional as F`, etc.) at the top of the file. Only lazy-import heavy optional dependencies (`wandb`, `gymnasium`, `mani_skill`, `metaworld`) inside the functions that use them.
* **Top-to-bottom reading order.** Organize files linearly:
  1. Constants & Config
  2. Normalization & Bounds
  3. Neural Network Modules
  4. Objective / Flow / RL Math
  5. Policy
  6. Buffer & Rollout Collector
  7. Optimizer & Update Step
  8. Evaluator & Training Loop
* **Short, sharp docstrings.** One line explaining *what* or *why* (invariants, tensor shapes like `[B, T, D]`). No multi-paragraph boilerplate docstrings on trivial methods.
* **Keep validation concise.** Validate invariants directly at boundaries (1–2 `if ...: raise ValueError(...)` lines), not via separate `_check_*` helper functions.

---

## 3. PyTorch & RL Performance Rules

* **Stage batches on `device` once per update.** Never call `torch.as_tensor(..., device=device)` inside minibatch or epoch loops. Move the rollout batch to `device` once before the epoch loop and slice on-device.
* **Be explicit about CPU vs. GPU boundaries.** If a buffer lives on CPU and the policy/env runs on CUDA, convert explicitly at the boundary (`.to(device)` / `.detach().cpu()`) and avoid hidden round-trips.
* **No redundant tensor round-trips.** Don't scale a tensor to space A and immediately unscale it back to space B if you already have the original tensor.
* **Guard inference & target updates.** Use `@torch.inference_mode()` or `@torch.no_grad()` on all rollout collection, bootstrap estimation, evaluation, and target-network updates.
* **Avoid duplicate logging.** Log each metric once per step in a single `log(...)` call; never merge/overlap two logging calls at the same step.

---

## 4. Hard Limits (Self-Check Before Finishing)

Before finalizing any code change, audit your diff against these checks:

- [ ] **Inline test:** Is any helper function ≤ 5 lines and called only once? **Inline it.**
- [ ] **Indirection test:** Does reading a core loop (collection, loss, update, training) require jumping across ≥ 3 helper definitions? **Flatten it.**
- [ ] **Wrapper test:** Is there an abstract class, wrapper, or dataclass holding state solely for one function? **Remove it.**
- [ ] **Transfer test:** Are any host-to-device transfers happening inside `for epoch` or `for minibatch`? **Hoist them out.**
- [ ] **Dead-code test:** Are there any unused imports, constants, attributes, config fields, or unreferenced functions? **Delete them.**
- [ ] **Parity test:** Do `state_dict` keys, tensor shapes, RNG/math semantics, and logged metric names remain exact? **Verify with tests.**