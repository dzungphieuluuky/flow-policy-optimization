/**
 * memory - agent-curated facts that survive across sessions.
 *
 * Scope is deliberately narrow: only what no repo can derive (how the user
 * works, environment quirks, standing preferences). Repo rules belong in that
 * repo's AGENTS.md, which already loads and is staleness-checked by
 * scripts/check_claims.py. A second per-project memory file would drift from
 * AGENTS.md with nothing to disagree with it -- the failure already paid for
 * once, when 12 code comments cited a deleted JAX_MIGRATION.md.
 *
 * Every entry carries a mandatory `why`. That is the part Hermes does not have,
 * and the part that makes rot detectable: a stored fact with no recorded reason
 * cannot be checked against anything later.
 *
 * The prompt snapshot is frozen at session_start rather than re-read per turn.
 * That is deliberate -- it keeps the system-prompt prefix cacheable, so injected
 * text that changes every turn would cost a full cache miss on every request.
 *
 * No auto-compaction. A write past MAX_CHARS fails loudly and the agent must
 * consolidate itself; silently dropping entries is how a memory store becomes
 * noise nobody reads.
 */

import { appendFile, mkdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";

import { Type } from "@earendil-works/pi-ai";
import {
	defineTool,
	withFileMutationQueue,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

const MEMORY_PATH = join(process.env.PI_CODING_AGENT_DIR ?? join(process.env.HOME ?? "", ".pi", "agent"), "MEMORY.md");

/** Refuse writes past this rather than dropping the oldest entry. */
const MAX_CHARS = 4000;

/** Turns between self-nudges, and only when nothing has been written. */
const NUDGE_EVERY_TURNS = 10;

const HEADER = `# MEMORY.md — durable facts about how this user works

Only what no repo can derive. Repo rules belong in that repo's AGENTS.md.

- fact: <one line>
  why: <why it is believed; makes a stale entry checkable>
  at: <ISO 8601>
`;

export default function (pi: ExtensionAPI) {
	// Frozen at session start so the injected section is byte-identical every
	// turn and the prefix keeps hitting cache.
	let snapshot = "";
	let turns = 0;
	let wroteThisSession = false;

	pi.on("session_start", async () => {
		snapshot = (await readMemory()).trim();
		turns = 0;
		wroteThisSession = false;
	});

	// Mutate the prompt sections rather than returning systemPrompt: returning
	// it replaces the whole prompt for the run, losing every other section.
	pi.on("before_agent_start", async (event) => {
		if (!snapshot) return;
		event.systemPromptOptions.sections.memory =
			"Facts the user established earlier. Treat as true unless the " +
			"conversation or a repo contradicts them.\n" +
			snapshot;
	});

	pi.on("turn_end", () => {
		turns += 1;
	});

	pi.on("agent_before_settle", async () => {
		// Guarded: an unconditional continue loops forever.
		if (wroteThisSession || turns < NUDGE_EVERY_TURNS) return;
		turns = 0;
		return {
			continue: true,
			entries: [
				{
					type: "custom_message" as const,
					customType: "memory-nudge",
					content:
						"Before finishing: is there a durable fact from this session " +
						"that no repo already records? If yes, call `remember` with the " +
						"fact and why you believe it. If not, say so and stop.",
					display: false,
				},
			],
		};
	});

	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "remember" || !ctx.hasUI) return;
		const { fact, why } = event.input as { fact?: string; why?: string };
		const ok = await ctx.ui.confirm(
			"Allow remember()?",
			`fact: ${fact ?? ""}\nwhy: ${why ?? ""}\n\nSaved to ${MEMORY_PATH}`,
		);
		if (!ok) return { block: true, reason: "remember was denied by the user" };
	});

	pi.registerTool(
		defineTool({
			name: "remember",
			label: "Remember",
			description:
				"Persist one durable fact that no repo can derive: a user preference, " +
				"an environment quirk, a standing correction. Do NOT use it for repo " +
				"rules, file contents, or anything already recorded in an AGENTS.md, " +
				"README, or PLAN.md — those are already loaded and already staleness-checked. " +
				"`why` is required: it is what makes a stale entry checkable later.",
			parameters: Type.Object({
				fact: Type.String({ description: "One line. The fact itself." }),
				why: Type.String({
					description:
						"Why this is believed, and what would falsify it. E.g. 'user corrected " +
						"this twice in review' or 'measured on the P16, differs 3x on the workstation'.",
				}),
			}),

			async execute(_toolCallId, params, _signal, _onUpdate, _ctx): Promise<{
				content: { type: "text"; text: string }[];
				details: { saved: boolean; fact: string };
			}> {
				const existing = await readMemory();
				const entry = `\n- fact: ${params.fact}\n  why: ${params.why}\n  at: ${new Date().toISOString()}\n`;

				if (existing.length + entry.length > MAX_CHARS) {
					return {
						content: [
							{
								type: "text",
								text:
									`MEMORY.md is at ${existing.length}/${MAX_CHARS} chars and this entry would ` +
									`overflow. Consolidate first: drop or merge entries in ${MEMORY_PATH}, then retry.`,
							},
						],
						details: { saved: false, fact: params.fact },
					};
				}

				await withFileMutationQueue(MEMORY_PATH, async () => {
					await appendFile(MEMORY_PATH, entry, "utf-8");
				});

				wroteThisSession = true;
				return {
					content: [{ type: "text", text: `Remembered: ${params.fact}` }],
					details: { saved: true, fact: params.fact },
				};
			},
		}),
	);
}

async function readMemory(): Promise<string> {
	try {
		const text = await readFile(MEMORY_PATH, "utf-8");
		return text.trim() === HEADER.trim() ? "" : text;
	} catch {
		await mkdir(dirname(MEMORY_PATH), { recursive: true });
		await appendFile(MEMORY_PATH, HEADER, "utf-8");
		return "";
	}
}
