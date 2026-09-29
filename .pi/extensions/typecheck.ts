/**
 * typecheck - run ruff over the project and return structured findings.
 *
 * One-shot `ruff check`, not an LSP client: a language server needs document
 * sync and a long-lived process, and the gap this closes is narrow (unused
 * names, bad base-class dispatch) which a batch check already sees.
 *
 * No cache by design. Nothing to tear down, so reload, session swap, and
 * process exit cannot leave stale findings behind.
 */

import { execFile } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

import { Type } from "@earendil-works/pi-ai";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	defineTool,
	formatSize,
	truncateHead,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

const execFileAsync = promisify(execFile);

const DEFAULT_TARGETS = ["src", "scripts", "tests"];

interface TypecheckFinding {
	file: string;
	line: number;
	column: number;
	code: string;
	message: string;
}

interface TypecheckDetails {
	available: boolean;
	findings: TypecheckFinding[];
	raw?: string;
	truncated?: boolean;
	fullOutputPath?: string;
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "typecheck",
			label: "Type check",
			description: `Run ruff over the project and return findings grouped by file. ` +
				`Defaults to ${DEFAULT_TARGETS.join(", ")}; pass paths to narrow it. ` +
				`Output is truncated to ${DEFAULT_MAX_LINES} lines or ` +
				`${formatSize(DEFAULT_MAX_BYTES)}, whichever comes first.`,
			parameters: Type.Object({
				paths: Type.Optional(
					Type.Array(Type.String(), {
						description: `Files or directories to check (default: ${DEFAULT_TARGETS.join(", ")})`,
					}),
				),
			}),

			async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<{
				content: { type: "text"; text: string }[];
				details: TypecheckDetails;
			}> {
				// --output-format concise is what parseConcise understands. Without
				// it ruff prints a pretty report and the parse silently yields zero.
				const args = [
					"check",
					"--output-format",
					"concise",
					...(params.paths ?? DEFAULT_TARGETS),
				];

				let stdout: string;
				let stderr: string;
				try {
					// Ruff exits 1 when it finds violations, so a non-zero status
					// with usable stdout is a result, not a failure.
					({ stdout, stderr } = await execFileAsync("ruff", args, {
						cwd: ctx.cwd,
						encoding: "utf-8",
						maxBuffer: 32 * 1024 * 1024,
					}));
				} catch (err) {
					const e = err as { code?: string | number; stderr?: string };
					if (e.code === "ENOENT") {
						return {
							content: [
								{
									type: "text",
									text: "ruff is not installed. Install it with `uv tool install ruff` (or `uv add --dev ruff`).",
								},
							],
							details: { available: false, findings: [] },
						};
					}
					// Exit 1 means "found violations": ruff writes them to stdout
					// and leaves stderr empty, so guard on stdout, not stderr.
					if (e.code === 1 && e.stdout) {
						stdout = e.stdout;
						stderr = "";
					} else {
						throw new Error(`ruff failed: ${e.stderr || String(err)}`);
					}
				}

				const findings = parseConcise(stdout);
				const text = formatFindings(findings, stderr, stdout);
				const truncation = truncateHead(text, {
					maxLines: DEFAULT_MAX_LINES,
					maxBytes: DEFAULT_MAX_BYTES,
				});

				const details: TypecheckDetails = {
					available: true,
					findings,
					raw: stdout,
					truncated: truncation.truncated,
				};

				let body = truncation.content;
				if (truncation.truncated) {
					// Per docs/extensions.md, tell the model where the full output
					// is instead of silently dropping the tail.
					const dir = await mkdtemp(join(tmpdir(), "typecheck-"));
					details.fullOutputPath = join(dir, "ruff.txt");
					await writeFile(details.fullOutputPath, stdout, "utf-8");
					body += `\n\nTruncated. Full output: ${details.fullOutputPath}`;
				}

				return { content: [{ type: "text", text: body }], details };
			},
		}),
	);
}

/** Parse `path:line:col: CODE message`, the format `ruff --output-format concise`. */
function parseConcise(output: string): TypecheckFinding[] {
	const findings: TypecheckFinding[] = [];
	for (const line of output.split("\n")) {
		const m = /^(.+?):(\d+):(\d+):\s+(\w+)\s+(.*)$/.exec(line);
		if (!m) continue;
		findings.push({
			file: m[1],
			line: Number(m[2]),
			column: Number(m[3]),
			code: m[4],
			message: m[5],
		});
	}
	return findings;
}

function formatFindings(findings: TypecheckFinding[], stderr: string, raw: string): string {
	if (stderr.trim()) return stderr.trim();
	if (findings.length === 0) {
		// Never claim a clean result while ruff printed something we failed to
		// read. Show the raw text rather than report a false negative.
		return raw.trim() || "ruff: no findings.";
	}
	const byFile = new Map<string, TypecheckFinding[]>();
	for (const f of findings) {
		const list = byFile.get(f.file);
		if (list) list.push(f);
		else byFile.set(f.file, [f]);
	}
	const lines: string[] = [`ruff: ${findings.length} finding(s) in ${byFile.size} file(s)`];
	for (const [file, list] of byFile) {
		lines.push(`\n${file}`);
		for (const f of list) {
			lines.push(`  ${f.line}:${f.column}  ${f.code}  ${f.message}`);
		}
	}
	return lines.join("\n");
}
