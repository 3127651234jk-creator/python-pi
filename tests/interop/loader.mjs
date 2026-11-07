/**
 * Resolve hook mapping @earendil-works package specifiers to workspace TS
 * sources, so the interop harness runs against the packages source tree
 * without a build (mirrors what the vitest path mapping does).
 */
import path from "node:path";
import { pathToFileURL } from "node:url";

const PACKAGES = path.resolve(import.meta.dirname, "../../../packages");

const MAP = new Map([
	["@earendil-works/pi-protocol", "protocol/src/index.ts"],
	["@earendil-works/chord", "chord/src/index.ts"],
	["@earendil-works/chord/context", "chord/src/context/index.ts"],
	["@earendil-works/pi-server", "server/src/index.ts"],
	["@earendil-works/pi-server/testing", "server/src/testing/index.ts"],
	["@earendil-works/pi-client", "client/src/index.ts"],
]);

export function resolve(specifier, context, nextResolve) {
	const mapped = MAP.get(specifier);
	if (mapped !== undefined) {
		return { url: pathToFileURL(path.join(PACKAGES, mapped)).href, shortCircuit: true };
	}
	return nextResolve(specifier, context);
}
