// Hosts one TS/JS pi extension and bridges setup(api) over stdio JSON lines.
// Parent -> child: {id, method: start|callTool|callCommand|dispatchEvent, params} and {method: shutdown}.
// Child -> parent: replies {id, result|error} and requests {id, method: registerTool|registerCommand|
// registerProvider|registerKeybinding|on|ui|appendEntry|sendMessage|log, params}.

import { pathToFileURL } from "node:url";
import readline from "node:readline";

const extensionPath = process.argv[2];
let nextId = 1;
const pending = new Map();
const handlers = new Map();

const out = (message) => {
	process.stdout.write(JSON.stringify(message) + "\n");
};

function request(method, params) {
	const id = nextId++;
	const promise = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
	out({ id, method, params });
	return promise;
}

function reply(id, result) {
	out({ id, result });
}

function replyError(id, error) {
	out({ id, error: error instanceof Error ? error.message : String(error) });
}

const failUnimplemented = (feature) => () => {
	throw new Error(`${feature} is not available when the extension runs under the Python host`);
};

const ui = {
	notify: (message, level) => request("ui", { call: "notify", message, level }),
	select: (title, options) => request("ui", { call: "select", title, options }),
	confirm: (message) => request("ui", { call: "confirm", message }),
	input: (message, placeholder) => request("ui", { call: "input", message, placeholder }),
	setStatus: (text) => request("ui", { call: "setStatus", text }),
	setWidget: (placement, lines) => request("ui", { call: "setWidget", placement, lines }),
	setTitle: (title) => request("ui", { call: "setTitle", title }),
};

const pi = {
	on(event, handler) {
		if (typeof handler !== "function") throw new Error("on() requires a handler function");
		const handlerId = "h" + nextId++;
		handlers.set(handlerId, handler);
		request("on", { event, handlerId });
		return () => handlers.delete(handlerId);
	},
	registerTool(definition) {
		if (!definition || typeof definition.name !== "string" || typeof definition.execute !== "function") {
			throw new Error("registerTool requires a definition with name and execute");
		}
		const handlerId = "t" + nextId++;
		handlers.set(handlerId, definition.execute);
		request("registerTool", {
			name: definition.name,
			description: definition.description ?? "",
			parameters: definition.parameters ?? { type: "object", properties: {} },
			outputSchema: definition.outputSchema,
			handlerId,
		});
	},
	registerCommand(name, options = {}) {
		const handler = options.handler ?? options.handler ?? null;
		if (typeof name !== "string" || typeof handler !== "function") {
			throw new Error("registerCommand requires a name and an options.handler function");
		}
		const handlerId = "c" + nextId++;
		handlers.set(handlerId, handler);
		request("registerCommand", { name, description: options.description ?? "", handlerId });
	},
	registerProvider(...args) {
		const [name, config] = args.length === 1 ? [args[0]?.name, args[0]] : args;
		request("registerProvider", { name, config: config ?? {} });
	},
	registerKeybinding(...args) {
		const [action, keys, handler] = args.length === 1
			? [args[0]?.action, args[0]?.keys, args[0]?.handler]
			: args;
		if (typeof handler !== "function") throw new Error("registerKeybinding requires a handler function");
		const handlerId = "k" + nextId++;
		handlers.set(handlerId, handler);
		request("registerKeybinding", { action, keys, handlerId });
	},
	appendEntry: (customType, data) => request("appendEntry", { customType, data }),
	sendMessage: (customType, content, display = true) => request("sendMessage", { customType, content, display }),
	log: (message) => request("log", { message: typeof message === "string" ? message : String(message) }),
	ui,
	// Present but unsupported under the Python host, so extensions fail with a clear message.
	registerToolRenderer: failUnimplemented("registerToolRenderer"),
	registerMcpServer: failUnimplemented("registerMcpServer"),
	registerVirtualModel: failUnimplemented("registerVirtualModel"),
	registerMessageRenderer: failUnimplemented("registerMessageRenderer"),
	sendUserMessage: failUnimplemented("sendUserMessage"),
	events: undefined,
};

function toScriptValue(value) {
	if (value === undefined || value === null) return undefined;
	if (typeof value === "string") return value;
	if (typeof value === "function") return String(value);
	return value;
}

async function callHandler(msg, methodName) {
	const { handlerId } = msg.params ?? {};
	const handler = handlers.get(handlerId);
	if (!handler) {
		replyError(msg.id, `${methodName}: unknown handler ${handlerId}`);
		return;
	}
	try {
		let result;
		if (methodName === "callTool") {
			result = await handler(msg.params.toolCallId, msg.params.params ?? {}, undefined, (update) => {
				out({ method: "toolUpdate", params: { toolCallId: msg.params.toolCallId, result: serializeToolResult(update) } });
			}, undefined);
			result = serializeToolResult(result);
		} else if (methodName === "callCommand") {
			result = await handler(msg.params.arguments ?? "", msg.params.context ?? {});
			result = { result: toScriptValue(result) };
		} else {
			const returned = await handler(msg.params.event ?? {}, msg.params.context ?? {});
			result = returned === undefined ? null : returned;
		}
		reply(msg.id, result ?? {});
	} catch (error) {
		replyError(msg.id, error);
	}
}

function serializeToolResult(result) {
	if (result === undefined || result === null) return { content: [] };
	if (typeof result === "string") return { content: [{ type: "text", text: result }] };
	const content = (result.content ?? []).map((block) =>
		block?.type === "image"
			? { type: "image", data: block.data, mimeType: block.mimeType ?? block.mime_type ?? "image/png" }
			: { type: "text", text: String(block?.text ?? "") });
	const serialized = { content };
	if (result.details !== undefined) serialized.details = result.details;
	if (result.structuredContent !== undefined) serialized.structuredContent = result.structuredContent;
	if (result.isError !== undefined) serialized.isError = result.isError;
	if (result.terminate !== undefined) serialized.terminate = result.terminate;
	return serialized;
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
	if (!line.trim()) return;
	let message;
	try {
		message = JSON.parse(line);
	} catch {
		return;
	}
	if (message.id !== undefined && pending.has(message.id)) {
		const entry = pending.get(message.id);
		pending.delete(message.id);
		if (message.error !== undefined) entry.reject(new Error(String(message.error)));
		else entry.resolve(message.result);
		return;
	}
	if (message.method === "start") {
		void (async () => {
			try {
				const module_ = await import(pathToFileURL(extensionPath).href);
				const setup = module_.setup ?? module_.default;
				if (typeof setup !== "function") throw new Error("Extension must export setup(pi)");
				await setup(pi);
				reply(message.id, {});
			} catch (error) {
				replyError(message.id, error);
				process.exitCode = 1;
			}
		})();
		return;
	}
	if (message.method === "shutdown") {
		process.exit(0);
		return;
	}
	if (message.method === "callTool" || message.method === "callCommand" || message.method === "dispatchEvent") {
		void callHandler(message, message.method);
	}
});

rl.on("close", () => {
	for (const entry of pending.values()) entry.reject(new Error("Bridge closed"));
	process.exit(0);
});
