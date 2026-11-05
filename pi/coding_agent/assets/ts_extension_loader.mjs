// Hosts one TS/JS pi extension and bridges setup(api) over stdio JSON lines.
// Parent -> child: {id, method: start|callTool|callCommand|dispatchEvent, params} and {method: shutdown}.
// Child -> parent: replies {id, result|error} and requests {id, method: registerTool|registerCommand|
// registerProvider|registerKeybinding|on|ui|appendEntry|sendMessage|log, params}.

import { pathToFileURL } from "node:url";
import readline from "node:readline";
import { execFile } from "node:child_process";

const extensionPath = process.argv[2];
let nextId = 1;
const pending = new Map();
const handlers = new Map();
let context = {};
const busListeners = new Map();
const flags = new Map();
const mcpServers = new Map();

const plainTheme = new Proxy({
	fg: (_color, text) => String(text ?? ""),
	bg: (_color, text) => String(text ?? ""),
	bold: (text) => String(text ?? ""),
	italic: (text) => String(text ?? ""),
	underline: (text) => String(text ?? ""),
	strikethrough: (text) => String(text ?? ""),
	getMarkdownTheme: () => ({}),
}, { get(target, property) {
	return property in target ? target[property] : (...args) => String(args.at(-1) ?? "");
} });

async function renderLines(component, width) {
	if (component === undefined || component === null) return undefined;
	let rendered = component;
	if (typeof component.render === "function") rendered = await component.render(width);
	if (typeof rendered === "string") return rendered.split("\n");
	if (Array.isArray(rendered)) return rendered.flatMap((line) => String(line).split("\n"));
	return [String(rendered)];
}

function subscribe(channel, handler, once = false) {
	if (typeof channel !== "string" || typeof handler !== "function") {
		throw new TypeError("events.on/once requires a channel and handler");
	}
	const listeners = busListeners.get(channel) ?? new Set();
	busListeners.set(channel, listeners);
	const listener = { handler, once };
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
		if (!listeners.size) busListeners.delete(channel);
	};
}

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
			rendererId: definition.renderCall || definition.renderResult ? `${handlerId}:renderer` : undefined,
		});
		if (definition.renderCall || definition.renderResult) handlers.set(`${handlerId}:renderer`, definition);
	},
	registerCommand(name, options = {}) {
		const handler = options.handler ?? null;
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
	unregisterProvider: name => request("unregisterProvider", { name }),
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
	sendMessage(message, options = {}) {
		if (message && typeof message === "object" && !Array.isArray(message)) {
			return request("sendMessage", { customType: message.customType, content: message.content,
				display: message.display ?? true, details: message.details, options });
		}
		return request("sendMessage", { customType: message, content: arguments[1], display: arguments[2] ?? true });
	},
	log: (message) => request("log", { message: typeof message === "string" ? message : String(message) }),
	ui,
	// Present but unsupported under the Python host, so extensions fail with a
	// clear message instead of an opaque TypeError.
	registerToolRenderer(resolver) {
		if (typeof resolver !== "function") throw Error("registerToolRenderer requires a resolver");
		const handlerId = "tr" + nextId++; handlers.set(handlerId, resolver);
		request("registerToolRenderer", { handlerId });
	},
	registerMcpServer(name, config) {
		if (typeof name !== "string" || !config || typeof config !== "object") throw Error("Invalid MCP server");
		mcpServers.set(name, structuredClone(config));
		request("registerMcpServer", { name, config });
	},
	registerVirtualModel(definition) {
		if (!definition || typeof definition.route !== "function") throw Error("Virtual model requires route");
		const handlerId = "vm" + nextId++; handlers.set(handlerId, definition.route);
		const { route: _route, ...model } = definition;
		request("registerVirtualModel", { definition: model, handlerId });
	},
	unregisterVirtualModel: (provider, id) => request("unregisterVirtualModel", { provider, id }),
	registerMessageRenderer(customType, renderer) {
		if (typeof renderer !== "function") throw Error("registerMessageRenderer requires a renderer");
		const handlerId = "mr" + nextId++; handlers.set(handlerId, renderer);
		request("registerMessageRenderer", { customType, handlerId });
	},
	registerEntryRenderer(customType, renderer) {
		if (typeof renderer !== "function") throw Error("registerEntryRenderer requires a renderer");
		const handlerId = "er" + nextId++; handlers.set(handlerId, renderer);
		request("registerEntryRenderer", { customType, handlerId });
	},
	registerMarkdownTransformer(transformer) {
		if (typeof transformer !== "function") throw Error("registerMarkdownTransformer requires a transformer");
		const handlerId = "md" + nextId++; handlers.set(handlerId, transformer);
		request("registerMarkdownTransformer", { handlerId });
	},
	registerShortcut(shortcut, options) {
		const keys = shortcut.replaceAll("ctrl+", "c-").replaceAll("alt+", "escape ");
		pi.registerKeybinding(`shortcut-${nextId++}`, [keys], (_args, ctx) => options.handler(ctx));
	},
	registerFlag(name, options) {
		if (flags.has(name) || !["boolean", "string"].includes(options.type)) throw Error("Invalid or duplicate flag");
		if (options.default !== undefined && typeof options.default !== options.type) throw Error("Invalid flag default");
		flags.set(name, options.default);
		request("registerFlag", { name, options });
	},
	getFlag: name => flags.get(name),
	setSessionName(name) {
		context.sessionName = name;
		return request("setSessionName", { name });
	},
	getSessionName: () => context.sessionName ?? undefined,
	setLabel: (entryId, label) => request("setLabel", { entryId, label }),
	exec(command, args = [], options = {}) {
		return new Promise((resolve, reject) => {
			execFile(command, args, { cwd: options.cwd ?? context.cwd, timeout: options.timeout,
				encoding: "utf8", windowsHide: true, maxBuffer: 16 * 1024 * 1024 }, (error, stdout, stderr) => {
				if (error && typeof error.code !== "number" && !error.killed) { reject(error); return; }
				resolve({ stdout, stderr, code: typeof error?.code === "number" ? error.code : (error ? -1 : 0), killed: Boolean(error?.killed) });
			});
		});
	},
	getActiveTools: () => [...(context.activeTools ?? [])],
	getAllTools: () => structuredClone(context.allTools ?? []),
	setActiveTools: names => request("setActiveTools", { names }).then(value => { context = value; }),
	refreshTools: () => { void request("refreshTools", {}).then(value => { context = value; }); },
	getSettings: () => structuredClone(context.settings ?? {}),
	getCommands: () => structuredClone(context.commands ?? []),
	setModel: model => request("setModel", { model }),
	getThinkingLevel: () => context.thinkingLevel ?? "off",
	setThinkingLevel(level) {
		return request("setThinkingLevel", { level }).then(() => { context.thinkingLevel = level; });
	},
	unregisterMcpServer(name) {
		mcpServers.delete(name);
		void request("unregisterMcpServer", { name });
	},
	getMcpServers: () => {
		const visible = new Map((context.mcpServers ?? []).map((server) => [server.name, server]));
		for (const [name, config] of mcpServers) visible.set(name, { name, config, extensionPath });
		return [...visible.values()].map((server) => structuredClone(server));
	},
	events: {
		on: (channel, handler) => subscribe(channel, handler),
		once: (channel, handler) => subscribe(channel, handler, true),
		off(channel, handler) {
			const listeners = busListeners.get(channel);
			if (!listeners) return;
			for (const listener of listeners) {
				if (listener.handler === handler) listeners.delete(listener);
			}
			if (!listeners.size) busListeners.delete(channel);
		},
		emit(channel, data) {
			// Match TS fire-and-forget semantics; handler failures stay diagnostic.
			void request("events.emit", { channel, data }).catch((error) => {
				process.stderr.write(`Event emission failed (${channel}): ${error}\n`);
			});
		},
	},
};

function toScriptValue(value) {
	if (value === undefined || value === null) return undefined;
	if (typeof value === "string") return value;
	if (typeof value === "function") return String(value);
	return value;
}

async function callHandler(msg, methodName) {
	context = msg.params?.context ?? context;
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

async function callRenderer(msg) {
	const { handlerId, kind, width = 80 } = msg.params ?? {};
	try {
		let handler = handlers.get(handlerId);
		if (!handler) throw Error(`render: unknown handler ${handlerId}`);
		let component;
		if (kind === "markdown") {
			component = await handler(msg.params.markdown ?? "", msg.params.context ?? {});
			reply(msg.id, { markdown: String(component ?? "") }); return;
		}
		if (kind === "tool") {
			let renderers = handler;
			if (typeof handler === "function") renderers = await handler(msg.params.toolName, () => undefined);
			if (!renderers) { reply(msg.id, {}); return; }
			const renderContext = { ...(msg.params.context ?? {}), invalidate() {}, lastComponent: undefined, state: {} };
			component = msg.params.phase === "call"
				? await renderers.renderCall?.(msg.params.payload ?? {}, plainTheme, renderContext)
				: await renderers.renderResult?.(msg.params.payload ?? {}, msg.params.options ?? {}, plainTheme, renderContext);
		} else if (kind === "message") {
			component = await handler(msg.params.payload ?? {}, msg.params.options ?? {}, plainTheme);
		} else if (kind === "entry") {
			component = await handler(msg.params.payload ?? {}, msg.params.options ?? {}, plainTheme);
		}
		reply(msg.id, { lines: await renderLines(component, width) });
	} catch (error) { replyError(msg.id, error); }
}

async function routeVirtualModel(msg) {
	const handler = handlers.get(msg.params?.handlerId);
	if (!handler) { replyError(msg.id, "unknown virtual model router"); return; }
	try { reply(msg.id, await handler(msg.params.request ?? {}, msg.params.context ?? {})); }
	catch (error) { replyError(msg.id, error); }
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
	if (message.method === undefined && message.id !== undefined && pending.has(message.id)) {
		const entry = pending.get(message.id);
		pending.delete(message.id);
		if (message.error !== undefined) entry.reject(new Error(String(message.error)));
		else entry.resolve(message.result);
		return;
	}
	if (message.method === "start") {
		context = message.params?.context ?? {};
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
	if (message.method === "busEvent") {
		const { channel, data } = message.params;
		const listeners = busListeners.get(channel);
		for (const listener of [...(listeners ?? [])]) {
			if (listener.once) listeners.delete(listener);
			try {
				Promise.resolve(listener.handler(data)).catch((error) => {
					process.stderr.write(`Event handler error (${channel}): ${error}\n`);
				});
			} catch (error) {
				process.stderr.write(`Event handler error (${channel}): ${error}\n`);
			}
		}
		if (listeners && !listeners.size) busListeners.delete(channel);
		return;
	}
	if (message.method === "setFlags") {
		for (const [name, value] of Object.entries(message.params?.values ?? {})) if (flags.has(name)) flags.set(name, value);
		reply(message.id, {}); return;
	}
	if (message.method === "shutdown") {
		process.exit(0);
		return;
	}
	if (message.method === "callTool" || message.method === "callCommand" || message.method === "dispatchEvent") {
		void callHandler(message, message.method);
	}
	if (message.method === "render") void callRenderer(message);
	if (message.method === "routeVirtualModel") void routeVirtualModel(message);
});

rl.on("close", () => {
	for (const entry of pending.values()) entry.reject(new Error("Bridge closed"));
	process.exit(0);
});
