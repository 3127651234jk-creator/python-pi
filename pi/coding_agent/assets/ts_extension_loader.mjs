// Hosts one TS/JS pi extension and bridges setup(api) over stdio JSON lines.
// Parent -> child: {id, method: start|callTool|callCommand|dispatchEvent|customRender|
// customInput|customMouse|editorRender|editorInput|editorMouse, params},
// {method: customClose|editorClose|shutdown}.
// Child -> parent: replies {id, result|error} and requests {id, method: registerTool|registerCommand|
// registerProvider|registerKeybinding|on|ui|uiCustom|uiEditor|terminalSize|appendEntry|sendMessage|log,
// params}, plus notifications {method: toolUpdate|customDone|customRenderRequest|customHandle|
// editorSubmit|editorAction|editorRenderRequest}.

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

// Interactive components: ctx.ui.custom(factory) keeps the component in this
// process and routes keys/mouse from the Python host.
const CURSOR_MARKER = "\x1b_pi:c\x07";
const ANSI_PATTERN = /\x1b\[[0-9;:?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g;
const customSessions = new Map();

function stripAnsi(text) {
	return String(text).replace(ANSI_PATTERN, "");
}

/** Strip cursor markers and report the focused input position (character index). */
function extractCursor(lines) {
	let cursor;
	const cleaned = lines.map((line, y) => {
		const text = String(line);
		const index = text.indexOf(CURSOR_MARKER);
		if (index === -1 || cursor) return text;
		cursor = { x: stripAnsi(text.slice(0, index)).length, y };
		return text.slice(0, index) + text.slice(index + CURSOR_MARKER.length);
	});
	return { lines: cleaned, cursor };
}

const KEY_SEQUENCES = {
	up: ["\x1b[A", "\x1bOA"],
	down: ["\x1b[B", "\x1bOB"],
	right: ["\x1b[C", "\x1bOC"],
	left: ["\x1b[D", "\x1bOD"],
	home: ["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"],
	end: ["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"],
	pageUp: ["\x1b[5~"],
	pageDown: ["\x1b[6~"],
	insert: ["\x1b[2~"],
	delete: ["\x1b[3~"],
	enter: ["\r", "\n", "\x1bOM"],
	escape: ["\x1b"],
	tab: ["\t"],
	space: [" "],
	backspace: ["\x7f", "\x08"],
};

const CSI_KEY_CODES = { up: "A", down: "B", right: "C", left: "D", home: "H", end: "F" };

function matchesKey(data, key) {
	if (Array.isArray(key)) return key.some((entry) => matchesKey(data, entry));
	const parts = String(key).split("+");
	const base = parts.pop() ?? "";
	const ctrl = parts.includes("ctrl");
	const alt = parts.includes("alt");
	const shift = parts.includes("shift");

	if (base.length === 1) {
		let expected = base;
		if (ctrl) expected = String.fromCharCode(base.toLowerCase().charCodeAt(0) - 96);
		if (shift && !ctrl) expected = base.toUpperCase();
		if (alt) expected = "\x1b" + expected;
		return data === expected;
	}

	const sequences = KEY_SEQUENCES[base] ?? [];
	if (!ctrl && !alt && !shift) return sequences.includes(data);

	const code = CSI_KEY_CODES[base];
	if (!code) return false;
	const modifier = 1 + (shift ? 1 : 0) + (alt ? 2 : 0) + (ctrl ? 4 : 0);
	return data === `\x1b[1;${modifier}${code}`;
}

// Defaults mirror pi's TUI/app keybindings for the actions extension views use.
const DEFAULT_KEYBINDINGS = {
	"tui.select.up": ["up"],
	"tui.select.down": ["down"],
	"tui.select.pageUp": ["pageUp"],
	"tui.select.pageDown": ["pageDown"],
	"tui.select.confirm": ["enter"],
	"tui.select.cancel": ["escape", "ctrl+c"],
	"tui.input.submit": ["enter"],
	"tui.input.newLine": ["shift+enter", "ctrl+j"],
	"tui.input.tab": ["tab"],
	"tui.input.copy": ["ctrl+c"],
	"tui.editor.undo": ["ctrl+z"],
	"app.interrupt": ["escape"],
	"app.clear": ["ctrl+c"],
	"app.message.copy": ["ctrl+x"],
	"app.tools.expand": ["ctrl+o"],
	"app.model.select": ["ctrl+l"],
	"app.thinking.toggle": ["ctrl+t"],
};

const keybindings = {
	matches(data, action) {
		return (DEFAULT_KEYBINDINGS[action] ?? []).some((key) => matchesKey(data, key));
	},
	getKeys: (action) => [...(DEFAULT_KEYBINDINGS[action] ?? [])],
};

function customTui(session) {
	const requestRender = () => out({ method: "customRenderRequest", params: { sessionId: session.sessionId } });
	return {
		requestRender,
		invalidate: requestRender,
		setFocus() {},
		getSize: () => ({ width: session.width ?? 80, height: session.height ?? 24 }),
		terminal: {
			get columns() { return session.width ?? 80; },
			get rows() { return session.height ?? 24; },
		},
	};
}

function overlayHandle(session) {
	const requestRender = () => out({ method: "customRenderRequest", params: { sessionId: session.sessionId } });
	return {
		hide: () => out({ method: "customHandle", params: { sessionId: session.sessionId, action: "hide" } }),
		setHidden(hidden) {
			session.hidden = Boolean(hidden);
			requestRender();
		},
		isHidden: () => Boolean(session.hidden),
		focus() {
			session.hidden = false;
			requestRender();
		},
		unfocus() {},
		isFocused: () => !session.hidden,
		getBounds: () => session.bounds,
	};
}

function jsonSafe(value) {
	try {
		return JSON.parse(JSON.stringify(value ?? null));
	} catch {
		return null;
	}
}

// Widget/header/footer factories: components stay here and are re-rendered on
// tui.requestRender(); Python displays the returned lines.
const widgetViews = new Map();
let headerView;
let footerView;

function notifyUi(call, params = {}) {
	void request("ui", { call, ...params }).catch((error) => {
		process.stderr.write(`ui.${call} failed: ${error}\n`);
	});
}

async function terminalSize() {
	try {
		const size = await request("terminalSize", {});
		if (size && typeof size.width === "number") {
			return { width: size.width, height: typeof size.height === "number" ? size.height : 24 };
		}
	} catch {
		/* fall through to the default size */
	}
	return { width: 80, height: 24 };
}

function slotTui(view, render) {
	return {
		requestRender: () => { void render(); },
		invalidate: () => { void render(); },
		setFocus() {},
		getSize: () => ({ width: view.width ?? 80, height: view.height ?? 24 }),
		terminal: {
			get columns() { return view.width ?? 80; },
			get rows() { return view.height ?? 24; },
		},
	};
}

function makeView(factory) {
	return { factory, component: undefined, width: 80, height: 24 };
}

function disposeView(view) {
	if (!view) return;
	try { view.component?.dispose?.(); } catch { /* dispose is best effort */ }
}

async function renderSlot(view, send) {
	try {
		const size = await terminalSize();
		view.width = size.width;
		view.height = size.height;
		if (!view.component) {
			view.component = await view.factory(slotTui(view, () => renderSlot(view, send)), plainTheme);
		}
		if (!view.component) return;
		const lines = (await renderLines(view.component, view.width)) ?? [];
		send(lines);
	} catch (error) {
		process.stderr.write(`Extension view render failed: ${error}\n`);
	}
}

function setWidgetSlot(key, content, options = {}) {
	const placement = options.placement === "belowEditor" ? "below" : "above";
	const previous = widgetViews.get(key);
	if (previous) {
		disposeView(previous);
		widgetViews.delete(key);
	}
	if (content === undefined || content === null) {
		notifyUi("setWidget", { key, placement, lines: [] });
		return;
	}
	if (Array.isArray(content)) {
		notifyUi("setWidget", { key, placement, lines: content.map((line) => String(line)) });
		return;
	}
	if (typeof content !== "function") {
		throw new TypeError("ui.setWidget requires a string array or a component factory");
	}
	const view = makeView(content);
	widgetViews.set(key, view);
	void renderSlot(view, (lines) => notifyUi("setWidget", { key, placement, lines }));
}

function setSlot(kind, factory) {
	const current = kind === "header" ? headerView : footerView;
	disposeView(current);
	if (kind === "header") headerView = undefined;
	else footerView = undefined;
	if (factory === undefined || factory === null) {
		notifyUi(kind === "header" ? "setHeader" : "setFooter", { lines: [] });
		return;
	}
	if (typeof factory !== "function") {
		throw new TypeError(`ui.set${kind === "header" ? "Header" : "Footer"} requires a component factory`);
	}
	const view = makeView(factory);
	if (kind === "header") headerView = view;
	else footerView = view;
	void renderSlot(view, (lines) =>
		notifyUi(kind === "header" ? "setHeader" : "setFooter", { lines }));
}

// Custom editor: the extension's component owns the text buffer. Python runs
// it as the input view and forwards raw keys; onSubmit resolves the read.
let editorHandlerId;
const editorSessions = new Map();

function editorTui(session) {
	const requestRender = () => out({ method: "editorRenderRequest", params: { sessionId: session.sessionId } });
	return {
		requestRender,
		invalidate: requestRender,
		setFocus() {},
		getSize: () => ({ width: session.width ?? 80, height: session.height ?? 24 }),
		terminal: {
			get columns() { return session.width ?? 80; },
			get rows() { return session.height ?? 24; },
		},
	};
}

async function editorComponent(session) {
	if (session.component) return session.component;
	const factory = handlers.get(editorHandlerId);
	if (!factory) throw Error("ui.setEditorComponent: no factory registered");
	const theme = { borderColor: (text) => String(text ?? ""), selectList: plainTheme };
	session.component = await factory(editorTui(session), theme, keybindings);
	if (!session.component) throw Error("ui.setEditorComponent factory returned no component");
	if ("focused" in session.component) session.component.focused = true;
	if (session.component.onEscape === undefined) {
		session.component.onEscape = () =>
			out({ method: "editorAction", params: { sessionId: session.sessionId, action: "escape" } });
	}
	if (session.component.onCtrlD === undefined) {
		session.component.onCtrlD = () =>
			out({ method: "editorAction", params: { sessionId: session.sessionId, action: "exit" } });
	}
	session.component.onSubmit = (text) =>
		out({ method: "editorSubmit", params: { sessionId: session.sessionId, text: String(text ?? "") } });
	return session.component;
}

function setEditorComponent(factory) {
	if (factory === undefined || factory === null) {
		editorHandlerId = undefined;
		request("uiEditor", { handlerId: null }).catch((error) => {
			process.stderr.write(`ui.setEditorComponent failed: ${error}\n`);
		});
		return;
	}
	if (typeof factory !== "function") {
		throw new TypeError("ui.setEditorComponent requires a component factory");
	}
	const handlerId = "ed" + nextId++;
	editorHandlerId = handlerId;
	handlers.set(handlerId, factory);
	request("uiEditor", { handlerId }).then((created) => {
		if (!created || created.unavailable) {
			if (created && created.unavailable) {
				process.stderr.write("ui.setEditorComponent: no interactive host\n");
			}
			return;
		}
		editorSessions.set(created.sessionId, {
			sessionId: created.sessionId, component: undefined, width: 80, height: 24,
		});
	}).catch((error) => {
		process.stderr.write(`ui.setEditorComponent failed: ${error}\n`);
	});
}

async function handleEditor(msg) {
	const params = msg.params ?? {};
	const session = editorSessions.get(params.sessionId);
	if (!session) { replyError(msg.id, "ui.setEditorComponent: unknown session"); return; }
	try {
		const component = await editorComponent(session);
		if (msg.method === "editorInput") component.handleInput?.(String(params.data ?? ""));
		else if (msg.method === "editorMouse") component.handleMouse?.(params.event ?? {});
		session.width = params.width ?? session.width;
		if (typeof params.height === "number") session.height = params.height;
		const lines = (await renderLines(component, session.width)) ?? [];
		reply(msg.id, extractCursor(lines));
	} catch (error) { replyError(msg.id, error); }
}


async function customComponent(session) {
	if (session.component) return session.component;
	const factory = handlers.get(session.handlerId);
	if (!factory) throw Error(`ui.custom: unknown factory ${session.handlerId}`);
	const done = (result) => {
		if (session.closed) return;
		session.closed = true;
		customSessions.delete(session.sessionId);
		out({ method: "customDone", params: { sessionId: session.sessionId, result: jsonSafe(result) } });
		session.resolve(result);
	};
	session.component = await factory(customTui(session), plainTheme, keybindings, done);
	if (!session.component) throw Error("ui.custom factory returned no component");
	// The view owns the keyboard, so Focusable components should show their cursor.
	if ("focused" in session.component) session.component.focused = true;
	if (typeof session.options.onHandle === "function") {
		try {
			session.options.onHandle(overlayHandle(session));
		} catch (error) {
			process.stderr.write(`ui.custom onHandle failed: ${error}\n`);
		}
	}
	return session.component;
}

async function renderCustom(session, width, height) {
	session.width = width;
	if (height !== undefined) session.height = height;
	const component = await customComponent(session);
	const lines = (await renderLines(component, width)) ?? [];
	let overlay;
	let visible = true;
	if (session.options.overlay) {
		overlay = typeof session.options.overlayOptions === "function"
			? session.options.overlayOptions()
			: (session.options.overlayOptions ?? {});
		if (overlay && typeof overlay.visible === "function") {
			visible = Boolean(overlay.visible(session.width, session.height));
		}
		overlay = { ...(overlay ?? {}) };
		delete overlay.visible;
	}
	return {
		...extractCursor(lines),
		done: session.closed,
		overlay,
		hidden: Boolean(session.hidden) || !visible,
	};
}


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
	setWidget(key, content, options) {
		setWidgetSlot(key, content, options ?? {});
	},
	setHeader(factory) {
		setSlot("header", factory);
	},
	setFooter(factory) {
		setSlot("footer", factory);
	},
	setTitle: (title) => request("ui", { call: "setTitle", title }),
	setEditorComponent(factory) {
		setEditorComponent(factory);
	},
	getEditorComponent() {
		return editorHandlerId ? handlers.get(editorHandlerId) : undefined;
	},
	custom(factory, options = {}) {
		if (typeof factory !== "function") return Promise.reject(new TypeError("ui.custom requires a component factory"));
		const handlerId = "cv" + nextId++;
		const sessionId = "view" + nextId++;
		handlers.set(handlerId, factory);
		const completed = new Promise((resolve, reject) => {
			customSessions.set(sessionId, {
				sessionId, handlerId, options, resolve, reject,
				closed: false, component: undefined, width: 80, height: 24,
			});
		});
		request("uiCustom", { handlerId, sessionId, options }).then((created) => {
			if (!created || created.unavailable) {
				const session = customSessions.get(sessionId);
				customSessions.delete(sessionId);
				session?.resolve(undefined);
			}
		}).catch((error) => {
			const session = customSessions.get(sessionId);
			customSessions.delete(sessionId);
			session?.reject(error);
		});
		return completed;
	},
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

async function handleCustom(msg) {
	const params = msg.params ?? {};
	const session = customSessions.get(params.sessionId);
	if (!session) { replyError(msg.id, "ui.custom: unknown session"); return; }
	if (params.bounds) session.bounds = params.bounds;
	try {
		if (!session.component) await customComponent(session);
		if (msg.method === "customInput") session.component.handleInput?.(String(params.data ?? ""));
		else if (msg.method === "customMouse") session.component.handleMouse?.(params.event ?? {});
		reply(msg.id, await renderCustom(session, params.width ?? session.width, params.height));
	} catch (error) { replyError(msg.id, error); }
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
	if (message.method === "customRender" || message.method === "customInput" || message.method === "customMouse") {
		void handleCustom(message);
	}
	if (message.method === "editorRender" || message.method === "editorInput" || message.method === "editorMouse") {
		void handleEditor(message);
	}
	if (message.method === "editorClose") {
		const session = editorSessions.get(message.params?.sessionId);
		if (session) {
			editorSessions.delete(session.sessionId);
			try { session.component?.dispose?.(); } catch { /* dispose is best effort */ }
		}
	}
	if (message.method === "customClose") {
		const session = customSessions.get(message.params?.sessionId);
		if (session) {
			session.closed = true;
			customSessions.delete(session.sessionId);
			try { session.component?.dispose?.(); } catch { /* component dispose is best effort */ }
			session.resolve(message.params?.result ?? undefined);
		}
	}
});

rl.on("close", () => {
	for (const entry of pending.values()) entry.reject(new Error("Bridge closed"));
	process.exit(0);
});
