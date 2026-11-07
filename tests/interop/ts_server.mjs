/**
 * TypeScript routed-server interop peer. Hosts the real `@earendil-works/pi-server`
 * over a TCP listener adapter with a chord provider (`pi.example` replicated state
 * plus methods, `pi.session-management` attachment flow), driven by JSON lines on
 * stdin; emits JSON lines on stdout.
 */
import net from "node:net";
import { randomUUID } from "node:crypto";
import { Server } from "@earendil-works/pi-server";
import {
	createRemoteServiceEndpoint,
	defineService,
	RemoteServiceProvider,
	replicatedState,
} from "@earendil-works/chord";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";

const Example = defineService("pi.example");
const Management = defineService("pi.session-management");

const sharedState = replicatedState({ count: 0, label: "alpha", items: [] });

async function echo(...args) {
	args.pop(); // trailing Context supplied by the provider
	return args;
}

async function wait(...args) {
	const milliseconds = typeof args[0] === "number" ? args[0] : 30_000;
	await new Promise((resolve) => setTimeout(resolve, milliseconds));
	return true;
}

const host = {
	serverServices: {
		async attachClient(presentation) {
			try {
				const provider = new RemoteServiceProvider([
					{ service: Example, mode: "singleton" },
					{ service: Management, mode: "singleton" },
				]);
				provider.provide(Example, { state: sharedState, echo, wait });
				provider.provide(Management, {
					async attach(sessionId) {
						await presentation.attachSession(sessionId);
					},
					async detach() {
						await presentation.detachSession();
					},
				});
				const endpoint = createRemoteServiceEndpoint(provider);
				return {
					invokeService: (call, publish, context) => endpoint.invoke(call, publish, context),
					release: () => endpoint.dispose(),
				};
			} catch (error) {
				emit({ event: "attach-client-error", message: String(error && error.stack || error) });
				throw error;
			}
		},
	},
	async resolveSession(sessionId) {
		return { id: sessionId };
	},
	async openSession(metadata) {
		return {
			async attachClient() {
				return {
					invokeService: async (call) => call.args,
					release: async () => {},
				};
			},
			async close() {},
		};
	},
};

class TcpListener {
	constructor(port) {
		this.port = port;
		this.server = null;
	}

	async start(accept) {
		this.server = net.createServer((socket) => {
			const connection = {
				get closed() {
					return socket.destroyed;
				},
				async send(chunk) {
					socket.write(Buffer.from(chunk));
				},
				async close(finalChunk) {
					if (finalChunk !== undefined) socket.write(Buffer.from(finalChunk));
					socket.end();
				},
			};
			const handler = accept(connection);
			socket.on("data", (chunk) => handler.onData(new Uint8Array(chunk)));
			socket.on("close", () => handler.onClose());
			socket.on("error", (error) => handler.onError(error));
		});
		await new Promise((resolve, reject) => {
			this.server.once("error", reject);
			this.server.listen(this.port, "127.0.0.1", () => {
				this.port = this.server.address().port;
				resolve();
			});
		});
	}

	async close() {
		await new Promise((resolve) => this.server.close(resolve));
	}
}

const emit = (message) => process.stdout.write(JSON.stringify(message) + "\n");

const serverId = process.argv[2] ?? randomUUID();
const listener = new TcpListener(Number(process.argv[3] ?? 0));
const server = new Server(host, {
	listeners: [listener],
	serverId,
	onError: (error) => emit({ event: "server-error", stack: String((error && error.stack) || error) }),
});
await server.start();
emit({ event: "listening", port: listener.port, serverId });

async function handleLine(command) {
	switch (command.op) {
		case "bump":
			await sharedState.change(BACKGROUND_CONTEXT, (draft) => {
				draft.count += 1;
			});
			emit({ event: "changed" });
			return;
		case "set":
			await sharedState.change(BACKGROUND_CONTEXT, (draft) => {
				draft.label = command.value;
			});
			emit({ event: "changed" });
			return;
		case "push":
			await sharedState.change(BACKGROUND_CONTEXT, (draft) => {
				draft.items.push(command.value);
			});
			emit({ event: "changed" });
			return;
		case "stop":
			emit({ event: "stopping" });
			await server.close();
			process.exit(0);
			return;
		default:
			emit({ event: "error", message: `Unknown op ${JSON.stringify(command.op)}` });
	}
}

let buffer = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let index;
	while ((index = buffer.indexOf("\n")) !== -1) {
		const line = buffer.slice(0, index).trim();
		buffer = buffer.slice(index + 1);
		if (line) {
			handleLine(JSON.parse(line)).catch((error) => emit({ event: "error", message: String(error) }));
		}
	}
});
process.stdin.on("end", () => {
	process.exit(0);
});
