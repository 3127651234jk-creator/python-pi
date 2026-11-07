/**
 * TypeScript routed-client interop peer. Connects the real `@earendil-works/pi-client`
 * over a TCP transport factory to the Python server address given on argv and runs a
 * stdin-driven scenario (catalogue, subscribe with chord delta decoding, requests);
 * emits JSON lines on stdout.
 */
import net from "node:net";
import { Client } from "@earendil-works/pi-client";

const [host, port, serverId] = process.argv.slice(2);

const emit = (message) => process.stdout.write(JSON.stringify(message) + "\n");

const transportFactory = (handlers) =>
	new Promise((resolve, reject) => {
		const socket = net.connect(Number(port), host);
		socket.on("connect", () => {
			resolve({
				async send(chunk) {
					socket.write(Buffer.from(chunk));
				},
				close() {
					socket.destroy();
				},
			});
		});
		socket.on("data", (chunk) => handlers.onData(new Uint8Array(chunk)));
		socket.on("close", () => handlers.onClose());
		socket.on("error", (error) => handlers.onError(error));
	});

const client = await Client.connect({ transportFactory, serverId });
emit({ event: "connected" });

let subscription = null;

async function handleLine(command) {
	switch (command.op) {
		case "catalogue": {
			const entries = await client.serviceCatalogue({ serverId });
			emit({ event: "catalogue", entries });
			return;
		}
		case "subscribe": {
			subscription = await client.subscribeService({ serverId }, command.serviceId, command.mode, (update) => {
				emit({ event: "update", update });
			});
			subscription.start();
			emit({ event: "subscribed", snapshot: subscription.snapshot, id: subscription.id });
			return;
		}
		case "request": {
			try {
				const result = await client.request({ serverId }, command.call);
				emit({ event: "result", result });
			} catch (error) {
				emit({ event: "error", code: error.code, message: String(error.message) });
			}
			return;
		}
		case "stop": {
			if (subscription !== null) await subscription.dispose();
			await client.dispose();
			emit({ event: "stopping" });
			process.exit(0);
			return;
		}
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
