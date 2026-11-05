"""Pi v8 routed envelopes and bounded, definite-length CBOR framing."""

import math
import re
import struct

PROTOCOL_VERSION = 8
MAX_FRAME_LENGTH = 16 * 1024 * 1024
MAX_ITEMS = 1_000_000


class ProtocolValidationError(ValueError):
    pass


def _json(value, depth=0):
    if depth > 64:
        raise ProtocolValidationError("JSON nesting limit exceeded")
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float) and math.isfinite(value):
        return
    if type(value) is list and len(value) <= MAX_ITEMS:
        for item in value:
            _json(item, depth + 1)
        return
    if type(value) is dict and len(value) <= MAX_ITEMS and all(type(k) is str for k in value):
        for item in value.values():
            _json(item, depth + 1)
        return
    raise ProtocolValidationError("Value is not strict JSON")


def _object(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ProtocolValidationError("Invalid envelope fields")


def _id(value):
    if type(value) is not str or not value:
        raise ProtocolValidationError("Expected nonempty identifier")


def _server_id(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", value):
        raise ProtocolValidationError("Invalid server ID")


def validate_target(target):
    session = type(target) is dict and "sessionId" in target
    _object(target, ("serverId", "sessionId", "attachmentId") if session else ("serverId",))
    _server_id(target["serverId"])
    if session:
        _id(target["sessionId"])
        _id(target["attachmentId"])


def validate(message, side):
    _json(message)
    if type(message) is not dict:
        raise ProtocolValidationError("Expected envelope")
    kind = message.get("type")
    if kind == "hello":
        _object(message, ("type", "version") if side == "client" else ("type", "version", "serverId"))
        if type(message["version"]) is not int or message["version"] < 0:
            raise ProtocolValidationError("Invalid version")
        if side == "server":
            _server_id(message["serverId"])
            if message["version"] != PROTOCOL_VERSION:
                raise ProtocolValidationError("Unsupported version")
    elif side == "client" and kind in ("request", "cancel"):
        _object(message, ("type", "id", "target", "call") if kind == "request" else ("type", "id", "target"))
        _id(message["id"])
        validate_target(message["target"])
    elif side == "server" and kind == "response":
        if type(message.get("ok")) is not bool:
            raise ProtocolValidationError("Invalid response status")
        _object(message, ("type", "id", "ok") if message["ok"] else ("type", "id", "ok", "error"),
                ("result",) if message["ok"] else ())
        _id(message["id"])
    elif side == "server" and kind == "hello_error":
        _object(message, ("type", "error"))
    elif side == "server" and kind == "attachment":
        _object(message, ("type", "attachment"))
        if message["attachment"] is not None:
            validate_target(message["attachment"])
            if "sessionId" not in message["attachment"]:
                raise ProtocolValidationError("Expected session attachment")
    elif side == "server" and kind == "service_update":
        _object(message, ("type", "subscriptionId", "update"))
        _id(message["subscriptionId"])
    else:
        raise ProtocolValidationError("Unknown envelope type")
    if "error" in message:
        _object(message["error"], ("code", "message"))
        _id(message["error"]["code"])
        if type(message["error"]["message"]) is not str:
            raise ProtocolValidationError("Invalid error message")
    return message


def _head(major, size):
    for limit, marker, fmt in ((24, None, None), (256, 24, ">B"), (65536, 25, ">H"), (2**32, 26, ">I"), (2**64, 27, ">Q")):
        if size < limit:
            return bytes([(major << 5) | size]) if marker is None else bytes([(major << 5) | marker]) + struct.pack(fmt, size)
    raise ProtocolValidationError("Integer outside CBOR range")


def _encode(value):
    if value is None:
        return b"\xf6"
    if type(value) is bool:
        return b"\xf5" if value else b"\xf4"
    if type(value) is int:
        return _head(0, value) if value >= 0 else _head(1, -1 - value)
    if type(value) is float:
        return b"\xfb" + struct.pack(">d", value)
    if type(value) is str:
        data = value.encode("utf-8")
        return _head(3, len(data)) + data
    if type(value) is list:
        return _head(4, len(value)) + b"".join(_encode(v) for v in value)
    return _head(5, len(value)) + b"".join(_encode(k) + _encode(v) for k, v in value.items())


def encode_message(message, side, max_frame_length=MAX_FRAME_LENGTH):
    data = _encode(validate(message, side))
    if not data or len(data) > max_frame_length:
        raise ProtocolValidationError("Frame size limit exceeded")
    return struct.pack(">I", len(data)) + data


def _decode(data):
    offset = 0
    def take(size):
        nonlocal offset
        if offset + size > len(data):
            raise ProtocolValidationError("Truncated CBOR")
        value = data[offset:offset + size]
        offset += size
        return value
    def item(depth=0):
        if depth > 64:
            raise ProtocolValidationError("CBOR nesting limit exceeded")
        initial = take(1)[0]
        major, extra = initial >> 5, initial & 31
        if major == 7:
            if extra in (20, 21, 22):
                return {20: False, 21: True, 22: None}[extra]
            if extra in (25, 26, 27):
                fmt, size = {25: (">e", 2), 26: (">f", 4), 27: (">d", 8)}[extra]
                return struct.unpack(fmt, take(size))[0]
            raise ProtocolValidationError("Unsupported CBOR simple value")
        if extra < 24:
            size = extra
        elif extra in (24, 25, 26, 27):
            size = int.from_bytes(take(2 ** (extra - 24)), "big")
        else:
            raise ProtocolValidationError("Indefinite CBOR is forbidden")
        if major in (0, 1):
            return size if major == 0 else -1 - size
        if major == 3:
            return take(size).decode("utf-8")
        if major not in (4, 5) or size > MAX_ITEMS:
            raise ProtocolValidationError("Unsupported or oversized CBOR item")
        if major == 4:
            return [item(depth + 1) for _ in range(size)]
        result = {}
        for _ in range(size):
            key = item(depth + 1)
            if type(key) is not str or key in result:
                raise ProtocolValidationError("Invalid CBOR map key")
            result[key] = item(depth + 1)
        return result
    result = item()
    if offset != len(data):
        raise ProtocolValidationError("Trailing CBOR bytes")
    return result


class MessageDecoder:
    def __init__(self, side, max_frame_length=MAX_FRAME_LENGTH):
        self.side = side
        self.limit = max_frame_length
        self.buffer = bytearray()
        self.failed = False

    def push(self, chunk):
        if self.failed:
            raise ProtocolValidationError("Decoder has failed")
        try:
            self.buffer.extend(chunk)
            messages = []
            while len(self.buffer) >= 4:
                size = int.from_bytes(self.buffer[:4], "big")
                if not 0 < size <= self.limit:
                    raise ProtocolValidationError("Invalid frame length")
                if len(self.buffer) < size + 4:
                    break
                payload = bytes(self.buffer[4:size + 4])
                del self.buffer[:size + 4]
                messages.append(validate(_decode(payload), self.side))
            return messages
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            self.failed = True
            raise ProtocolValidationError(str(exc)) from exc

    def end(self):
        if self.failed or self.buffer:
            self.failed = True
            raise ProtocolValidationError("Incomplete or invalid stream")
