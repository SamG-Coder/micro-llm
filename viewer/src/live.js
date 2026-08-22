import { HTR1_HEADER_BYTES, RECORD_BYTES } from "./constants.js";
import { decodeRecord, parseHTR1 } from "./ring.js";

export function stopSampleReplay(globalObj = globalThis) {
  if (globalObj.__sampleReplay != null) {
    const clear = globalObj.clearInterval || clearInterval;
    clear(globalObj.__sampleReplay);
    globalObj.__sampleReplay = null;
  }
}

export function decodeHostPayload(data) {
  if (!data) return { kind: "empty" };
  if (typeof data === "string") {
    try {
      data = JSON.parse(data);
    } catch {
      return { kind: "unknown", raw: data };
    }
  }
  if (data instanceof ArrayBuffer) {
    return decodeHostBytes(data);
  }
  if (ArrayBuffer.isView(data)) {
    return decodeHostBytes(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data.type === "live-attach") {
    return { kind: "attach", attach: data };
  }
  if (data.type === "htr1" && data.b64) {
    const bin = Uint8Array.from(atob(data.b64), (c) => c.charCodeAt(0));
    return decodeHostBytes(bin.buffer);
  }
  if (data.type === "htr1") {
    return { kind: "htr1-notice" };
  }
  return { kind: "unknown", raw: data };
}

function decodeHostBytes(buf, offset = 0, length = buf.byteLength) {
  const slice = buf.slice ? buf.slice(offset, offset + length) : buf;
  if (length === RECORD_BYTES) {
    return { kind: "record", record: decodeRecord(slice) };
  }
  if (length >= HTR1_HEADER_BYTES + RECORD_BYTES) {
    const parsed = parseHTR1(slice);
    const rec = parsed.records[parsed.records.length - 1];
    return { kind: "record", record: rec };
  }
  return { kind: "unknown" };
}

export function installLivePush(onPayload, globalObj = globalThis) {
  const handle = (data) => {
    const parsed = decodeHostPayload(data);
    onPayload(parsed, data);
  };
  globalObj.__htr1Push = (data) => handle(data);
  const wv = globalObj.chrome && globalObj.chrome.webview;
  if (wv && typeof wv.addEventListener === "function") {
    wv.addEventListener("message", (ev) => handle(ev.data));
    wv.addEventListener("sharedbufferreceived", (ev) => {
      const buf = ev.getBuffer();
      handle(buf);
    });
  }
  return handle;
}
