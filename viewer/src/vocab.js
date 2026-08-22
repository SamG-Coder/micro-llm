// Tiny toy vocab for the sample ring. Not the Qwen tokenizer.
// Unmapped ids print as the numeric id.

export const TOY_VOCAB = Object.freeze({
  10: "ok",
  11: ".",
  12: " ",
  13: "here",
  14: "is",
  15: "a",
  16: "small",
  17: "function",
  18: "that",
  19: "returns",
  20: "the",
  21: "code",
  22: "for",
  23: "this",
  24: "job",
  25: "\n",
  26: "void",
  27: "int",
  28: "main",
  29: "(",
  30: ")",
  31: "{",
  32: "}",
  33: "return",
  34: "0",
  35: ";",
});

export const RARE_ID = 200003;

// Deterministic outgoing sample. One rare id. Spaces are real tokens.
export const SAMPLE_IDS = Object.freeze([
  13, 12, 14, 12, 15, 12, 16, 12, 17, 12, 18, 12, 19, 12, 20, 12, 21, 11, 25,
  26, 12, 27, 12, 28, 29, 30, 12, 31, 25, 12, 33, 12, 34, 35, 25, 32, 11, 12,
  23, 12, 24, 11, RARE_ID, 25, 10, 11, 12, 21,
]);

export const SPECIAL_TOKEN_INDEX = 27;

export function decodeId(id) {
  if (Object.prototype.hasOwnProperty.call(TOY_VOCAB, id)) {
    return TOY_VOCAB[id];
  }
  return `#${id}`;
}
