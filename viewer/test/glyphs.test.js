import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SAMPLE_TOKEN_MS } from "../src/constants.js";
import { glyphLabel, glyphScale } from "../src/glyphs.js";

describe("flying glyph labels", () => {
  it("replays the sample ring fast, not at 160ms", () => {
    assert.ok(SAMPLE_TOKEN_MS >= 40 && SAMPLE_TOKEN_MS <= 60);
  });

  it("keeps words and ids readable in the volume", () => {
    assert.equal(glyphLabel("function"), "function");
    assert.equal(glyphLabel(" "), "␣");
    assert.equal(glyphLabel("\n"), "\\n");
    assert.equal(glyphLabel("#200003"), "#200003");
    const word = glyphScale("function");
    const id = glyphScale("#200003");
    assert.ok(word.x >= 1.1 && word.x <= 2.5);
    assert.ok(word.y >= 0.45 && word.y <= 0.7);
    assert.ok(id.x >= 1.1 && id.x <= 2.5);
  });
});
