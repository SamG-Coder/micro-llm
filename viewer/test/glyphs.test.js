import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { glyphLabel } from "../src/glyphs.js";

describe("flying glyph labels", () => {
  it("keeps words and ids readable in the volume", () => {
    assert.equal(glyphLabel("function"), "function");
    assert.equal(glyphLabel(" "), "␣");
    assert.equal(glyphLabel("\n"), "\\n");
    assert.equal(glyphLabel("#200003"), "#200003");
  });
});
