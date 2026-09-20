import { describe, expect, it } from "vitest";
import { formatBytes, formatCount, formatPercent } from "./format";

describe("format", () => {
  it("writes sizes the way the language does", () => {
    expect(formatBytes(9_800_000_000, "ru")).toBe("9.8 ГБ");
    expect(formatBytes(9_800_000_000, "en")).toBe("9.8 GB");
    expect(formatBytes(512, "en")).toBe("512 B");
    expect(formatBytes(1_127_755, "ru")).toBe("1.1 МБ");
  });
  it("groups thousands", () => {
    expect(formatCount(1_495_166, "ru")).toBe("1 495 166");
    expect(formatCount(1_495_166, "en")).toBe("1,495,166");
  });
  it("rounds a fraction to whole percent", () => {
    expect(formatPercent(0.1209)).toBe("12 %");
  });
});
