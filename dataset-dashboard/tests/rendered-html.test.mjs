import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { createServer } from "node:net";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function availablePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.equal(typeof address, "object");
  const port = address.port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function waitForPage(url, child) {
  let lastError;
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (child.exitCode !== null) {
      throw new Error(`Dashboard server exited with code ${child.exitCode}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return response.text();
      lastError = new Error(`Dashboard returned HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw lastError || new Error("Dashboard did not start");
}

test("production dashboard renders and every diagnostic image exists", async (context) => {
  const port = await availablePort();
  const child = spawn("npm", ["run", "start", "--", "--port", String(port)], {
    cwd: root,
    env: process.env,
    stdio: "ignore",
  });
  context.after(() => {
    if (child.exitCode === null) child.kill("SIGTERM");
  });

  const html = await waitForPage(`http://127.0.0.1:${port}/`, child);
  assert.match(html, /Spider Lab/);
  assert.match(html, /Screenshot QA/);
  assert.match(html, /GUI grounding/);
  assert.match(html, /QA · latest/);
  assert.match(html, /QA · baseline/);

  const payload = JSON.parse(await readFile(path.join(root, "app/qa-probe.json"), "utf8"));
  for (const task of ["qa", "grounding", "action"]) {
    if (!payload[task]) continue;
    assert.ok(payload[task].records.length > 0, `${task} must retain diagnostic records`);
    for (const record of payload[task].records) {
      const image = path.join(root, "public", record.image.replace(/^\//, ""));
      const bytes = await readFile(image);
      assert.ok(bytes.length > 0, `${record.image} must be non-empty`);
    }
  }
});
