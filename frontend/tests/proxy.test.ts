import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { proxy } from "../lib/proxy";
test("BFF streams the file with the secret header and strips upstream headers", async (t) => {
  let received = "";
  const server = createServer(async (req, res) => {
    assert.equal(req.url, "/api/v1/analyze");
    assert.equal(req.headers["x-api-key"], "test-server-secret");
    assert.match(req.headers["content-type"] ?? "", /multipart\/form-data; boundary=/);
    for await (const chunk of req) received += chunk.toString();
    res.setHeader("X-API-Key", "test-server-secret");
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ job_id: "proxy-job" }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  process.env.API_BASE_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  process.env.API_KEY = "test-server-secret";
  const form = new FormData();
  form.set("file", new File(["test-video-payload"], "cashier.mov"));
  const request = new Request("http://localhost:3000/api/analysis", {
    method: "POST",
    body: form,
    headers: { origin: "http://localhost:3000" },
  });
  const response = await proxy(request, "/api/v1/analyze");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("X-API-Key"), null);
  assert.deepEqual(await response.json(), { job_id: "proxy-job" });
  assert.match(received, /name="file"; filename="cashier.mov"/);
  assert.match(received, /test-video-payload/);
});
test("BFF rejects foreign origins and missing configuration", async () => {
  const form = new FormData();
  form.set("file", new File(["test"], "test.mov"));
  const foreign = new Request("http://localhost:3000/api/analysis", {
    method: "POST",
    body: form,
    headers: { origin: "https://other.example" },
  });
  assert.equal((await proxy(foreign, "/api/v1/analyze")).status, 403);
  delete process.env.API_KEY;
  assert.equal(
    (await proxy(new Request("http://localhost:3000/api/jobs/job-1"), "/api/v1/jobs/job-1")).status,
    503,
  );
});
test("BFF sanitizes upstream errors and will not follow redirects with credentials", async (t) => {
  process.env.API_KEY = "test-server-secret";
  t.mock.method(globalThis, "fetch", async (_url: unknown, options: RequestInit) => {
    assert.equal(options.redirect, "error");
    return new Response("private upstream error test-server-secret", { status: 401 });
  });
  const response = await proxy(
    new Request("http://localhost:3000/api/jobs/job-1"),
    "/api/v1/jobs/job-1",
  );
  assert.equal(response.status, 401);
  assert.equal((await response.text()).includes("test-server-secret"), false);
});
