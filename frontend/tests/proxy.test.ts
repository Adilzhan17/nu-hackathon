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
    assert.equal(Number(req.headers["content-length"]), Buffer.byteLength(received));
    assert.equal(req.headers["transfer-encoding"], undefined);
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
test("BFF preserves an explicit Content-Length for streaming Python uploads", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8765";
  process.env.API_KEY = "test-server-secret";
  t.mock.method(globalThis, "fetch", async (_url: unknown, init: RequestInit) => {
    assert.equal(new Headers(init.headers).get("Content-Length"), "4");
    assert.ok(init.body instanceof ReadableStream);
    return Response.json({ job_id: "job-1" });
  });
  const request = new Request("http://localhost:3000/api/analysis", {
    method: "POST",
    body: "test",
    headers: {
      Origin: "http://localhost:3000",
      "Content-Type": "multipart/form-data; boundary=test",
      "Content-Length": "4",
    },
  });
  assert.equal((await proxy(request, "/api/v1/analyze")).status, 200);
});
test("configured public origin works behind a reverse proxy without trusting forwarded headers", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8765";
  process.env.API_KEY = "test-server-secret";
  process.env.PUBLIC_ORIGIN = "https://forte.example";
  t.after(() => {
    delete process.env.PUBLIC_ORIGIN;
  });
  t.mock.method(globalThis, "fetch", async () => Response.json({ job_id: "job-1" }));
  const request = new Request("http://localhost:3000/api/analysis", {
    method: "POST",
    body: "test",
    headers: {
      Origin: "https://forte.example",
      "Content-Type": "multipart/form-data; boundary=test",
      "Content-Length": "4",
    },
  });
  assert.equal((await proxy(request, "/api/v1/analyze")).status, 200);
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
test("job previews are rewritten to a same-origin authenticated route", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8765";
  process.env.API_KEY = "test-server-secret";
  t.mock.method(globalThis, "fetch", async () =>
    Response.json({
      job_id: "job-1",
      status: "processing",
      preview_url: "/api/v1/jobs/job-1/preview",
    }),
  );
  const response = await proxy(
    new Request("http://localhost:3000/api/jobs/job-1"),
    "/api/v1/jobs/job-1",
  );
  assert.equal((await response.json()).preview_url, "/api/jobs/job-1/preview");
});
test("BFF does not authorize arbitrary preview URLs", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8765";
  process.env.API_KEY = "test-server-secret";
  for (const preview_url of [
    "https://other.example/file.mp4",
    "/api/v1/jobs/another-job/preview",
    "/admin/secrets",
    "/api/v1/jobs/job-1/preview?token=private",
  ]) {
    const mock = t.mock.method(globalThis, "fetch", async () =>
      Response.json({ job_id: "job-1", preview_url }),
    );
    const response = await proxy(
      new Request("http://localhost:3000/api/jobs/job-1"),
      "/api/v1/jobs/job-1",
    );
    assert.equal((await response.json()).preview_url, undefined);
    mock.mock.restore();
  }
});
test("preview streams Range responses and keeps credentials on the server", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8765";
  process.env.API_KEY = "test-server-secret";
  t.mock.method(globalThis, "fetch", async (url: unknown, init: RequestInit) => {
    assert.equal(url, "http://127.0.0.1:8765/api/v1/jobs/job-1/preview");
    const headers = new Headers(init.headers);
    assert.equal(headers.get("range"), "bytes=0-3");
    assert.equal(headers.get("x-api-key"), "test-server-secret");
    return new Response("test", {
      status: 206,
      headers: {
        "Content-Type": "video/mp4",
        "Content-Range": "bytes 0-3/20",
        "Accept-Ranges": "bytes",
        "Content-Length": "4",
        "X-API-Key": "test-server-secret",
      },
    });
  });
  const response = await proxy(
    new Request("http://localhost:3000/api/jobs/job-1/preview", {
      headers: { Range: "bytes=0-3" },
    }),
    "/api/v1/jobs/job-1/preview",
    true,
  );
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-range"), "bytes 0-3/20");
  assert.equal(response.headers.get("x-api-key"), null);
  assert.equal(await response.text(), "test");
});

test("BFF accepts the browser Host after Next normalizes loopback URLs and rejects other origins", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8080";
  process.env.API_KEY = "test-server-secret";
  let forwarded = 0;
  t.mock.method(globalThis, "fetch", async () => {
    forwarded++;
    return Response.json({ job_id: "local-job" }, { status: 202 });
  });
  for (const [origin, status] of [
    ["http://127.0.0.1:3000", 202],
    ["http://localhost:3000", 403],
    ["http://127.0.0.1:4000", 403],
    ["https://other.example", 403],
    ["null", 403],
    ["", 403],
  ] as const) {
    const form = new FormData();
    form.set("file", new File(["video"], "test.mov"));
    const request = new Request("http://localhost:3000/api/analysis", {
      method: "POST",
      body: form,
      headers: {
        host: "127.0.0.1:3000",
        origin,
        "x-forwarded-host": "other.example",
      },
    });
    assert.equal((await proxy(request, "/api/v1/analyze")).status, status, origin);
  }
  assert.equal(forwarded, 1);
});
