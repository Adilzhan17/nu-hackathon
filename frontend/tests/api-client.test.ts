import test from "node:test";
import assert from "node:assert/strict";
delete process.env.NEXT_PUBLIC_USE_MOCK_API;
const client = import("../services/api");
test("real mode is the default when no demo flag is configured", async () => {
  assert.equal((await client).USE_MOCK_API, false);
});
test("real adapter sends multipart without a browser API key", async (t) => {
  const { analyzeVideo } = await client;
  t.mock.method(globalThis, "fetch", async (path: string, init: RequestInit) => {
    assert.equal(path, "/api/analysis");
    assert.ok(init.body instanceof FormData);
    assert.equal((init.body.get("file") as File).name, "cashier.mov");
    assert.equal(new Headers(init.headers).has("X-API-Key"), false);
    return Response.json({ job_id: "job-1" });
  });
  assert.deepEqual(await analyzeVideo(new File(["video"], "cashier.mov"), "violation"), {
    job_id: "job-1",
  });
});
test("unknown demo IDs are not replaced with fabricated mock results", async (t) => {
  const { getJob } = await client;
  t.mock.method(globalThis, "fetch", async (path: string) => {
    assert.equal(path, "/api/jobs/demo-unknown");
    return new Response("", { status: 404 });
  });
  await assert.rejects(() => getJob("demo-unknown"), /Задача не найдена/);
});
test("real adapter explains unauthorized, network and malformed responses", async (t) => {
  const { getJob } = await client;
  const responses = [
    async () => new Response("", { status: 401 }),
    async () => {
      throw new TypeError("Failed to fetch");
    },
    async () => new Response(""),
    async () => Response.json({ job_id: "other-job", status: "processing" }),
  ];
  const expected = [
    /не авторизовал/,
    /Не удалось связаться/,
    /пустой или некорректный/,
    /API-контракту/,
  ];
  for (let i = 0; i < responses.length; i++) {
    const mock = t.mock.method(globalThis, "fetch", responses[i]);
    await assert.rejects(() => getJob("job-1"), expected[i]);
    mock.mock.restore();
  }
});
