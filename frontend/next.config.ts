import type { NextConfig } from "next";
const config: NextConfig = {
  poweredByHeader: false,
  distDir: process.env.FORTE_NEXT_DIST || ".next-forte-single-ui",
  async redirects() {
    // One public UI. Keep Next API compatibility for existing consumers.
    return [
      { source: "/", destination: "/demo/gemma", permanent: false },
      { source: "/analysis/new", destination: "/demo/gemma", permanent: false },
      { source: "/analysis/tests", destination: "/scenarios", permanent: false },
      { source: "/analysis", destination: "/analyze", permanent: false },
      { source: "/analysis/:path*", destination: "/analyze", permanent: false },
    ];
  },
  async rewrites() {
    return ["/analyze", "/scenarios", "/api/models", "/api/uploads/:path*"].map(source => ({
      source, destination: `http://127.0.0.1:8765${source}`,
    }));
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
      { source: "/analyze", headers: [{key: "X-Frame-Options", value: "SAMEORIGIN"}] },
    ];
  },
};
export default config;
