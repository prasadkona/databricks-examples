import express from "express";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import type { McpUiResourceCsp } from "@modelcontextprotocol/ext-apps";

const directory = join(dirname(fileURLToPath(import.meta.url)), "dist");
const hostPort = 8080;
const sandboxPort = 8081;

function safeDomains(domains?: string[]): string[] {
  return (domains ?? []).filter((domain) =>
    typeof domain === "string" && !/[;\r\n'" ]/.test(domain)
  );
}

function cspHeader(csp?: McpUiResourceCsp): string {
  const resources = safeDomains(csp?.resourceDomains).join(" ");
  const connections = safeDomains(csp?.connectDomains).join(" ");
  const frames = safeDomains(csp?.frameDomains).join(" ");
  const bases = safeDomains(csp?.baseUriDomains).join(" ");
  return [
    "default-src 'self' 'unsafe-inline'",
    `script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: data: ${resources}`,
    `style-src 'self' 'unsafe-inline' blob: data: ${resources}`,
    `img-src 'self' data: blob: ${resources}`,
    `font-src 'self' data: blob: ${resources}`,
    `media-src 'self' data: blob: ${resources}`,
    `connect-src 'self' ${connections}`,
    `worker-src 'self' blob: ${resources}`,
    frames ? `frame-src ${frames}` : "frame-src 'none'",
    "object-src 'none'",
    bases ? `base-uri ${bases}` : "base-uri 'none'",
  ].join("; ");
}

const host = express();
host.use((request, response, next) => {
  if (request.path === "/sandbox.html") {
    response.status(404).send("Sandbox is available only on its isolated origin");
    return;
  }
  next();
});
host.use(express.static(directory));
host.get("/", (_request, response) => response.redirect("/index.html"));

const sandbox = express();
sandbox.get(["/", "/sandbox.html"], (request, response) => {
  let csp: McpUiResourceCsp | undefined;
  if (typeof request.query.csp === "string") {
    try {
      csp = JSON.parse(request.query.csp) as McpUiResourceCsp;
    } catch {
      response.status(400).send("Invalid CSP");
      return;
    }
  }
  response.setHeader("Content-Security-Policy", cspHeader(csp));
  response.setHeader("Cache-Control", "no-store");
  response.sendFile(join(directory, "sandbox.html"));
});
sandbox.use((_request, response) => response.status(404).send("Not found"));

host.listen(hostPort, "127.0.0.1", () => {
  console.log(`Host:    http://localhost:${hostPort}`);
});
sandbox.listen(sandboxPort, "127.0.0.1", () => {
  console.log(`Sandbox: http://localhost:${sandboxPort}`);
});
