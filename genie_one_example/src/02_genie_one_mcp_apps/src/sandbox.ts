import {
  buildAllowAttribute,
  type McpUiSandboxProxyReadyNotification,
  type McpUiSandboxResourceReadyNotification,
} from "@modelcontextprotocol/ext-apps/app-bridge";

if (window.self === window.top) throw new Error("Sandbox must run in an iframe");
if (!document.referrer) throw new Error("Missing embedding referrer");

const expectedHostOrigin = new URL(document.referrer).origin;
if (!/^http:\/\/(localhost|127\.0\.0\.1):8080$/.test(expectedHostOrigin)) {
  throw new Error(`Embedding origin is not allowed: ${expectedHostOrigin}`);
}
const ownOrigin = window.location.origin;

// This access must fail. If it succeeds, untrusted app HTML can reach the host.
try {
  window.top!.document;
  throw new Error("Sandbox origin isolation failed");
} catch (error) {
  if (error instanceof Error && error.message === "Sandbox origin isolation failed") throw error;
}

const inner = document.createElement("iframe");
inner.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms");
document.body.appendChild(inner);

const resourceReady: McpUiSandboxResourceReadyNotification["method"] =
  "ui/notifications/sandbox-resource-ready";
const proxyReady: McpUiSandboxProxyReadyNotification["method"] =
  "ui/notifications/sandbox-proxy-ready";

window.addEventListener("message", (event) => {
  if (event.source === window.parent) {
    if (event.origin !== expectedHostOrigin) return;
    if (event.data?.method === resourceReady) {
      const { html, sandbox, permissions } = event.data.params;
      if (typeof sandbox === "string") inner.setAttribute("sandbox", sandbox);
      const allow = buildAllowAttribute(permissions);
      if (allow) inner.setAttribute("allow", allow);
      if (typeof html === "string") {
        const doc = inner.contentDocument ?? inner.contentWindow?.document;
        if (!doc) throw new Error("Cannot access inner sandbox document");
        doc.open();
        doc.write(html);
        doc.close();
      }
    } else {
      inner.contentWindow?.postMessage(event.data, "*");
    }
  } else if (event.source === inner.contentWindow) {
    if (event.origin !== ownOrigin) return;
    window.parent.postMessage(event.data, expectedHostOrigin);
  }
});

window.parent.postMessage(
  { jsonrpc: "2.0", method: proxyReady, params: {} },
  expectedHostOrigin,
);
