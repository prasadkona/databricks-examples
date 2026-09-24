import {
  AppBridge,
  PostMessageTransport,
  RESOURCE_MIME_TYPE,
  buildAllowAttribute,
  getToolUiResourceUri,
  type McpUiResourceCsp,
  type McpUiResourcePermissions,
  type McpUiSandboxProxyReadyNotification,
} from "@modelcontextprotocol/ext-apps/app-bridge";
import {
  Client,
  StreamableHTTPClientTransport,
  type CallToolResult,
  type Resource,
  type Tool,
} from "@modelcontextprotocol/client";

const APP_EXTENSION = "io.modelcontextprotocol/ui";
const SANDBOX_URL = "http://localhost:8081/sandbox.html";
const IMPLEMENTATION = { name: "Genie One MCP Apps local host", version: "1.0.0" };

export interface ServerInfo {
  client: Client;
  tools: Map<string, Tool>;
  resources: Map<string, Resource>;
}

export interface UiResource {
  html: string;
  csp?: McpUiResourceCsp;
  permissions?: McpUiResourcePermissions;
}

export interface ToolCallInfo {
  server: ServerInfo;
  tool: Tool;
  input: Record<string, unknown>;
  resultPromise: Promise<CallToolResult>;
  resourcePromise: Promise<UiResource>;
}

export async function connectToGenieOne(): Promise<ServerInfo> {
  const client = new Client(IMPLEMENTATION, {
    capabilities: {
      extensions: {
        [APP_EXTENSION]: { mimeTypes: [RESOURCE_MIME_TYPE] },
      },
    },
  });
  await client.connect(
    new StreamableHTTPClientTransport(new URL("http://localhost:8000/mcp")),
  );
  const listed = await client.listTools();
  const tools = new Map(listed.tools.map((tool) => [tool.name, tool]));
  const viewAsk = tools.get("view_ask");
  if (!viewAsk) {
    throw new Error(
      `Genie One did not expose view_ask. Discovered: ${[...tools.keys()].join(", ") || "none"}`,
    );
  }
  let resources = new Map<string, Resource>();
  try {
    const listedResources = await client.listResources();
    resources = new Map(listedResources.resources.map((resource) => [resource.uri, resource]));
  } catch {
    // resources/list is optional; resources/read for the tool URI is sufficient.
  }
  return { client, tools, resources };
}

export function callViewAsk(
  server: ServerInfo,
  question: string,
): ToolCallInfo {
  const tool = server.tools.get("view_ask");
  if (!tool) throw new Error("view_ask is unavailable");
  const resourceUri = getToolUiResourceUri(tool);
  if (!resourceUri) throw new Error("view_ask did not advertise _meta.ui.resourceUri");
  const input = { question };
  return {
    server,
    tool,
    input,
    resultPromise: server.client.callTool({
      name: tool.name,
      arguments: input,
    }) as Promise<CallToolResult>,
    resourcePromise: readUiResource(server, resourceUri),
  };
}

async function readUiResource(server: ServerInfo, uri: string): Promise<UiResource> {
  const response = await server.client.readResource({ uri });
  if (response.contents.length !== 1) {
    throw new Error(`Expected one UI resource, received ${response.contents.length}`);
  }
  const content = response.contents[0];
  if (content.mimeType !== RESOURCE_MIME_TYPE) {
    throw new Error(`Unsupported UI MIME type: ${content.mimeType ?? "missing"}`);
  }
  const html = "blob" in content ? atob(content.blob) : content.text;
  const contentMeta = (content as { _meta?: Record<string, unknown>; meta?: Record<string, unknown> })
    ._meta ?? (content as { meta?: Record<string, unknown> }).meta;
  const listingMeta = server.resources.get(uri)?._meta;
  const ui = ((contentMeta?.ui ?? listingMeta?.ui) as {
    csp?: McpUiResourceCsp;
    permissions?: McpUiResourcePermissions;
  } | undefined);
  return { html, csp: ui?.csp, permissions: ui?.permissions };
}

export async function loadSandbox(
  iframe: HTMLIFrameElement,
  csp?: McpUiResourceCsp,
  permissions?: McpUiResourcePermissions,
): Promise<void> {
  iframe.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms");
  const allow = buildAllowAttribute(permissions);
  if (allow) iframe.setAttribute("allow", allow);
  const readyMethod: McpUiSandboxProxyReadyNotification["method"] =
    "ui/notifications/sandbox-proxy-ready";
  const ready = new Promise<void>((resolve) => {
    const listener = (event: MessageEvent) => {
      if (event.source === iframe.contentWindow && event.data?.method === readyMethod) {
        window.removeEventListener("message", listener);
        resolve();
      }
    };
    window.addEventListener("message", listener);
  });
  const url = new URL(SANDBOX_URL);
  if (csp) url.searchParams.set("csp", JSON.stringify(csp));
  iframe.src = url.href;
  await ready;
}

export async function initializeApp(
  iframe: HTMLIFrameElement,
  info: ToolCallInfo,
  onEvent: (message: string) => void,
): Promise<AppBridge> {
  const resource = await info.resourcePromise;
  await loadSandbox(iframe, resource.csp, resource.permissions);

  const bridge = new AppBridge(
    info.server.client,
    IMPLEMENTATION,
    {
      openLinks: {},
      serverTools: info.server.client.getServerCapabilities()?.tools,
      serverResources: info.server.client.getServerCapabilities()?.resources,
      updateModelContext: { text: {} },
    },
    {
      hostContext: {
        theme: "light",
        platform: "web",
        displayMode: "inline",
        availableDisplayModes: ["inline", "fullscreen"],
        containerDimensions: { maxHeight: 6000 },
      },
    },
  );
  bridge.onopenlink = async ({ url }) => {
    onEvent(`View requested link: ${url}`);
    window.open(url, "_blank", "noopener,noreferrer");
    return {};
  };
  bridge.onmessage = async ({ role }) => {
    onEvent(`View sent ${role} message`);
    return {};
  };
  bridge.onloggingmessage = ({ level, data }) => onEvent(`View ${level}: ${String(data)}`);
  bridge.onupdatemodelcontext = async () => {
    onEvent("View updated model context");
    return {};
  };
  bridge.onsizechange = async ({ width, height }) => {
    if (width) iframe.style.minWidth = `min(${width}px, 100%)`;
    if (height) iframe.style.height = `${height}px`;
  };
  bridge.onrequestdisplaymode = async ({ mode }) => ({ mode });

  let initialized: () => void = () => undefined;
  const appInitialized = new Promise<void>((resolve) => { initialized = resolve; });
  const priorInitialized = bridge.oninitialized;
  bridge.oninitialized = (...args) => {
    initialized();
    priorInitialized?.(...args);
  };
  await bridge.connect(
    new PostMessageTransport(iframe.contentWindow!, iframe.contentWindow!),
  );
  await bridge.sendSandboxResourceReady(resource);
  await appInitialized;
  onEvent("MCP App initialized");
  bridge.sendToolInput({ arguments: info.input });
  info.resultPromise.then(
    (result) => {
      onEvent("Tool result forwarded to View");
      bridge.sendToolResult(result);
    },
    (error) => bridge.sendToolCancelled({
      reason: error instanceof Error ? error.message : String(error),
    }),
  );
  return bridge;
}
