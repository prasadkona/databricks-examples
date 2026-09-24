import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { CallToolResult } from "@modelcontextprotocol/client";
import { callViewAsk, connectToGenieOne, initializeApp, type ToolCallInfo } from "./implementation";
import "./global.css";

const DEFAULT_QUESTION = "get the total revenue for my bakehouse";
const DEFAULT_REDIRECT_URI = "http://localhost:8020/callback";
const PROXY_URL = "http://localhost:8000";

interface Trace {
  sequence: number;
  method: string;
  tool?: string;
  status: number;
  durationMs: number;
}

interface ConnectionForm {
  host: string;
  clientId: string;
  clientSecret: string;
  redirectUri: string;
}

function collectUrls(value: unknown, urls = new Set<string>()): Set<string> {
  if (typeof value === "string") {
    for (const match of value.matchAll(/https?:\/\/[^\s<>"')\]]+/g)) urls.add(match[0]);
  } else if (Array.isArray(value)) {
    value.forEach((item) => collectUrls(item, urls));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectUrls(item, urls));
  }
  return urls;
}

function message(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

async function responseError(response: Response): Promise<string> {
  const body = await response.json().catch(() => ({})) as { error?: string };
  return body.error || `Request failed (HTTP ${response.status})`;
}

function App() {
  const [activeTab, setActiveTab] = useState<"connection" | "query">("connection");
  const [form, setForm] = useState<ConnectionForm>({
    host: "",
    clientId: "",
    clientSecret: "",
    redirectUri: DEFAULT_REDIRECT_URI,
  });
  const [connectedWorkspace, setConnectedWorkspace] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [status, setStatus] = useState("Connection required");
  const [call, setCall] = useState<ToolCallInfo>();
  const [result, setResult] = useState<CallToolResult>();
  const [events, setEvents] = useState<string[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [error, setError] = useState("");
  const serverRef = useRef<
    Awaited<ReturnType<typeof connectToGenieOne>> | undefined
  >(undefined);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const connected = Boolean(serverRef.current && connectedWorkspace);
  const addEvent = (event: string) =>
    setEvents((current) => [...current, `${new Date().toLocaleTimeString()} — ${event}`]);

  useEffect(() => {
    if (!connected) return;
    const timer = window.setInterval(() => {
      fetch(`${PROXY_URL}/api/traces`)
        .then((response) => response.json())
        .then((body: { traces: Trace[] }) => setTraces(body.traces))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [connected]);

  useEffect(() => {
    if (!call || !iframeRef.current) return;
    let active = true;
    initializeApp(iframeRef.current, call, addEvent)
      .then(() => active && setStatus("Genie One interactive View running"))
      .catch((reason) => active && setError(message(reason)));
    call.resultPromise.then(
      (toolResult) => active && setResult(toolResult),
      (reason) => active && setError(message(reason)),
    );
    return () => { active = false; };
  }, [call]);

  function updateForm(field: keyof ConnectionForm, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function configureConnection(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const missing = [
      !form.host.trim() && "workspace host",
      !form.clientId.trim() && "client ID",
      !form.clientSecret.trim() && "client secret",
      !form.redirectUri.trim() && "redirect URL",
    ].filter(Boolean);
    if (missing.length) {
      setError(`Fill in ${missing.join(", ")}.`);
      return;
    }

    setConnecting(true);
    setError("");
    setStatus("Waiting for Databricks OAuth…");
    try {
      const response = await fetch(`${PROXY_URL}/api/configure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: form.host,
          client_id: form.clientId,
          client_secret: form.clientSecret,
          redirect_uri: form.redirectUri,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const configured = await response.json() as { workspace: string };

      setStatus("OAuth succeeded · discovering MCP Apps tools…");
      const server = await connectToGenieOne();
      serverRef.current = server;
      setConnectedWorkspace(configured.workspace);
      setForm((current) => ({ ...current, clientSecret: "" }));
      setStatus("Connected · view_ask discovered");
      addEvent("MCP initialized with io.modelcontextprotocol/ui");
      setActiveTab("query");
    } catch (reason) {
      setStatus("Connection failed");
      setError(message(reason));
    } finally {
      setConnecting(false);
    }
  }

  async function changeConnection() {
    try {
      await serverRef.current?.client.close();
    } catch {
      // The server may already be closed; local proxy reset still proceeds.
    }
    await fetch(`${PROXY_URL}/api/disconnect`, { method: "POST" }).catch(() => undefined);
    serverRef.current = undefined;
    setConnectedWorkspace("");
    setCall(undefined);
    setResult(undefined);
    setEvents([]);
    setTraces([]);
    setError("");
    setStatus("Connection required");
    setActiveTab("connection");
  }

  function executeQuery() {
    const server = serverRef.current;
    if (!server) {
      setError("Connect to Databricks before running a query.");
      setActiveTab("connection");
      return;
    }
    if (!question.trim()) {
      setError("Enter a question before selecting Execute.");
      return;
    }
    setError("");
    setResult(undefined);
    setEvents([]);
    setStatus("Invoking view_ask…");
    try {
      setCall(callViewAsk(server, question.trim()));
    } catch (reason) {
      setError(message(reason));
    }
  }

  const links = result ? [...collectUrls(result)] : [];

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">MODE 3 · IMPROVED MCP APPS UI</p>
          <h1>Genie One interactive View</h1>
          <p className="subtitle">
            Configure OAuth first, then explicitly execute questions in Genie One’s MCP App.
          </p>
        </div>
        <span className={`status ${connected ? "" : "statusNeutral"}`}>{status}</span>
      </header>

      <nav className="tabs" aria-label="Setup steps">
        <button
          className={activeTab === "connection" ? "active" : ""}
          onClick={() => setActiveTab("connection")}
          type="button"
        >
          <span>1</span> Databricks connection
        </button>
        <button
          className={activeTab === "query" ? "active" : ""}
          disabled={!connected}
          onClick={() => setActiveTab("query")}
          type="button"
        >
          <span>2</span> Ask Genie One
        </button>
      </nav>

      {activeTab === "connection" && (
        <section className="connectionPanel">
          <div className="panelHeading">
            <div>
              <h2>Connect to Databricks</h2>
              <p className="muted">
                Configure a Databricks OAuth app before connecting. In the app’s
                redirect URLs, add exactly:
              </p>
              <code className="redirectValue">{form.redirectUri || DEFAULT_REDIRECT_URI}</code>
            </div>
            {connected && <span className="connectedBadge">Connected to {connectedWorkspace}</span>}
          </div>

          <div className="instruction">
            <strong>Databricks setup:</strong> create or open an OAuth app in the
            Databricks Account Console, copy its client ID and secret, add the
            redirect URL shown above, and save the app before selecting Connect.
            The app must be allowed to request the <code>ai-gateway</code> scope.
            This local example keeps one user’s token in process memory; a
            production multi-user app requires isolated server sessions.
          </div>

          <form onSubmit={configureConnection}>
            <div className="formGrid">
              <label>
                Workspace host
                <input
                  autoComplete="url"
                  placeholder="https://your-workspace-host"
                  value={form.host}
                  onChange={(event) => updateForm("host", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                OAuth client ID
                <input
                  autoComplete="username"
                  placeholder="Client ID"
                  value={form.clientId}
                  onChange={(event) => updateForm("clientId", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                OAuth client secret
                <input
                  autoComplete="off"
                  placeholder={connected ? "Cleared after connection" : "Client secret"}
                  type="password"
                  value={form.clientSecret}
                  onChange={(event) => updateForm("clientSecret", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                Redirect URL
                <input
                  autoComplete="off"
                  value={form.redirectUri}
                  onChange={(event) => updateForm("redirectUri", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
            </div>
            <p className="securityNote">
              Credentials are sent only to the local server-side proxy, held in
              memory for OAuth, and are not stored in browser storage or files.
            </p>
            <div className="formActions">
              {connected ? (
                <>
                  <button type="button" className="secondary" onClick={changeConnection}>
                    Change connection
                  </button>
                  <button type="button" onClick={() => setActiveTab("query")}>
                    Continue to query
                  </button>
                </>
              ) : (
                <button type="submit" disabled={connecting}>
                  {connecting ? "Complete OAuth in your browser…" : "Connect to Databricks"}
                </button>
              )}
            </div>
          </form>
        </section>
      )}

      {activeTab === "query" && connected && (
        <>
          <section className="ask">
            <div className="sectionTitlePlain">
              <div>
                <h2>Ask Genie One</h2>
                <p className="muted">
                  Edit or clear the default question. Nothing runs until you select Execute.
                </p>
              </div>
              <button type="button" className="textButton" onClick={() => setActiveTab("connection")}>
                Connection settings
              </button>
            </div>
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="queryActions">
              <button type="button" className="secondary" onClick={() => setQuestion("")}>
                Clear
              </button>
              <button type="button" onClick={executeQuery} disabled={!question.trim()}>
                Execute
              </button>
            </div>
          </section>

          <details className="overview">
            <summary>MCP Apps details — the result panel is Genie One’s View</summary>
            <div className="overviewBody">
              <p>
                This host advertises <code>io.modelcontextprotocol/ui</code>,
                calls <code>view_ask</code> once on Execute, reads the advertised
                <code>ui://</code> resource, and hosts it with AppBridge in an
                isolated iframe. Genie One’s View renders progress, results,
                citations, charts, and Explore in Genie One.
              </p>
            </div>
          </details>

          <section className="view">
            <div className="sectionTitle">
              <h2>Interactive View</h2>
              <span>Genie One’s MCP App · ui:// resource in sandbox</span>
            </div>
            {call ? (
              <iframe ref={iframeRef} title="Genie One MCP App" />
            ) : (
              <div className="placeholder">Ready. Select Execute to call view_ask.</div>
            )}
          </section>

          <div className="details">
            <section>
              <h2>Request trace</h2>
              <div className="traceList">
                {traces.map((trace) => (
                  <div className="trace" key={`${trace.sequence}-${trace.method}`}>
                    <code>{trace.method}{trace.tool ? ` · ${trace.tool}` : ""}</code>
                    <span>HTTP {trace.status} · {trace.durationMs} ms</span>
                  </div>
                ))}
                {!traces.length && <p className="muted">No MCP requests yet.</p>}
              </div>
              {events.length > 0 && (
                <details><summary>AppBridge events</summary><pre>{events.join("\n")}</pre></details>
              )}
            </section>

            <section>
              <h2>Tool input and result</h2>
              <details open><summary>Input</summary><pre>{JSON.stringify(call?.input ?? {}, null, 2)}</pre></details>
              <details><summary>Raw tool result</summary><pre>{JSON.stringify(result ?? {}, null, 2)}</pre></details>
              <h3>Sources and deep links</h3>
              {links.length ? (
                <ul>{links.map((url) => <li key={url}><a href={url} target="_blank" rel="noreferrer">{url}</a></li>)}</ul>
              ) : <p className="muted">Links appear here when Genie One returns them.</p>}
            </section>
          </div>
        </>
      )}

      {error && <div role="alert" className="error"><strong>Error:</strong> {error}</div>}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
