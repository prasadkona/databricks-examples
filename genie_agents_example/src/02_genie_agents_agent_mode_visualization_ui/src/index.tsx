import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./global.css";

const DEFAULT_QUESTION =
  "show bakehouse total revenue by franchise as a bar chart";
const DEFAULT_REDIRECT_URI = "http://localhost:8020/callback";
const PROXY_URL = "http://localhost:8000";

interface ConnectionForm {
  host: string;
  agentId: string;
  clientId: string;
  clientSecret: string;
  redirectUri: string;
}

interface Trace {
  sequence: number;
  operation: string;
  status: number;
  durationMs: number;
}

interface OutputItem {
  type?: string;
  name?: string;
  arguments?: string;
  output?: unknown;
  metadata?: {
    viz?: {
      attachment_id?: string;
      query_attachment_id?: string;
    };
  };
  content?: Array<{ text?: string; metadata?: unknown }>;
}

interface AgentResponse {
  id?: string;
  status?: string;
  conversation_id?: string;
  output?: OutputItem[];
  error?: unknown;
}

interface SsePayload {
  type?: string;
  sequence_number?: number;
  item?: OutputItem;
  response?: AgentResponse;
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

async function responseError(response: Response): Promise<string> {
  const body = (await response.json().catch(() => ({}))) as {
    error?: string;
    detail?: unknown;
  };
  if (body.error) return body.error;
  if (body.detail) {
    return typeof body.detail === "string"
      ? body.detail
      : JSON.stringify(body.detail);
  }
  return `Request failed (HTTP ${response.status})`;
}

function collectUrls(value: unknown, urls = new Set<string>()): Set<string> {
  if (typeof value === "string") {
    for (const match of value.matchAll(/https?:\/\/[^\s<>"')\]]+/g)) {
      urls.add(match[0]);
    }
  } else if (Array.isArray(value)) {
    value.forEach((item) => collectUrls(item, urls));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectUrls(item, urls));
  }
  return urls;
}

async function readSse(
  response: Response,
  onEvent: (payload: SsePayload) => void,
): Promise<AgentResponse> {
  if (!response.body) throw new Error("The SSE response did not include a body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal: AgentResponse | undefined;

  function consume(block: string) {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data || data === "[DONE]") return;
    const payload = JSON.parse(data) as SsePayload;
    onEvent(payload);
    if (payload.type === "response.completed") terminal = payload.response;
    if (payload.type === "response.failed") {
      throw new Error(`Agent response failed: ${JSON.stringify(payload.response?.error)}`);
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) consume(block);
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
  if (!terminal) throw new Error("SSE stream ended without response.completed.");
  return terminal;
}

function parseArguments(item: OutputItem): Record<string, unknown> {
  try {
    return JSON.parse(item.arguments ?? "{}") as Record<string, unknown>;
  } catch {
    return {};
  }
}

function App() {
  const [activeTab, setActiveTab] = useState<"connection" | "query">("connection");
  const [form, setForm] = useState<ConnectionForm>({
    host: "",
    agentId: "",
    clientId: "",
    clientSecret: "",
    redirectUri: DEFAULT_REDIRECT_URI,
  });
  const [workspace, setWorkspace] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [running, setRunning] = useState(false);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [status, setStatus] = useState("Connection required");
  const [events, setEvents] = useState<SsePayload[]>([]);
  const [response, setResponse] = useState<AgentResponse>();
  const [traces, setTraces] = useState<Trace[]>([]);
  const [error, setError] = useState("");
  const [visualizationError, setVisualizationError] = useState("");

  const connected = Boolean(workspace);

  useEffect(() => {
    if (!connected) return;
    const timer = window.setInterval(() => {
      fetch(`${PROXY_URL}/api/traces`)
        .then((result) => result.json())
        .then((body: { traces: Trace[] }) => setTraces(body.traces))
        .catch(() => undefined);
    }, 750);
    return () => window.clearInterval(timer);
  }, [connected]);

  const output = response?.output ?? [];
  const answer = output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .map((part) => part.text?.trim())
    .filter((text): text is string => Boolean(text))
    .join("\n\n");
  const sqlCalls = output
    .filter((item) => item.type === "function_call" && item.name === "execute_sql")
    .map(parseArguments);
  const queryOutputs = output.filter(
    (item) => item.type === "function_call_output" && !item.metadata?.viz,
  );
  const visualization = output.find(
    (item) =>
      item.type === "function_call_output" &&
      Boolean(item.metadata?.viz?.attachment_id),
  );
  const attachmentId = visualization?.metadata?.viz?.attachment_id;
  const visualizationUrl =
    response?.conversation_id && response.id && attachmentId
      ? `${PROXY_URL}/api/visualizations/${encodeURIComponent(response.conversation_id)}/${encodeURIComponent(response.id)}/${encodeURIComponent(attachmentId)}`
      : "";
  const links = useMemo(() => [...collectUrls(response)], [response]);

  function updateForm(field: keyof ConnectionForm, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function configureConnection(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setConnecting(true);
    setError("");
    setStatus("Waiting for Databricks OAuth…");
    try {
      const result = await fetch(`${PROXY_URL}/api/configure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: form.host,
          agent_id: form.agentId,
          client_id: form.clientId,
          client_secret: form.clientSecret,
          redirect_uri: form.redirectUri,
        }),
      });
      if (!result.ok) throw new Error(await responseError(result));
      const configured = (await result.json()) as { workspace: string };
      setWorkspace(configured.workspace);
      setForm((current) => ({ ...current, clientSecret: "" }));
      setStatus("Connected · visualization enabled");
      setActiveTab("query");
    } catch (reason) {
      setStatus("Connection failed");
      setError(errorMessage(reason));
    } finally {
      setConnecting(false);
    }
  }

  async function disconnect() {
    await fetch(`${PROXY_URL}/api/disconnect`, { method: "POST" }).catch(
      () => undefined,
    );
    setWorkspace("");
    setEvents([]);
    setResponse(undefined);
    setTraces([]);
    setError("");
    setVisualizationError("");
    setStatus("Connection required");
    setActiveTab("connection");
  }

  async function executeQuery() {
    if (!question.trim()) {
      setError("Enter a question before selecting Execute.");
      return;
    }
    setRunning(true);
    setError("");
    setVisualizationError("");
    setEvents([]);
    setResponse(undefined);
    setStatus("Genie Agent Mode is researching…");
    try {
      const result = await fetch(`${PROXY_URL}/api/responses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      if (!result.ok) throw new Error(await responseError(result));
      const completed = await readSse(result, (payload) => {
        setEvents((current) => [...current, payload]);
        setStatus(payload.type?.replaceAll(".", " → ") ?? "Streaming");
      });
      setResponse(completed);
      setStatus("Completed · visualization enabled");
    } catch (reason) {
      setStatus("Request failed");
      setError(errorMessage(reason));
    } finally {
      setRunning(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">EXAMPLE 02 · GENIE AGENT MODE UI</p>
          <h1>Genie Agent visualization experience</h1>
          <p className="subtitle">
            Genie Agent Mode SSE with <code>enable_viz: true</code>, rendered in a custom UI.
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
          <span>2</span> Ask Genie Agent
        </button>
      </nav>

      {activeTab === "connection" && (
        <section className="connectionPanel">
          <div className="panelHeading">
            <div>
              <h2>Connect to Databricks</h2>
              <p className="muted">
                Add this exact redirect URL to your Databricks OAuth app:
              </p>
              <code className="redirectValue">{form.redirectUri}</code>
            </div>
            {connected && <span className="connectedBadge">Connected to {workspace}</span>}
          </div>

          <div className="instruction">
            The OAuth app must allow the <code>genie</code> scope. This example
            stores one user token only in the local proxy’s memory. Every Agent
            Mode request made by this UI forces <code>enable_viz: true</code>.
          </div>

          <form onSubmit={configureConnection}>
            <div className="formGrid">
              <label>
                Workspace host
                <input
                  placeholder="https://your-workspace-host"
                  required
                  value={form.host}
                  onChange={(event) => updateForm("host", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                Genie Agent ID
                <input
                  placeholder="32-character agent ID"
                  required
                  value={form.agentId}
                  onChange={(event) => updateForm("agentId", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                OAuth client ID
                <input
                  placeholder="Client ID"
                  required
                  value={form.clientId}
                  onChange={(event) => updateForm("clientId", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label>
                OAuth client secret
                <input
                  placeholder={connected ? "Cleared after connection" : "Client secret"}
                  required
                  type="password"
                  value={form.clientSecret}
                  onChange={(event) => updateForm("clientSecret", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
              <label className="wide">
                Redirect URL
                <input
                  required
                  value={form.redirectUri}
                  onChange={(event) => updateForm("redirectUri", event.target.value)}
                  disabled={connecting || connected}
                />
              </label>
            </div>
            <p className="securityNote">
              Credentials go only to the local server-side proxy and are not
              stored in browser storage or written to files.
            </p>
            <div className="formActions">
              {connected ? (
                <>
                  <button type="button" className="secondary" onClick={disconnect}>
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
                <h2>Ask the Genie Agent</h2>
                <p className="muted">
                  Use a grouped or time-series question to make a chart appropriate.
                </p>
              </div>
              <span className="vizBadge">Visualization ON</span>
            </div>
            <textarea
              aria-label="Question"
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="queryActions">
              <button type="button" className="secondary" onClick={() => setQuestion("")}>
                Clear
              </button>
              <button
                type="button"
                onClick={executeQuery}
                disabled={running || !question.trim()}
              >
                {running ? "Agent is working…" : "Execute"}
              </button>
            </div>
          </section>

          <section className="result">
            <div className="sectionTitle">
              <h2>Agent response</h2>
              <span>REST + Server-Sent Events</span>
            </div>
            {!response && (
              <div className="placeholder">
                {running ? "Streaming Genie Agent Mode output…" : "Ready. Select Execute."}
              </div>
            )}
            {response && (
              <div className="resultBody">
                <div className="answer">
                  <h3>Answer</h3>
                  <div className="prose">{answer || "No narrative text returned."}</div>
                </div>
                <div className="chartCard">
                  <div className="chartHeading">
                    <div>
                      <p className="label">Generated visualization</p>
                      <h3>{String(visualization?.output ?? "Chart")}</h3>
                    </div>
                    <span className="vizBadge">enable_viz=true</span>
                  </div>
                  {visualizationUrl ? (
                    <>
                      <img
                        src={visualizationUrl}
                        alt={String(visualization?.output ?? "Genie Agent visualization")}
                        onError={() =>
                          setVisualizationError(
                            "The visualization metadata arrived, but the rendered image could not be downloaded.",
                          )
                        }
                      />
                      {visualizationError && (
                        <p className="warning">{visualizationError}</p>
                      )}
                    </>
                  ) : (
                    <p className="muted">
                      The Genie Agent did not generate a visualization for this response.
                      Visualizations are requested but are produced only when appropriate.
                    </p>
                  )}
                </div>
              </div>
            )}
          </section>

          <div className="details">
            <section>
              <h2>SSE event stream</h2>
              <div className="traceList">
                {events.map((event, index) => (
                  <div className="trace" key={`${event.sequence_number ?? index}-${event.type}`}>
                    <code>seq={event.sequence_number ?? "?"}</code>
                    <span>{event.type}</span>
                  </div>
                ))}
                {!events.length && <p className="muted">No events yet.</p>}
              </div>
              <h3>HTTP trace</h3>
              {traces.map((trace) => (
                <div className="trace" key={trace.sequence}>
                  <code>{trace.operation}</code>
                  <span>HTTP {trace.status} · {trace.durationMs} ms</span>
                </div>
              ))}
            </section>

            <section>
              <h2>SQL and data</h2>
              {sqlCalls.map((sql, index) => (
                <details open key={index}>
                  <summary>{String(sql.title ?? `SQL ${index + 1}`)}</summary>
                  <pre>{String(sql.sql ?? "")}</pre>
                </details>
              ))}
              {queryOutputs.map((item, index) => (
                <details key={index}>
                  <summary>Query result {index + 1}</summary>
                  <pre>
                    {typeof item.output === "string"
                      ? item.output
                      : JSON.stringify(item.output, null, 2)}
                  </pre>
                </details>
              ))}
              <h3>Sources and deep links</h3>
              {links.length ? (
                <ul>
                  {links.map((url) => (
                    <li key={url}>
                      <a href={url} target="_blank" rel="noreferrer">{url}</a>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">Links appear here when the Genie Agent returns them.</p>
              )}
              <details>
                <summary>Raw completed response</summary>
                <pre>{JSON.stringify(response ?? {}, null, 2)}</pre>
              </details>
            </section>
          </div>
        </>
      )}

      {error && <div role="alert" className="error"><strong>Error:</strong> {error}</div>}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
