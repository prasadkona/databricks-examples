import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { CallToolResult } from "@modelcontextprotocol/client";
import { callViewAsk, connectToGenieOne, initializeApp, type ToolCallInfo } from "./implementation";
import "./global.css";

const DEFAULT_QUESTION = "get the total revenue for my bakehouse";

interface Trace {
  sequence: number;
  method: string;
  tool?: string;
  status: number;
  durationMs: number;
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

function App() {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [status, setStatus] = useState("Connecting to Genie One MCP…");
  const [call, setCall] = useState<ToolCallInfo>();
  const [result, setResult] = useState<CallToolResult>();
  const [events, setEvents] = useState<string[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [error, setError] = useState("");
  const serverRef = useRef<Awaited<ReturnType<typeof connectToGenieOne>> | undefined>(undefined);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const startedRef = useRef(false);

  const addEvent = (event: string) =>
    setEvents((current) => [...current, `${new Date().toLocaleTimeString()} — ${event}`]);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    connectToGenieOne()
      .then((server) => {
        serverRef.current = server;
        setStatus("Connected · view_ask discovered");
        addEvent("MCP initialized with io.modelcontextprotocol/ui");
        startCall(server, DEFAULT_QUESTION);
      })
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus("Connection failed");
      });
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      fetch("http://localhost:8000/api/traces")
        .then((response) => response.json())
        .then((body: { traces: Trace[] }) => setTraces(body.traces))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!call || !iframeRef.current) return;
    let active = true;
    initializeApp(iframeRef.current, call, addEvent)
      .then(() => active && setStatus("Genie One interactive View running"))
      .catch((reason) => active && setError(reason instanceof Error ? reason.message : String(reason)));
    call.resultPromise.then(
      (toolResult) => active && setResult(toolResult),
      (reason) => active && setError(reason instanceof Error ? reason.message : String(reason)),
    );
    return () => { active = false; };
  }, [call]);

  function startCall(server = serverRef.current, nextQuestion = question) {
    if (!server) return;
    setError("");
    setResult(undefined);
    setEvents([]);
    setStatus("Invoking view_ask…");
    try {
      setCall(callViewAsk(server, nextQuestion));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  const links = result ? [...collectUrls(result)] : [];

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">MODE 2 · MCP APPS</p>
          <h1>Genie One interactive View</h1>
          <p className="subtitle">
            Test host based on the MCP Apps client spec — the panel below is Genie One’s View, not a custom chart.
          </p>
        </div>
        <span className="status">{status}</span>
      </header>

      <details className="overview">
        <summary>Test app overview — MCP Apps spec client, not a custom Genie One UI</summary>
        <div className="overviewBody">
          <p>
            This is a <strong>local test app</strong>. It implements an{" "}
            <strong>MCP Apps client</strong> against the Genie One MCP Service. The
            Interactive View panel is <strong>Genie One’s own MCP App</strong>, not a
            custom React chart we built.
          </p>
          <p>
            What comes out of the box from Genie One MCP (MCP Apps spec):
          </p>
          <ol>
            <li>
              Advertise <code>io.modelcontextprotocol/ui</code> on MCP{" "}
              <code>initialize</code>.
            </li>
            <li>
              The server offers <code>view_ask</code> with{" "}
              <code>_meta.ui.resourceUri</code> (<code>ui://…</code>).
            </li>
            <li>
              The host <code>resources/read</code>s that HTML (
              <code>text/html;profile=mcp-app</code>).
            </li>
            <li>
              Official <code>AppBridge</code> + <code>PostMessageTransport</code>{" "}
              load it in a sandboxed iframe and pass tool input/results.
            </li>
            <li>
              The View itself draws progress, answer, citations, “Explore in
              Genie One”, and polls <code>view_poll_response</code>.
            </li>
          </ol>
          <p>
            This harness only adds the question box, request traces, source-link
            list, and a local U2M proxy so the token never enters the browser.
          </p>
          <p className="overviewLinks">
            Specs:{" "}
            <a href="https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp" target="_blank" rel="noreferrer">
              Genie One MCP
            </a>
            {" · "}
            <a href="https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx" target="_blank" rel="noreferrer">
              MCP Apps specification
            </a>
          </p>
        </div>
      </details>

      <section className="ask">
        <label htmlFor="question">Question</label>
        <div className="askRow">
          <input
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button onClick={() => startCall()} disabled={!serverRef.current || !question.trim()}>
            Ask Genie One
          </button>
        </div>
      </section>

      {error && <div role="alert" className="error"><strong>Error:</strong> {error}</div>}

      <section className="view">
        <div className="sectionTitle">
          <h2>Interactive View</h2>
          <span>Genie One’s MCP App · ui:// resource in sandbox</span>
        </div>
        {call ? <iframe ref={iframeRef} title="Genie One MCP App" /> : <div className="placeholder">Waiting for view_ask…</div>}
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
          {events.length > 0 && <details><summary>AppBridge events</summary><pre>{events.join("\n")}</pre></details>}
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
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
