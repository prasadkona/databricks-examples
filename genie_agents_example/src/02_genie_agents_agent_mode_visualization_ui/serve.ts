import express from "express";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const directory = join(dirname(fileURLToPath(import.meta.url)), "dist");
const port = 8080;
const app = express();

app.use((_request, response, next) => {
  response.setHeader("Cache-Control", "no-store");
  next();
});
app.use(express.static(directory));
app.get("/", (_request, response) => response.redirect("/index.html"));
app.listen(port, "127.0.0.1", () => {
  console.log(`UI: http://localhost:${port}`);
});
