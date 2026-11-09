import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const outputPath = join(root, "standalone.html");

const files = [
  "vendor/force-graph/force-graph.min.js",
  "sample/graph.js",
  "src/replay.js",
  "src/app.js"
];

function escapeScript(source) {
  return source.replace(/<\/(script)/gi, "<\\/$1");
}

export function renderStandalone() {
  let html = readFileSync(join(root, "index.html"), "utf8");
  html = html.replace(/\n?  <script src="[^"]+"><\/script>/g, "");
  const source = files
    .map((file) => escapeScript(readFileSync(join(root, file), "utf8")))
    .join("\n;\n");
  const block =
    "\n  <script>\n" +
    source +
    "\n  </script>\n";
  if (!html.includes("</body>")) {
    throw new Error("index.html 缺少 </body>");
  }
  html = html.replace("</body>", () => block + "</body>");
  if (html.includes('<script src="')) {
    throw new Error("standalone 仍引用外部脚本");
  }
  return html;
}

const html = renderStandalone();
const check = process.argv.includes("--check");
if (check) {
  const current = readFileSync(outputPath, "utf8");
  if (current !== html) {
    console.error("standalone.html 与构建结果不一致。请重新运行 build-standalone.mjs。");
    process.exit(1);
  }
} else {
  writeFileSync(outputPath, html);
  console.log(outputPath);
}
