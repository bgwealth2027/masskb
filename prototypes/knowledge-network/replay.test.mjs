import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const sample = require("./sample/graph.js");
const replay = require("./src/replay.js");

function preset(id) {
  return sample.presets.find((item) => item.id === id);
}

function playback(id) {
  return replay.buildPlayback(sample, preset(id));
}

function edge(id) {
  return sample.edges.find((item) => item.id === id);
}

test("样例只使用三种结构边，并且端点都存在", () => {
  const ids = new Set(sample.nodes.map((node) => node.id));
  assert.equal(new Set(sample.nodes.map((node) => node.id)).size, sample.nodes.length);
  assert.equal(new Set(sample.edges.map((item) => item.id)).size, sample.edges.length);
  assert.ok(sample.edges.length >= 8);
  for (const item of sample.edges) {
    assert.ok(replay.STRUCTURAL_RELATIONS.includes(item.relation));
    assert.ok(ids.has(item.source));
    assert.ok(ids.has(item.target));
    assert.notEqual(item.relation, "ALIAS_OF");
  }
  assert.doesNotMatch(JSON.stringify(sample), /ALIAS_OF|sk-|api_key|BEGIN [A-Z ]*PRIVATE/i);
});

test("预置问题按完整问法匹配，对不上就不生成答案", () => {
  const hit = replay.matchPreset(sample, "  图书馆开放日几点开始？ ");
  assert.equal(hit.status, "match");
  assert.equal(hit.preset.id, "library-hours");
  assert.equal(hit.answer, null);
  assert.equal(replay.matchPreset(sample, "图书馆开放日几点开始").preset.id, "library-hours");
  assert.equal(replay.matchPreset(sample, "儿童阅览室谁负责").preset.id, "children-room");

  for (const query of ["", "   ", "图书馆", "开放日", "请问图书馆开放日几点开始", sample.noMatchExample.query]) {
    const missed = replay.matchPreset(sample, query);
    assert.notEqual(missed.status, "match");
    assert.equal(missed.preset, null);
    assert.equal(missed.answer, null);
    assert.equal(missed.message, query.trim() ? replay.MESSAGES.noMatch : replay.MESSAGES.empty);
  }
});

test("回放只沿样例里真实存在的有向边，分支会分开", () => {
  const hours = playback("library-hours");
  const known = new Set(sample.edges.map((item) => item.id));
  assert.equal(hours.answer, null);
  assert.deepEqual(hours.seeds, ["chunk-lib-hours"]);
  for (const id of hours.evidenceEdgeIds) assert.ok(known.has(id));
  assert.ok(!hours.evidenceNodeIds.includes("chunk-evt-market"));
  assert.ok(!hours.evidenceEdgeIds.includes("e-evt-market-open"));
  assert.ok(hours.evidenceNodeIds.includes("concept-open-day"));
  assert.ok(hours.evidenceNodeIds.includes("person-linxia"));

  const lit = new Set();
  for (const step of hours.steps) {
    if (step.kind === "seed") {
      lit.add(step.nodeId);
      continue;
    }
    const item = edge(step.edgeId);
    assert.equal(item.source, step.source);
    assert.equal(item.target, step.target);
    assert.ok(lit.has(step.source));
    lit.add(step.target);
  }

  const branches = hours.steps.filter((step) => step.kind === "edge" && step.source === "chunk-lib-hours");
  assert.ok(branches.length >= 2);
  assert.ok(new Set(branches.map((step) => step.target)).size >= 2);

  const children = playback("children-room");
  const childEdges = children.steps.filter((step) => step.kind === "edge" && step.source === "chunk-lib-children");
  assert.equal(childEdges.length, 3);

  const doc = playback("library-doc");
  assert.ok(doc.evidenceEdgeIds.length >= 3);
  for (const id of doc.evidenceEdgeIds) assert.equal(edge(id).relation, "HAS_CHUNK");
});

test("独立命中之间不添加连边", () => {
  const split = playback("split-hits");
  assert.deepEqual(split.seeds, ["chunk-lib-hours", "chunk-garden-water"]);
  assert.ok(split.evidenceNodeIds.includes("chunk-lib-hours"));
  assert.ok(split.evidenceNodeIds.includes("chunk-garden-water"));
  const component = new Map(sample.nodes.map((node) => [node.id, node.component]));
  for (const id of split.evidenceEdgeIds) {
    const item = edge(id);
    assert.equal(component.get(item.source), component.get(item.target));
  }
  const garden = playback("garden-water");
  for (const id of garden.evidenceNodeIds) {
    assert.equal(component.get(id), "garden");
  }
});

test("减少动画的静态结果与播放结束相同", () => {
  for (const item of sample.presets) {
    const plan = replay.buildPlayback(sample, item);
    const frozen = replay.staticEquivalent(plan);
    assert.equal(frozen.sameNodes, true);
    assert.equal(frozen.sameEdges, true);
    assert.equal(frozen.activeEdgeId, null);
    assert.equal(frozen.answer, null);
    const path = replay.pathTo(plan, plan.expanded[0] || plan.seeds[0]);
    assert.equal(path[0].nodeId, plan.seeds[0]);
    assert.equal(path.at(-1).nodeId, plan.expanded[0] || plan.seeds[0]);
  }
  assert.equal(replay.pathTo(playback("garden-water"), "doc-library"), null);
});

test("适配层忽略 ALIAS_OF，不把它当成结构边", () => {
  const graph = {
    nodes: [{ id: "a" }, { id: "b" }, { id: "c" }],
    edges: [
      { id: "next", relation: "NEXT_CHUNK", source: "a", target: "b" },
      { id: "alias", relation: "ALIAS_OF", source: "b", target: "c" },
      { id: "mention", relation: "MENTIONS", source: "a", target: "c" }
    ]
  };
  const plan = replay.buildPlayback(graph, { id: "alias-check", seeds: ["a"], maxDepth: 2 });
  assert.deepEqual(plan.evidenceEdgeIds, ["next", "mention"]);
  assert.ok(plan.evidenceNodeIds.includes("c"));
  assert.equal(plan.answer, null);
});

test("页面用相对路径，项目站 /masskb/ 下仍指向原型自己的文件", () => {
  const html = readFileSync(join(root, "index.html"), "utf8");
  const sources = [...html.matchAll(/<(?:script|link)\b[^>]*\b(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(sources.includes("vendor/force-graph/force-graph.min.js"));
  assert.ok(sources.includes("sample/graph.js"));
  assert.ok(sources.includes("src/replay.js"));
  assert.ok(sources.includes("src/app.js"));
  for (const source of sources) {
    assert.equal(source.startsWith("/"), false);
    assert.equal(source.includes("://"), false);
    const hosted = new URL(source, "https://bgwealth2027.github.io/masskb/");
    assert.equal(hosted.pathname.startsWith("/masskb/"), true);
    const rooted = new URL(source, "https://example.test/");
    assert.equal(rooted.pathname, "/" + source);
  }
  assert.match(html, /示例数据\/交互原型，未接真实知识库|id="banner"/);
  assert.doesNotMatch(html, /sigma|react|unpkg|jsdelivr/i);
  assert.match(readFileSync(join(root, "src/replay.js"), "utf8"), /POST \/search/);
  for (const word of ["nodes", "edges", "evidence", "seed", "expanded", "path", "visibility"]) {
    assert.match(replay.FUTURE_INTERFACE.join("\n"), new RegExp(word));
  }
});

test("Pages 工作流只能手动触发，并且只准备原型产物", () => {
  const workflow = readFileSync(join(root, "../../.github/workflows/deploy-pages.yml"), "utf8");
  const head = workflow.split(/^jobs:/m)[0];
  assert.match(head, /workflow_dispatch/);
  assert.doesNotMatch(head, /^\s*push:/m);
  assert.doesNotMatch(head, /^\s*pull_request:/m);
  assert.doesNotMatch(head, /^\s*schedule:/m);
  assert.match(workflow, /prototypes\/knowledge-network/);
  assert.match(workflow, /path: _pages/);
  assert.match(workflow, /docker-compose\.yml/);
  assert.doesNotMatch(workflow, /masskb serve/);
});

test("自包含页面内联库和数据，且与构建结果一致", async () => {
  const { spawnSync } = await import("node:child_process");
  const result = spawnSync(process.execPath, ["build-standalone.mjs", "--check"], { cwd: root, encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const standalone = readFileSync(join(root, "standalone.html"), "utf8");
  assert.match(standalone, /Version 1\.52\.0 force-graph/);
  assert.match(standalone, /chunk-lib-hours/);
  assert.match(standalone, /示例关联回放，非后端实时推理/);
  assert.doesNotMatch(standalone, /<script src=/);
  assert.equal((standalone.match(/<!DOCTYPE/g) || []).length, 1);
  assert.equal((standalone.match(/<\/html>/g) || []).length, 1);
  const pin = readFileSync(join(root, "vendor/force-graph/PIN.txt"), "utf8");
  assert.match(pin, /version: 1\.52\.0/);
  assert.match(readFileSync(join(root, "vendor/force-graph/LICENSE"), "utf8"), /MIT License/);
});
