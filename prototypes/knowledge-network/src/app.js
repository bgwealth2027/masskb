"use strict";

(function () {
  var sample = window.MasskbSample;
  var replay = window.MasskbReplay;
  var banner = document.getElementById("banner");
  var replayNote = document.getElementById("replay-note");
  var statusEl = document.getElementById("status");
  var blurbEl = document.getElementById("blurb");
  var form = document.getElementById("query-form");
  var input = document.getElementById("query");
  var presetsEl = document.getElementById("presets");
  var hintEl = document.getElementById("match-hint");
  var resultsEl = document.getElementById("results");
  var detailEl = document.getElementById("detail");
  var futureEl = document.getElementById("future");
  var side = document.getElementById("side");
  var playBtn = document.getElementById("play");
  var pauseBtn = document.getElementById("pause");
  var replayBtn = document.getElementById("replay");
  var clearBtn = document.getElementById("clear");
  var fitBtn = document.getElementById("fit");
  var reduceBtn = document.getElementById("reduce");
  var toggleBtn = document.getElementById("toggle-results");
  var mount = document.getElementById("graph");

  if (!sample || !replay || typeof window.ForceGraph !== "function") {
    if (statusEl) {
      statusEl.textContent = "图组件没有载入。请打开本目录下的页面；离线双击请使用 standalone.html。";
    }
    return;
  }

  banner.textContent = sample.banner;
  replayNote.textContent = sample.replayNote;
  hintEl.textContent = "只接受上面这些完整问法。忽略首尾空白和末尾问号，不按词义搜索。对不上就不匹配，也不生成答案。";
  futureEl.replaceChildren();
  replay.FUTURE_INTERFACE.forEach(function (line) {
    var p = document.createElement("p");
    p.textContent = line;
    futureEl.append(p);
  });

  var nodesById = new Map(sample.nodes.map(function (node) { return [node.id, node]; }));
  var edgesById = new Map(sample.edges.map(function (edge) { return [edge.id, edge]; }));
  var kindLabel = { document: "文档", chunk: "分块", person: "人物", concept: "概念" };

  var state = {
    reduced: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    phase: "idle",
    playback: null,
    preset: null,
    lit: new Set(),
    roles: new Map(),
    evidenceEdges: new Set(),
    activeEdgeId: null,
    selectedId: null,
    stepIndex: 0,
    timer: null,
    userAdjusted: false,
    fitting: false,
    acceptUserZoom: false
  };

  var graph = window.ForceGraph()(mount)
    .backgroundColor("#141618")
    .nodeId("id")
    .nodeRelSize(5)
    .nodeLabel(function (node) { return node.label + " · " + kindLabel[node.kind]; })
    .linkColor(colorForLink)
    .linkWidth(widthForLink)
    .linkDirectionalArrowLength(5)
    .linkDirectionalArrowRelPos(0.72)
    .linkDirectionalArrowColor(colorForLink)
    .linkDirectionalParticles(particlesForLink)
    .linkDirectionalParticleWidth(2.2)
    .linkDirectionalParticleSpeed(0.02)
    .linkDirectionalParticleColor(function () { return "#ffd8b0"; })
    .linkCanvasObjectMode(function () { return "after"; })
    .linkCanvasObject(drawHalo)
    .nodeCanvasObjectMode(function () { return "replace"; })
    .nodeCanvasObject(drawNode)
    .nodePointerAreaPaint(paintHit)
    .minZoom(0.35)
    .maxZoom(6)
    .cooldownTicks(state.reduced ? 40 : 170)
    .d3AlphaDecay(state.reduced ? 0.16 : 0.035)
    .d3VelocityDecay(0.35)
    .onNodeClick(function (node) { select(node.id, true); })
    .onNodeDrag(function () { state.userAdjusted = true; })
    .onNodeDragEnd(function (node) {
      node.fx = node.x;
      node.fy = node.y;
      state.userAdjusted = true;
    })
    .onZoom(function () {
      if (state.acceptUserZoom && !state.fitting) state.userAdjusted = true;
    })
    .onEngineStop(function () { fitGraph(false, false); });

  graph.graphData({
    nodes: sample.nodes.map(function (node) {
      return {
        id: node.id,
        label: node.label,
        kind: node.kind,
        homeX: node.homeX,
        homeY: node.homeY,
        x: node.homeX,
        y: node.homeY
      };
    }),
    links: sample.edges.map(function (edge) {
      return {
        id: edge.id,
        relation: edge.relation,
        source: edge.source,
        target: edge.target
      };
    })
  });

  var charge = graph.d3Force("charge");
  if (charge) charge.strength(-28);
  var center = graph.d3Force("center");
  if (center) center.strength(0.03);
  var linkForce = graph.d3Force("link");
  if (linkForce) {
    linkForce.distance(function (link) {
      if (link.relation === "NEXT_CHUNK") return 62;
      if (link.relation === "HAS_CHUNK") return 74;
      return 92;
    });
    linkForce.strength(function (link) {
      return link.relation === "MENTIONS" ? 0.12 : 0.55;
    });
  }
  graph.d3Force("cluster", clusterForce());
  graph.d3ReheatSimulation();
  sizeGraph();
  window.addEventListener("resize", sizeGraph);
  window.requestAnimationFrame(function () {
    sizeGraph();
    fitGraph(false, false);
  });
  window.setTimeout(function () { fitGraph(false, false); }, 900);
  window.setTimeout(function () { state.acceptUserZoom = true; }, 1400);

  sample.presets.forEach(function (preset) {
    presetsEl.append(chip(preset.title, function () {
      input.value = preset.title;
      submitQuery(preset.title);
    }));
  });
  presetsEl.append(chip(sample.noMatchExample.title, function () {
    input.value = sample.noMatchExample.query;
    submitQuery(sample.noMatchExample.query);
  }));

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuery(input.value);
  });
  playBtn.addEventListener("click", play);
  pauseBtn.addEventListener("click", pause);
  replayBtn.addEventListener("click", function () {
    if (!state.playback || !state.preset) return;
    begin(state.playback, state.preset);
  });
  clearBtn.addEventListener("click", clearPlayback);
  fitBtn.addEventListener("click", function () {
    fitGraph(true, true);
  });
  reduceBtn.addEventListener("click", function () {
    setReduced(!state.reduced);
  });
  toggleBtn.addEventListener("click", function () {
    var collapsed = side.classList.toggle("is-collapsed");
    toggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    toggleBtn.textContent = collapsed ? "展开结果" : "折叠结果";
    sizeGraph();
  });
  resultsEl.addEventListener("keydown", function (event) {
    moveList(resultsEl, event);
  });
  detailEl.addEventListener("keydown", function (event) {
    if (event.target.closest(".path")) moveList(event.target.closest(".path"), event);
  });
  window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", function (event) {
    setReduced(event.matches);
  });

  setReduced(state.reduced);
  showIdle();
  document.documentElement.dataset.ready = "1";

  function chip(text, onClick) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = text;
    button.addEventListener("click", onClick);
    return button;
  }

  function sizeGraph() {
    var rect = mount.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    state.fitting = true;
    graph.width(rect.width);
    graph.height(rect.height);
    state.fitting = false;
  }

  function fitGraph(animated, force) {
    if (state.userAdjusted && !force) return;
    state.fitting = true;
    var nodes = graph.graphData().nodes.filter(function (node) {
      return Number.isFinite(node.x) && Number.isFinite(node.y);
    });
    var rect = mount.getBoundingClientRect();
    if (nodes.length && rect.width > 40 && rect.height > 40) {
      var minX = Infinity;
      var maxX = -Infinity;
      var minY = Infinity;
      var maxY = -Infinity;
      nodes.forEach(function (node) {
        if (node.x < minX) minX = node.x;
        if (node.x > maxX) maxX = node.x;
        if (node.y < minY) minY = node.y;
        if (node.y > maxY) maxY = node.y;
      });
      minX -= 42;
      maxX += 42;
      minY -= 28;
      maxY += 46;
      var pad = 24;
      var k = Math.min(
        (rect.width - pad * 2) / Math.max(120, maxX - minX),
        (rect.height - pad * 2) / Math.max(120, maxY - minY)
      );
      k = Math.max(0.6, Math.min(2.6, k));
      graph.centerAt((minX + maxX) / 2, (minY + maxY) / 2, 0);
      graph.zoom(k, 0);
    } else {
      graph.zoomToFit(0, 36);
    }
    window.setTimeout(function () { state.fitting = false; }, animated ? 80 : 40);
  }

  function clusterForce() {
    var nodes = [];
    function force(alpha) {
      nodes.forEach(function (node) {
        if (!Number.isFinite(node.homeX) || node.fx != null) return;
        node.vx += (node.homeX - node.x) * 0.9 * alpha;
        node.vy += (node.homeY - node.y) * 0.9 * alpha;
      });
    }
    force.initialize = function (next) { nodes = next || []; };
    return force;
  }

  function colorForLink(link) {
    if (state.evidenceEdges.has(link.id)) return "rgba(226, 132, 68, 0.92)";
    return "rgba(158, 214, 222, 0.72)";
  }

  function widthForLink(link) {
    if (state.evidenceEdges.has(link.id)) return 1.7;
    return 1.25;
  }

  function particlesForLink() {
    return 0;
  }

  function drawHalo(link, ctx) {
    var source = link.source;
    var target = link.target;
    if (!source || !target || !Number.isFinite(source.x) || !Number.isFinite(target.x)) return;
    if (state.evidenceEdges.has(link.id)) {
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = "rgba(224, 122, 61, 0.22)";
      ctx.lineWidth = 5.5;
      ctx.stroke();
    }
    var active = state.frame && state.frame.active;
    if (!active || active.edgeId !== link.id || !(active.progress >= 0)) return;
    var x = source.x + (target.x - source.x) * active.progress;
    var y = source.y + (target.y - source.y) * active.progress;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "rgba(255, 186, 120, 0.96)";
    ctx.lineWidth = 2.8;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffe1c2";
    ctx.fill();
    var head = graph.graph2ScreenCoords(x, y);
    var from = graph.graph2ScreenCoords(source.x, source.y);
    var to = graph.graph2ScreenCoords(target.x, target.y);
    window.__knTransit = {
      edgeId: active.edgeId,
      progress: active.progress,
      litCount: state.lit.size,
      phase: state.phase,
      head: { x: head.x, y: head.y },
      source: { x: from.x, y: from.y },
      target: { x: to.x, y: to.y }
    };
  }

  function radiusFor(node) {
    if (node.kind === "document") return 11;
    if (node.kind === "chunk") return 8;
    return 7;
  }

  function drawNode(node, ctx, globalScale) {
    var role = state.roles.get(node.id);
    var selected = state.selectedId === node.id;
    var radius = radiusFor(node);
    if (role) {
      ctx.beginPath();
      ctx.fillStyle = role === "seed" ? "rgba(232, 137, 58, 0.28)" : "rgba(224, 122, 61, 0.16)";
      ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
      ctx.fill();
    }
    traceShape(ctx, node, radius);
    ctx.fillStyle = role === "seed" ? "#e8893a" : role === "arrived" ? "#c46a28" : "#2a2f34";
    ctx.fill();
    ctx.lineWidth = selected ? 2.4 : 1.35;
    ctx.strokeStyle = selected ? "#ffd7b0" : role ? "#ffb067" : "#9ec5cb";
    ctx.stroke();
    var fontSize = Math.max(14 / globalScale, 4);
    ctx.font = fontSize + 'px "WenQuanYi Micro Hei", "Noto Sans SC", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = role ? "#ffe7d0" : "rgba(226, 232, 234, 0.94)";
    ctx.fillText(node.label, node.x, node.y + radius + 3);
  }

  function traceShape(ctx, node, radius) {
    ctx.beginPath();
    if (node.kind === "document") {
      roundedRect(ctx, node.x - radius - 2, node.y - radius + 1, radius * 2 + 4, radius * 2 - 2, 3);
    } else if (node.kind === "concept") {
      roundedRect(ctx, node.x - radius, node.y - radius, radius * 2, radius * 2, 2);
    } else if (node.kind === "person") {
      ctx.moveTo(node.x, node.y - radius);
      ctx.lineTo(node.x + radius, node.y);
      ctx.lineTo(node.x, node.y + radius);
      ctx.lineTo(node.x - radius, node.y);
      ctx.closePath();
    } else {
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    }
  }

  function roundedRect(ctx, x, y, width, height, r) {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function paintHit(node, color, ctx) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radiusFor(node) + 6, 0, Math.PI * 2);
    ctx.fill();
  }

  function submitQuery(raw) {
    var matched = replay.matchPreset(sample, raw);
    markChips(raw);
    if (matched.status !== "match") {
      stopTimer();
      state.playback = null;
      state.preset = null;
      state.phase = matched.status === "empty" ? "empty" : "nomatch";
      state.stepIndex = 0;
      state.lit.clear();
      state.roles.clear();
      state.evidenceEdges.clear();
      state.activeEdgeId = null;
      state.selectedId = null;
      state.message = matched.message;
      resultsEl.replaceChildren();
      renderDetail();
      blurbEl.textContent = "";
      updateChrome();
      redraw();
      return;
    }
    begin(replay.buildPlayback(sample, matched.preset), matched.preset);
  }

  var player = null;
  var rafId = null;

  function stopLoop() {
    if (rafId != null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  function begin(playback, preset) {
    stopLoop();
    state.playback = playback;
    state.preset = preset;
    state.stepIndex = 0;
    state.lit = new Set();
    state.roles = new Map();
    state.evidenceEdges = new Set();
    state.activeEdgeId = null;
    state.progress = null;
    state.frame = null;
    state.selectedId = playback.seeds[0] || null;
    window.__knTransit = null;
    player = replay.createPlayer(playback, function () { return Date.now(); });
    graph.graphData().nodes.forEach(function (node) {
      if (Number.isFinite(node.x) && Number.isFinite(node.y)) {
        node.fx = node.x;
        node.fy = node.y;
      }
    });
    buildResults();
    renderDetail();
    blurbEl.textContent = preset.blurb || "";
    if (state.reduced) {
      showFinal();
      return;
    }
    player.play();
    state.phase = "playing";
    state.message = replay.MESSAGES.playing;
    updateChrome();
    loop();
  }

  function loop() {
    rafId = null;
    if (!player || state.phase !== "playing") return;
    applyFrame(player.frame());
    if (state.phase === "playing") rafId = window.requestAnimationFrame(loop);
  }

  function applyFrame(frame) {
    var playback = state.playback;
    if (!playback || !frame) return;
    state.frame = frame;
    state.lit = new Set(frame.lit);
    state.roles = new Map();
    var seeds = new Set(playback.seeds);
    frame.lit.forEach(function (id) {
      state.roles.set(id, seeds.has(id) ? "seed" : "arrived");
    });
    state.evidenceEdges = new Set(frame.retainedEdgeIds);
    state.activeEdgeId = frame.active ? frame.active.edgeId : null;
    state.progress = frame.active ? frame.active.progress : null;
    if (frame.done && state.phase === "playing") {
      state.phase = "done";
      state.message = replay.MESSAGES.done;
      state.activeEdgeId = null;
      state.progress = null;
      stopLoop();
    }
    if (!state.activeEdgeId) window.__knTransit = null;
    syncBadges();
    updateChrome();
    redraw();
  }

  function showFinal() {
    stopLoop();
    var playback = state.playback;
    var done = replay.frameAt(playback, replay.scheduleOf(playback).duration);
    state.phase = "done";
    applyFrame(done);
    state.phase = "done";
    state.message = replay.MESSAGES.reduced;
    state.activeEdgeId = null;
    state.progress = null;
    state.frame = done;
    window.__knTransit = null;
    updateChrome();
    redraw();
  }

  function play() {
    if (!state.playback || !player) return;
    if (state.reduced) {
      showFinal();
      return;
    }
    if (state.phase === "done" || player.frame().done) return;
    player.play();
    state.phase = "playing";
    state.message = replay.MESSAGES.playing;
    updateChrome();
    if (rafId == null) loop();
  }

  function pause() {
    if (!player || state.phase !== "playing") return;
    player.pause();
    state.phase = "paused";
    state.message = replay.MESSAGES.paused;
    stopLoop();
    applyFrame(player.frame());
  }

  function clearPlayback() {
    stopLoop();
    player = null;
    state.playback = null;
    state.preset = null;
    state.phase = "idle";
    state.stepIndex = 0;
    state.lit.clear();
    state.roles.clear();
    state.evidenceEdges.clear();
    state.activeEdgeId = null;
    state.progress = null;
    state.frame = null;
    state.selectedId = null;
    window.__knTransit = null;
    markChips("");
    showIdle();
    redraw();
  }

  function showIdle() {
    state.message = "选择一个预置问题后开始回放。清除只去掉点亮，问题会留在输入框。";
    blurbEl.textContent = "";
    resultsEl.replaceChildren();
    renderDetail();
    updateChrome();
  }

  function setReduced(next) {
    state.reduced = Boolean(next);
    reduceBtn.setAttribute("aria-pressed", state.reduced ? "true" : "false");
    reduceBtn.textContent = state.reduced ? "减少动画：开" : "减少动画：关";
    document.documentElement.dataset.reduced = state.reduced ? "1" : "0";
    if (state.reduced && state.playback && (state.phase === "playing" || state.phase === "paused")) {
      showFinal();
    } else {
      redraw();
    }
  }

  function buildResults() {
    var focusId = document.activeElement && document.activeElement.dataset
      ? document.activeElement.dataset.id
      : null;
    resultsEl.replaceChildren();
    state.playback.appearance.forEach(function (id) {
      var node = nodesById.get(id);
      var button = document.createElement("button");
      button.type = "button";
      button.className = "hit";
      button.dataset.id = id;
      button.dataset.state = "waiting";
      var label = document.createElement("span");
      label.className = "hit-label";
      label.textContent = node.title;
      var meta = document.createElement("span");
      meta.className = "hit-meta";
      var stateText = document.createElement("span");
      stateText.className = "hit-state";
      stateText.textContent = "等待";
      meta.append(kindLabel[node.kind] + " · ", stateText);
      var excerpt = document.createElement("span");
      excerpt.className = "hit-excerpt";
      excerpt.textContent = node.text.slice(0, 42) + (node.text.length > 42 ? "…" : "");
      button.append(label, meta, excerpt);
      button.addEventListener("click", function () { select(id, true); });
      var item = document.createElement("li");
      item.append(button);
      resultsEl.append(item);
    });
    syncBadges();
    if (focusId) {
      var again = resultsEl.querySelector('[data-id="' + cssEscape(focusId) + '"]');
      if (again) again.focus();
    }
  }

  function syncBadges() {
    resultsEl.querySelectorAll(".hit").forEach(function (button) {
      var role = state.roles.get(button.dataset.id);
      button.dataset.state = role || "waiting";
      var stateText = button.querySelector(".hit-state");
      stateText.textContent = role === "seed" ? "种子已点亮" : role === "arrived" ? "已抵达" : "等待";
      button.setAttribute("aria-current", button.dataset.id === state.selectedId ? "true" : "false");
    });
    var litCount = state.roles.size;
    var total = state.playback ? state.playback.evidenceNodeIds.length : 0;
    document.getElementById("results-title").textContent = state.playback
      ? "证据子图 " + litCount + " / " + total
      : "证据子图";
  }

  function select(id, center) {
    state.selectedId = id;
    syncBadges();
    renderDetail();
    redraw();
    if (!center) return;
    var node = graph.graphData().nodes.find(function (item) { return item.id === id; });
    if (!node || !Number.isFinite(node.x)) return;
    graph.centerAt(node.x, node.y, state.reduced ? 0 : 280);
  }

  function renderDetail() {
    detailEl.replaceChildren();
    var node = state.selectedId ? nodesById.get(state.selectedId) : null;
    if (!node) {
      var empty = document.createElement("p");
      empty.textContent = "选择图中的节点，或用键盘在结果列表和路径里移动。这里显示示例正文、来源、结构关系和路径。";
      detailEl.append(empty);
      return;
    }
    var title = document.createElement("h3");
    title.textContent = node.title;
    var meta = document.createElement("p");
    meta.textContent = kindLabel[node.kind] + roleSentence(node.id);
    detailEl.append(title, meta);
    appendBlock("示例正文", node.text);
    appendBlock("来源", node.source);
    appendRelations(node.id);
    appendPath(node.id);
    var note = document.createElement("p");
    note.className = "hint";
    note.textContent = "这些关系是包含、顺序或提及，不是因果。";
    detailEl.append(note);
  }

  function roleSentence(id) {
    if (!state.playback) return " · 尚未回放";
    var role = state.roles.get(id);
    if (role === "seed") return " · 本次种子";
    if (role === "arrived") return " · 沿结构边抵达";
    if (state.playback.evidenceNodeIds.indexOf(id) !== -1) return " · 在证据子图中，尚未点亮";
    return " · 不在本次证据子图";
  }

  function appendBlock(heading, text) {
    var h = document.createElement("h3");
    h.textContent = heading;
    var p = document.createElement("p");
    p.textContent = text;
    detailEl.append(h, p);
  }

  function appendRelations(nodeId) {
    var h = document.createElement("h3");
    h.textContent = "样例中的结构关系";
    var list = document.createElement("ul");
    sample.edges.forEach(function (edge) {
      if (edge.source !== nodeId && edge.target !== nodeId) return;
      var item = document.createElement("li");
      var otherId = edge.source === nodeId ? edge.target : edge.source;
      var other = nodesById.get(otherId);
      var direction = edge.source === nodeId ? "指出" : "指入";
      item.textContent = direction + " · " + replay.RELATION_LABEL[edge.relation] +
        " · " + other.title + "（" + edge.relation + "，非因果）";
      list.append(item);
    });
    if (!list.childElementCount) {
      var none = document.createElement("li");
      none.textContent = "样例里没有连到这个节点的结构边。";
      list.append(none);
    }
    detailEl.append(h, list);
  }

  function appendPath(nodeId) {
    var h = document.createElement("h3");
    h.id = "path-title";
    h.textContent = "本次路径";
    detailEl.append(h);
    if (!state.playback) {
      var waiting = document.createElement("p");
      waiting.textContent = "还没有回放。";
      detailEl.append(waiting);
      return;
    }
    var path = replay.pathTo(state.playback, nodeId);
    if (!path) {
      var outside = document.createElement("p");
      outside.className = "missing";
      outside.textContent = "这个节点不在本次证据子图里，没有从种子沿已有有向边走过来的路径。";
      detailEl.append(outside);
      return;
    }
    var line = document.createElement("div");
    line.className = "path";
    line.setAttribute("role", "list");
    line.setAttribute("aria-labelledby", "path-title");
    path.forEach(function (step, index) {
      if (step.viaEdgeId) {
        var edge = edgesById.get(step.viaEdgeId);
        var rel = document.createElement("span");
        rel.className = "rel";
        rel.textContent = "→ " + replay.RELATION_LABEL[edge.relation] + " →";
        line.append(rel);
      }
      var button = document.createElement("button");
      button.type = "button";
      button.className = "path-step";
      button.dataset.id = step.nodeId;
      button.setAttribute("role", "listitem");
      button.textContent = nodesById.get(step.nodeId).title;
      button.setAttribute("aria-current", step.nodeId === state.selectedId ? "true" : "false");
      button.addEventListener("click", function () { select(step.nodeId, true); });
      line.append(button);
      if (index === 0 && path.length === 1) {
        var only = document.createElement("span");
        only.className = "hint";
        only.textContent = "这是种子，路径不再向前。";
        line.append(only);
      }
    });
    detailEl.append(line);
  }

  function moveList(container, event) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "Home" && event.key !== "End") {
      return;
    }
    var buttons = Array.prototype.slice.call(container.querySelectorAll("button"));
    if (!buttons.length) return;
    event.preventDefault();
    var index = buttons.indexOf(document.activeElement);
    var next = 0;
    if (event.key === "ArrowDown") next = Math.min(buttons.length - 1, Math.max(0, index) + (index === -1 ? 0 : 1));
    if (event.key === "ArrowUp") next = Math.max(0, (index === -1 ? 0 : index) - 1);
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = buttons.length - 1;
    buttons[next].focus();
    select(buttons[next].dataset.id, true);
  }

  function updateChrome() {
    statusEl.textContent = state.message || "";
    playBtn.disabled = !state.playback || state.phase === "playing" || state.phase === "done";
    pauseBtn.disabled = state.phase !== "playing";
    replayBtn.disabled = !state.playback;
    clearBtn.disabled = state.phase === "idle";
    if (!state.playback) {
      document.getElementById("results-title").textContent = "证据子图";
    }
  }

  function markChips(raw) {
    var normalized = replay.normalizeQuery(raw);
    presetsEl.querySelectorAll(".chip").forEach(function (button) {
      var matched = replay.normalizeQuery(button.textContent) === normalized && normalized !== "";
      if (button.textContent === sample.noMatchExample.title) {
        matched = replay.normalizeQuery(sample.noMatchExample.query) === normalized;
      }
      button.setAttribute("aria-pressed", matched ? "true" : "false");
    });
  }

  function redraw() {
    var zoom = graph.zoom();
    if (typeof zoom === "number") graph.zoom(zoom, 0);
  }

  function cssEscape(value) {
    if (window.CSS && CSS.escape) return CSS.escape(value);
    return String(value).replace(/"/g, '\\"');
  }
})();
