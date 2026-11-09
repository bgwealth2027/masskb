"use strict";

(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.MasskbReplay = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var STRUCTURAL_RELATIONS = ["HAS_CHUNK", "NEXT_CHUNK", "MENTIONS"];

  var MESSAGES = {
    empty: "请输入或选择一个预置问题。",
    noMatch: "没有匹配的预置问题。原型不会生成答案，也不会点亮节点。",
    playing: "正在回放示例关联。电流只沿样例中已有的有向结构边移动。",
    paused: "已暂停。已经点亮的节点和已经走过的边会保留。",
    done: "回放结束。证据子图仍然保留。这里没有生成回答。",
    reduced: "已用静态证据子图代替电流回放。点亮结果与播放结束时相同。"
  };

  var FUTURE_INTERFACE = [
    "这个原型的样例和回放都在浏览器里，没有请求 MassKB。",
    "现有 POST /search 返回扁平命中，可以附带前后分块 id，打开图扩展时还可以带 graph_via。那不是完整子图，不能冒充 nodes、edges、evidence、seed、expanded 或 path。",
    "以后接入真实知识库时，需要在调用方的可见范围里单独提供 nodes、edges、evidence、seed、expanded 和 path。visibility 说明这次能看见哪些文档和节点。",
    "回放只能沿这些数据里真实存在的有向结构边。HAS_CHUNK、NEXT_CHUNK、MENTIONS 表示包含、顺序和提及，不是因果。本样例不使用 ALIAS_OF。"
  ];

  var RELATION_LABEL = {
    HAS_CHUNK: "包含分块",
    NEXT_CHUNK: "后一分块",
    MENTIONS: "提及"
  };

  function normalizeQuery(value) {
    return String(value == null ? "" : value)
      .normalize("NFKC")
      .trim()
      .replace(/\s+/g, "")
      .replace(/[?？!！。．.]+$/g, "")
      .toLowerCase();
  }

  function matchPreset(graph, query) {
    var normalized = normalizeQuery(query);
    if (!normalized) {
      return { status: "empty", preset: null, answer: null, message: MESSAGES.empty };
    }
    var presets = graph && graph.presets ? graph.presets : [];
    var preset = null;
    for (var i = 0; i < presets.length; i += 1) {
      var phrases = presets[i].matches || [];
      for (var j = 0; j < phrases.length; j += 1) {
        if (normalizeQuery(phrases[j]) === normalized) {
          preset = presets[i];
          break;
        }
      }
      if (preset) break;
    }
    if (!preset) {
      return { status: "no-match", preset: null, answer: null, message: MESSAGES.noMatch };
    }
    return { status: "match", preset: preset, answer: null, message: null };
  }

  function structuralEdges(graph) {
    var nodeIds = new Set((graph.nodes || []).map(function (node) { return node.id; }));
    return (graph.edges || []).filter(function (edge) {
      return STRUCTURAL_RELATIONS.indexOf(edge.relation) !== -1 &&
        nodeIds.has(edge.source) &&
        nodeIds.has(edge.target);
    });
  }

  function buildPlayback(graph, preset) {
    var nodeIds = new Set((graph.nodes || []).map(function (node) { return node.id; }));
    var outgoing = new Map();
    nodeIds.forEach(function (id) { outgoing.set(id, []); });
    structuralEdges(graph).forEach(function (edge) {
      outgoing.get(edge.source).push(edge);
    });
    outgoing.forEach(function (list) {
      list.sort(function (a, b) {
        var rank = STRUCTURAL_RELATIONS.indexOf(a.relation) - STRUCTURAL_RELATIONS.indexOf(b.relation);
        if (rank !== 0) return rank;
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
      });
    });

    var seeds = (preset.seeds || []).filter(function (id) { return nodeIds.has(id); });
    var maxDepth = Number.isFinite(preset.maxDepth) ? preset.maxDepth : 1;
    var lit = new Set(seeds);
    var parents = {};
    var steps = [];
    seeds.forEach(function (id) {
      parents[id] = null;
      steps.push({ kind: "seed", nodeId: id });
    });

    var usedEdges = new Set();
    var evidenceEdgeIds = [];
    var frontier = seeds.slice();
    var depth;
    for (depth = 0; depth < maxDepth; depth += 1) {
      var next = [];
      var seenNext = new Set();
      frontier.forEach(function (sourceId) {
        var branch = 0;
        (outgoing.get(sourceId) || []).forEach(function (edge) {
          if (usedEdges.has(edge.id)) return;
          usedEdges.add(edge.id);
          evidenceEdgeIds.push(edge.id);
          steps.push({
            kind: "edge",
            edgeId: edge.id,
            source: edge.source,
            target: edge.target,
            relation: edge.relation,
            branch: branch
          });
          branch += 1;
          if (!lit.has(edge.target)) {
            lit.add(edge.target);
            parents[edge.target] = { from: edge.source, edgeId: edge.id };
            if (!seenNext.has(edge.target)) {
              seenNext.add(edge.target);
              next.push(edge.target);
            }
          }
        });
      });
      frontier = next;
    }

    var evidenceNodeIds = [];
    var appearance = [];
    var seenAppearance = new Set();
    steps.forEach(function (step) {
      var id = step.kind === "seed" ? step.nodeId : step.target;
      if (!seenAppearance.has(id)) {
        seenAppearance.add(id);
        appearance.push(id);
        evidenceNodeIds.push(id);
      }
    });

    var seedSet = new Set(seeds);
    var paths = {};
    evidenceNodeIds.forEach(function (id) {
      paths[id] = walkPath(parents, id);
    });

    return {
      presetId: preset.id,
      seeds: seeds.slice(),
      expanded: evidenceNodeIds.filter(function (id) { return !seedSet.has(id); }),
      steps: steps,
      evidenceNodeIds: evidenceNodeIds,
      evidenceEdgeIds: evidenceEdgeIds,
      appearance: appearance,
      parents: parents,
      paths: paths,
      answer: null
    };
  }

  function walkPath(parents, nodeId) {
    var reversed = [];
    var cursor = nodeId;
    var guard = new Set();
    while (cursor) {
      if (guard.has(cursor)) return null;
      guard.add(cursor);
      var parent = parents[cursor];
      reversed.push({ nodeId: cursor, viaEdgeId: parent ? parent.edgeId : null });
      cursor = parent ? parent.from : null;
    }
    reversed.reverse();
    return reversed;
  }

  function pathTo(playback, nodeId) {
    if (!playback || !playback.paths || !Object.prototype.hasOwnProperty.call(playback.paths, nodeId)) {
      return null;
    }
    return playback.paths[nodeId].map(function (step) {
      return { nodeId: step.nodeId, viaEdgeId: step.viaEdgeId };
    });
  }

  function sameMembers(left, right) {
    if (left.length !== right.length) return false;
    var seen = new Set(left);
    return right.every(function (item) { return seen.has(item); });
  }

  function runToEnd(playback) {
    var lit = [];
    var edges = [];
    var litSeen = new Set();
    var edgeSeen = new Set();
    playback.steps.forEach(function (step) {
      if (step.kind === "seed") {
        if (!litSeen.has(step.nodeId)) {
          litSeen.add(step.nodeId);
          lit.push(step.nodeId);
        }
        return;
      }
      if (!edgeSeen.has(step.edgeId)) {
        edgeSeen.add(step.edgeId);
        edges.push(step.edgeId);
      }
      if (!litSeen.has(step.target)) {
        litSeen.add(step.target);
        lit.push(step.target);
      }
    });
    return {
      lit: lit,
      edges: edges,
      activeEdgeId: null,
      answer: null
    };
  }

  function staticEquivalent(playback) {
    var animated = runToEnd(playback);
    var finished = frameAt(playback, scheduleOf(playback).duration);
    return {
      lit: playback.evidenceNodeIds.slice(),
      edges: playback.evidenceEdgeIds.slice(),
      activeEdgeId: null,
      answer: null,
      sameNodes: sameMembers(animated.lit, playback.evidenceNodeIds) &&
        sameMembers(finished.lit, playback.evidenceNodeIds),
      sameEdges: sameMembers(animated.edges, playback.evidenceEdgeIds) &&
        sameMembers(finished.retainedEdgeIds, playback.evidenceEdgeIds)
    };
  }

  var SEED_BEAT_MS = 400;
  var TRANSIT_MS = 900;

  function scheduleOf(playback) {
    var events = [];
    var cursor = 0;
    (playback.steps || []).forEach(function (step) {
      if (step.kind === "seed") {
        events.push({ kind: "seed", at: cursor, nodeId: step.nodeId });
        cursor += SEED_BEAT_MS;
        return;
      }
      events.push({
        kind: "edge",
        at: cursor,
        until: cursor + TRANSIT_MS,
        edgeId: step.edgeId,
        source: step.source,
        target: step.target,
        relation: step.relation
      });
      cursor += TRANSIT_MS;
    });
    return { events: events, duration: cursor, seedBeatMs: SEED_BEAT_MS, transitMs: TRANSIT_MS };
  }

  function frameAt(playback, elapsed) {
    var schedule = scheduleOf(playback);
    var time = Math.max(0, Number(elapsed) || 0);
    var lit = [];
    var seen = new Set();
    var retained = [];
    var active = null;
    schedule.events.forEach(function (event) {
      if (event.kind === "seed") {
        if (time >= event.at && !seen.has(event.nodeId)) {
          seen.add(event.nodeId);
          lit.push(event.nodeId);
        }
        return;
      }
      if (time >= event.until) {
        retained.push(event.edgeId);
        if (!seen.has(event.target)) {
          seen.add(event.target);
          lit.push(event.target);
        }
        return;
      }
      if (time >= event.at) {
        active = {
          edgeId: event.edgeId,
          source: event.source,
          target: event.target,
          relation: event.relation,
          progress: (time - event.at) / (event.until - event.at)
        };
      }
    });
    var done = time >= schedule.duration;
    return {
      lit: lit,
      retainedEdgeIds: retained,
      active: done ? null : active,
      done: done,
      elapsed: Math.min(time, schedule.duration),
      duration: schedule.duration
    };
  }

  function createPlayer(playback, now) {
    var elapsedBase = 0;
    var runningSince = null;
    var phase = "idle";
    function clock() {
      return now();
    }
    function playingTime() {
      if (phase === "playing" && runningSince != null) return elapsedBase + (clock() - runningSince);
      return elapsedBase;
    }
    function frame() {
      var view = frameAt(playback, playingTime());
      if (view.done && phase === "playing") {
        elapsedBase = view.duration;
        runningSince = null;
        phase = "done";
      }
      view.phase = phase;
      return view;
    }
    return {
      play: function () {
        if (phase === "done" || frame().done) {
          phase = "done";
          return;
        }
        if (phase !== "playing") {
          runningSince = clock();
          phase = "playing";
        }
      },
      pause: function () {
        if (phase !== "playing") return;
        elapsedBase = playingTime();
        runningSince = null;
        phase = "paused";
      },
      frame: frame
    };
  }

  return {
    STRUCTURAL_RELATIONS: STRUCTURAL_RELATIONS,
    RELATION_LABEL: RELATION_LABEL,
    MESSAGES: MESSAGES,
    FUTURE_INTERFACE: FUTURE_INTERFACE,
    normalizeQuery: normalizeQuery,
    matchPreset: matchPreset,
    buildPlayback: buildPlayback,
    pathTo: pathTo,
    runToEnd: runToEnd,
    staticEquivalent: staticEquivalent,
    SEED_BEAT_MS: SEED_BEAT_MS,
    TRANSIT_MS: TRANSIT_MS,
    scheduleOf: scheduleOf,
    frameAt: frameAt,
    createPlayer: createPlayer
  };
});
