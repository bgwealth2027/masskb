"use strict";

(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.MasskbSample = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var nodes = [
    {
      id: "doc-library",
      kind: "document",
      component: "library",
      label: "读者指南",
      title: "市立图书馆读者指南",
      text: "虚构的读者指南，用来把同一份文档拆成前后相连的分块。图里不保存这段正文，正文只出现在这个示例面板。",
      source: "示例文档 · 市立图书馆读者指南（虚构）",
      homeX: -150,
      homeY: 0
    },
    {
      id: "chunk-lib-hours",
      kind: "chunk",
      component: "library",
      label: "开放时间",
      title: "开放时间",
      text: "市立图书馆每周二到周日开放，时间是 9:00 至 17:00。每月第一个周六定为开放日，当天闭馆时间延长到 20:00。周一闭馆，用来整理书架。",
      source: "示例文档 · 市立图书馆读者指南（虚构）",
      homeX: -60,
      homeY: -20
    },
    {
      id: "chunk-lib-children",
      kind: "chunk",
      component: "library",
      label: "儿童阅览",
      title: "儿童阅览室",
      text: "儿童阅览室在二楼东侧。周末故事会由馆员林夏负责，周六 10:30 开始。故事会使用馆内绘本，不另外收费。",
      source: "示例文档 · 市立图书馆读者指南（虚构）",
      homeX: 30,
      homeY: 0
    },
    {
      id: "chunk-lib-theme",
      kind: "chunk",
      component: "library",
      label: "主题陈列",
      title: "主题陈列",
      text: "开放日下午，一楼大厅换成主题书架，这一期的主题是本地植物。架上只放馆里已有的书，不对外售卖。",
      source: "示例文档 · 市立图书馆读者指南（虚构）",
      homeX: 120,
      homeY: 10
    },
    {
      id: "person-linxia",
      kind: "person",
      component: "library",
      label: "林夏",
      title: "林夏",
      text: "虚构的示例人物，没有真实身份、账号或联系方式。读者指南只写明周末故事会由林夏负责。",
      source: "示例文档 · 市立图书馆读者指南（虚构）",
      homeX: 20,
      homeY: 70
    },
    {
      id: "concept-open-day",
      kind: "concept",
      component: "library",
      label: "开放日",
      title: "开放日",
      text: "示例概念，指每月第一个周六延长开放。分块可以提及它。提及不是因果，也不表示开放日导致了其他事情。",
      source: "示例概念 · 由读者指南和活动月报提及（虚构）",
      homeX: -70,
      homeY: -85
    },
    {
      id: "concept-story",
      kind: "concept",
      component: "library",
      label: "故事会",
      title: "故事会",
      text: "示例概念，指儿童阅览室的周末故事会。它只记录被哪一块正文提及。",
      source: "示例概念 · 由读者指南提及（虚构）",
      homeX: 55,
      homeY: 78
    },
    {
      id: "concept-shelf",
      kind: "concept",
      component: "library",
      label: "主题书架",
      title: "主题书架",
      text: "示例概念，指开放日在一楼摆出的主题书架。架上的书来自馆藏示例，不是付费内容。",
      source: "示例概念 · 由读者指南提及（虚构）",
      homeX: 130,
      homeY: 78
    },
    {
      id: "doc-event",
      kind: "document",
      component: "event",
      label: "活动月报",
      title: "社区活动月报",
      text: "虚构的社区活动月报。它和读者指南只通过“开放日”这个概念上的提及相连，两篇文档的分块之间没有直接连边。",
      source: "示例文档 · 社区活动月报（虚构）",
      homeX: 210,
      homeY: -70
    },
    {
      id: "chunk-evt-market",
      kind: "chunk",
      component: "event",
      label: "广场市集",
      title: "广场市集",
      text: "社区市集和图书馆开放日同一天，地点在图书馆前广场，下午 14:00 开始。摊位展示手作和园艺苗，现场不收款。",
      source: "示例文档 · 社区活动月报（虚构）",
      homeX: 290,
      homeY: -50
    },
    {
      id: "chunk-evt-signup",
      kind: "chunk",
      component: "event",
      label: "现场报名",
      title: "现场报名",
      text: "参加广场市集的邻居当天现场报名。登记写在纸质表上，不需要账号，也不需要付费。",
      source: "示例文档 · 社区活动月报（虚构）",
      homeX: 370,
      homeY: -30
    },
    {
      id: "concept-market",
      kind: "concept",
      component: "event",
      label: "市集",
      title: "市集",
      text: "示例概念，指图书馆前广场的社区市集。它不连接浇水备忘。",
      source: "示例概念 · 由活动月报提及（虚构）",
      homeX: 290,
      homeY: -115
    },
    {
      id: "concept-paper",
      kind: "concept",
      component: "event",
      label: "纸质报名",
      title: "纸质报名",
      text: "示例概念，指当天用纸质表登记。样例里没有账号、密钥或支付信息。",
      source: "示例概念 · 由活动月报提及（虚构）",
      homeX: 370,
      homeY: -95
    },
    {
      id: "doc-garden",
      kind: "document",
      component: "garden",
      label: "园艺备忘",
      title: "社区园艺备忘",
      text: "虚构的园艺备忘。它和图书馆、市集都没有结构边，用来演示独立命中。",
      source: "示例文档 · 社区园艺备忘（虚构）",
      homeX: 190,
      homeY: 95
    },
    {
      id: "chunk-garden-water",
      kind: "chunk",
      component: "garden",
      label: "浇水时间",
      title: "浇水时间",
      text: "楼前共享花箱在清晨或傍晚浇水。夏季隔日一次，春秋每周两次。雨天跳过，正午不浇。",
      source: "示例文档 · 社区园艺备忘（虚构）",
      homeX: 280,
      homeY: 115
    },
    {
      id: "chunk-garden-tools",
      kind: "chunk",
      component: "garden",
      label: "工具归位",
      title: "工具归位",
      text: "浇水用水桶接雨水，水壶放在花箱旁的木箱里。用完放回原处，不收押金。",
      source: "示例文档 · 社区园艺备忘（虚构）",
      homeX: 360,
      homeY: 135
    },
    {
      id: "concept-water",
      kind: "concept",
      component: "garden",
      label: "浇水",
      title: "浇水",
      text: "示例概念，指照料共享花箱。它只连着园艺备忘，不连图书馆或市集。",
      source: "示例概念 · 由园艺备忘提及（虚构）",
      homeX: 280,
      homeY: 175
    }
  ];

  var edges = [
    { id: "e-lib-doc-hours", relation: "HAS_CHUNK", source: "doc-library", target: "chunk-lib-hours" },
    { id: "e-lib-doc-children", relation: "HAS_CHUNK", source: "doc-library", target: "chunk-lib-children" },
    { id: "e-lib-doc-theme", relation: "HAS_CHUNK", source: "doc-library", target: "chunk-lib-theme" },
    { id: "e-lib-hours-children", relation: "NEXT_CHUNK", source: "chunk-lib-hours", target: "chunk-lib-children" },
    { id: "e-lib-children-theme", relation: "NEXT_CHUNK", source: "chunk-lib-children", target: "chunk-lib-theme" },
    { id: "e-lib-hours-open", relation: "MENTIONS", source: "chunk-lib-hours", target: "concept-open-day" },
    { id: "e-lib-theme-open", relation: "MENTIONS", source: "chunk-lib-theme", target: "concept-open-day" },
    { id: "e-lib-theme-shelf", relation: "MENTIONS", source: "chunk-lib-theme", target: "concept-shelf" },
    { id: "e-lib-children-linxia", relation: "MENTIONS", source: "chunk-lib-children", target: "person-linxia" },
    { id: "e-lib-children-story", relation: "MENTIONS", source: "chunk-lib-children", target: "concept-story" },
    { id: "e-evt-doc-market", relation: "HAS_CHUNK", source: "doc-event", target: "chunk-evt-market" },
    { id: "e-evt-doc-signup", relation: "HAS_CHUNK", source: "doc-event", target: "chunk-evt-signup" },
    { id: "e-evt-market-signup", relation: "NEXT_CHUNK", source: "chunk-evt-market", target: "chunk-evt-signup" },
    { id: "e-evt-market-open", relation: "MENTIONS", source: "chunk-evt-market", target: "concept-open-day" },
    { id: "e-evt-market-market", relation: "MENTIONS", source: "chunk-evt-market", target: "concept-market" },
    { id: "e-evt-signup-paper", relation: "MENTIONS", source: "chunk-evt-signup", target: "concept-paper" },
    { id: "e-gar-doc-water", relation: "HAS_CHUNK", source: "doc-garden", target: "chunk-garden-water" },
    { id: "e-gar-doc-tools", relation: "HAS_CHUNK", source: "doc-garden", target: "chunk-garden-tools" },
    { id: "e-gar-water-tools", relation: "NEXT_CHUNK", source: "chunk-garden-water", target: "chunk-garden-tools" },
    { id: "e-gar-water-concept", relation: "MENTIONS", source: "chunk-garden-water", target: "concept-water" },
    { id: "e-gar-tools-concept", relation: "MENTIONS", source: "chunk-garden-tools", target: "concept-water" }
  ];

  var presets = [
    {
      id: "library-hours",
      title: "图书馆开放日几点开始？",
      matches: ["图书馆开放日几点开始？", "图书馆开放日几点开始"],
      seeds: ["chunk-lib-hours"],
      maxDepth: 2,
      blurb: "种子是“开放时间”。电流沿后一分块和提及分叉，不反向走上市集。"
    },
    {
      id: "children-room",
      title: "谁负责儿童阅览室？",
      matches: ["谁负责儿童阅览室？", "谁负责儿童阅览室", "儿童阅览室谁负责"],
      seeds: ["chunk-lib-children"],
      maxDepth: 1,
      blurb: "种子是“儿童阅览”。从这里分出后一分块、馆员和故事会三条结构边。"
    },
    {
      id: "library-doc",
      title: "读者指南包含哪些分块？",
      matches: ["读者指南包含哪些分块？", "读者指南包含哪些分块"],
      seeds: ["doc-library"],
      maxDepth: 1,
      blurb: "种子是读者指南。电流沿包含分块分到三个分块，表示文档结构，不是因果。"
    },
    {
      id: "garden-water",
      title: "社区园艺什么时候浇水？",
      matches: ["社区园艺什么时候浇水？", "社区园艺什么时候浇水"],
      seeds: ["chunk-garden-water"],
      maxDepth: 1,
      blurb: "种子是“浇水时间”。证据只留在园艺备忘里，不连到图书馆。"
    },
    {
      id: "split-hits",
      title: "开放日时间和浇水时间",
      matches: ["开放日时间和浇水时间", "开放日和浇水分别是什么时候"],
      seeds: ["chunk-lib-hours", "chunk-garden-water"],
      maxDepth: 1,
      blurb: "两个种子各自点亮。它们之间没有样例边，回放不会添一条强连边。"
    }
  ];

  return {
    banner: "示例数据/交互原型，未接真实知识库",
    replayNote: "示例关联回放，非后端实时推理",
    noMatchExample: {
      title: "无匹配示例",
      query: "量子纠缠的实验装置型号"
    },
    nodes: nodes,
    edges: edges,
    presets: presets
  };
});
