/* 小林面试笔记 RAG 问答 —— 前端逻辑
 *
 * 上半部分是纯函数（转义与 markdown 渲染），导出给测试用；
 * 下半部分操作 DOM，用 typeof document 守卫，使 node 可以直接 import 本模块。
 */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

/** 所有后端文本的唯一出口：先转义再拼进 HTML。 */
export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ESCAPES[ch]);
}

/** 同一段内的多行：中英交界处补空格，中文之间直接相接。 */
function joinLines(lines) {
  return lines.reduce((acc, line) => {
    if (!acc) return line;
    const prev = acc.charAt(acc.length - 1);
    const next = line.charAt(0);
    const needsSpace = /[A-Za-z0-9]/.test(prev) && /[A-Za-z0-9]/.test(next);
    return acc + (needsSpace ? " " : "") + line;
  }, "");
}

/**
 * 行内标记。用单次交替正则一次扫完：已替换出来的标签不会被再次扫描，
 * 因此不需要占位符，也不会出现「粗体里再套粗体」之类的互相污染。
 *
 * 顺序即优先级，两条最要紧：
 *   1. 行内代码排最前，代码里的 * 与 [n] 不被后续规则误伤；
 *   2. 图片与链接必须排在引用编号之前，否则 [1](url) 会先被编号分支吃掉。
 * 另外 ** 必须排在 * 之前，~~ 也必须排在 * 之前。
 */
const INLINE = new RegExp(
  [
    "`([^`]+)`", // 1 行内代码
    "!\\[([^\\]]*)\\]\\(([^)]+)\\)", // 2,3 图片 alt / src
    // 链接地址允许一层嵌套括号，否则 javascript:alert(1) 之类会只吃到第一个 )，
    // 把剩下的 ) 留在正文里
    "\\[([^\\]]+)\\]\\(((?:[^()\\s]|\\([^()]*\\))+)\\)", // 4,5 链接文字 / 地址
    "\\*\\*([^*]+)\\*\\*", // 6 粗体
    "~~([^~]+)~~", // 7 删除线
    "\\*([^\\s*][^*]*?)\\*", // 8 斜体
    "\\[(\\d{1,3})\\]", // 9 引用编号
  ].join("|"),
  "g",
);

const IMAGE_PREFIX = "assets/";

/**
 * 引用原文里的图片地址。
 *
 * 语料里的引用长这样：assets/<文章>/<文件>，是相对**文章所在目录**的。站点的目录
 * 层级不固定（两层站点是 <站点>/<栏目>/<文章>，深层站点还要多一层章节），所以基准
 * 取文章标识去掉最后一段，而不是固定切几刀。
 *
 * returns 可加载的地址；相对路径不是 assets/ 开头、或拿不到文章标识时返回 null（降级成文本）。
 */
export function citationImageUrl(pageId, src) {
  if (!src || !pageId || !src.startsWith(IMAGE_PREFIX)) return null;
  const cut = pageId.lastIndexOf("/");
  if (cut <= 0) return null;
  return `/kb/${pageId.slice(0, cut)}/${src}`;
}

/** 图片：交给调用方的解析器决定能不能加载，拿不到地址就降级成文本。 */
function renderImage(alt, src, options) {
  const name = alt || src.split("/").pop();
  const url = options.resolveImage ? options.resolveImage(src) : null;
  if (!url) return `[图片：${name}]`;
  return `<img src="${url}" alt="${name}" loading="lazy" decoding="async">`;
}

/**
 * 链接。只有解析成 http(s) 才渲染成链接，否则只留文字——原样回显
 * [文字](url) 那种「降级」等于把源码摆给用户看，不是渲染。
 * resolveLink 由调用方提供，用来把站内相对路径（/ai/rag/1.html）补成完整地址。
 */
function renderLink(text, href, options) {
  // href 已在整体转义时变成了 &amp; 等形式，解析前先还原
  const raw = href.replace(/&amp;/g, "&");
  const target = (options.resolveLink ? options.resolveLink(raw) : raw) || "";
  if (!/^https?:\/\//i.test(target)) return text || raw;
  return `<a href="${target}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}

function renderInline(lines, options) {
  return joinLines(lines).replace(
    INLINE,
    (match, code, imgAlt, imgSrc, linkText, linkHref, bold, strike, italic, cite) => {
      if (code !== undefined) return `<code>${code}</code>`;
      if (imgAlt !== undefined) return renderImage(imgAlt, imgSrc, options);
      if (linkText !== undefined) return renderLink(linkText, linkHref, options);
      if (bold !== undefined) return `<strong>${bold}</strong>`;
      if (strike !== undefined) return `<del>${strike}</del>`;
      if (italic !== undefined) return `<em>${italic}</em>`;
      return `<a class="cite" href="#cite-${cite}">[${cite}]</a>`;
    },
  );
}

/** 缩进宽度：制表符按 4 列算，缩进决定列表层数。 */
function indentWidth(whitespace) {
  return whitespace.replace(/\t/g, "    ").length;
}

/** 表格：要求两端都有竖线，且紧跟一行分隔行——只凭竖线会把正文里的单个 | 误判。 */
const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_DIVIDER = /^\s*\|[\s:|-]+\|\s*$/;

function splitRow(line) {
  let text = line.trim();
  if (text.startsWith("|")) text = text.slice(1);
  if (text.endsWith("|")) text = text.slice(0, -1);
  return text.split("|").map((cell) => cell.trim());
}

function parseAlign(cell) {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  if (left) return "left";
  return "";
}

function renderTable(header, align, rows, options) {
  const cell = (tag, text, column) => {
    const style = align[column] ? ` style="text-align:${align[column]}"` : "";
    return `<${tag}${style}>${renderInline([text], options)}</${tag}>`;
  };
  const head = `<thead><tr>${header.map((c, n) => cell("th", c, n)).join("")}</tr></thead>`;
  const body = rows.length
    ? `<tbody>${rows
        .map((row) => `<tr>${row.map((c, n) => cell("td", c, n)).join("")}</tr>`)
        .join("")}</tbody>`
    : "";
  return `<div class="table-wrap"><table>${head}${body}</table></div>`;
}

/**
 * 列表。entries 为 [{ ordered, indent, text }]，按缩进相对大小决定嵌套：
 * 比当前层深就开一层，浅就关到对应层，同级换 ul↔ol 就另开一层。
 */
function renderList(entries, options) {
  const stack = [];
  let html = "";

  for (const entry of entries) {
    const tag = entry.ordered ? "ol" : "ul";

    while (stack.length && entry.indent < stack[stack.length - 1].indent) {
      html += `</li></${stack.pop().tag}>`;
    }

    const top = stack[stack.length - 1];
    if (!top || entry.indent > top.indent) {
      html += `<${tag}><li>`;
      stack.push({ tag, indent: entry.indent });
    } else if (top.tag !== tag) {
      html += `</li></${stack.pop().tag}>`;
      html += `<${tag}><li>`;
      stack.push({ tag, indent: entry.indent });
    } else {
      html += "</li><li>";
    }
    html += renderInline([entry.text], options);
  }

  while (stack.length) {
    html += `</li></${stack.pop().tag}>`;
  }
  return html;
}

/**
 * markdown 子集渲染：标题、列表、表格、围栏代码块、引用块、段落。
 * 不支持的语法原样降级为纯文本，不抛异常。
 * 输入先整体转义，之后只做标记替换——顺序反了就等于开了 XSS 口子。
 *
 * options.resolveImage(src) 由调用方提供，返回可加载的地址或 null。
 * 渲染器因此不必知道知识库的目录结构，保持纯粹可测。
 */
export function renderMarkdown(markdown, options = {}) {
  const lines = escapeHtml(markdown).split(/\r?\n/);
  const out = [];
  let paragraph = [];
  let list = null;
  let quote = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${renderInline(paragraph, options)}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list && list.length) {
      out.push(renderList(list, options));
      list = null;
    }
  };
  const flushQuote = () => {
    if (quote.length) {
      out.push(`<blockquote>${renderInline(quote, options)}</blockquote>`);
      quote = [];
    }
  };
  const flushAll = () => {
    flushParagraph();
    flushList();
    flushQuote();
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flushAll();
      const body = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      out.push(`<pre><code>${body.join("\n")}</code></pre>`);
      continue;
    }

    if (!line.trim()) {
      flushAll();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 1, 6); // 一级标题留给页面，正文从 h2 起
      out.push(`<h${level}>${renderInline([heading[2]], options)}</h${level}>`);
      continue;
    }

    if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_DIVIDER.test(lines[i + 1])) {
      flushAll();
      const header = splitRow(line);
      const align = splitRow(lines[i + 1]).map(parseAlign);
      const rows = [];
      i += 2;
      while (i < lines.length && TABLE_ROW.test(lines[i])) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      i -= 1; // 回退一格，让循环末尾的 i += 1 落在表格后的第一行
      out.push(renderTable(header, align, rows, options));
      continue;
    }

    const bullet = line.match(/^(\s*)[-*+]\s+(.*)$/);
    const numbered = line.match(/^(\s*)\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      flushQuote();
      const matched = bullet || numbered;
      if (!list) list = [];
      list.push({
        ordered: Boolean(numbered),
        indent: indentWidth(matched[1]),
        text: matched[2],
      });
      continue;
    }

    // 输入已整体转义，引用块的 > 此时是 &gt;，两种写法都接
    const quoted = line.match(/^(?:&gt;|>)\s?(.*)$/);
    if (quoted) {
      flushParagraph();
      flushList();
      quote.push(quoted[1]);
      continue;
    }

    flushList();
    flushQuote();
    paragraph.push(line);
  }
  flushAll();

  return out.join("\n");
}

/* ---------- 状态与区域渲染 ---------- */

export const state = { meta: null, busy: false, result: null, debugOn: false };

const $ = (id) => document.getElementById(id);

function show(element, visible) {
  element.hidden = !visible;
}

function renderMeta(meta) {
  $("meta-line").textContent =
    `${meta.page_count} 篇笔记 · ${meta.child_count} 个子块 · 构建于 ${meta.built_at.slice(0, 10)}`;

  // 用 DOM 构造而非拼字符串：站点与栏目名来自后端，无需再操心转义
  const groups = (meta.sites || []).map((site) => {
    const group = document.createElement("div");
    group.className = "chip-group";

    const head = document.createElement("label");
    head.className = "chip-group-head";
    const all = document.createElement("input");
    all.type = "checkbox";
    const siteName = document.createElement("span");
    siteName.textContent = site.label;
    head.append(all, siteName);

    const chips = document.createElement("div");
    chips.className = "chips";
    const boxes = site.sections.map((section) => {
      const chip = document.createElement("label");
      chip.className = "chip";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = section.key;
      const text = document.createElement("span");
      text.textContent = section.label;
      chip.append(input, text);
      chips.append(chip);
      return input;
    });

    // 站点级复选框是「全选该站点」的快捷方式，勾选状态由组内栏目反推
    all.addEventListener("change", () => {
      for (const box of boxes) box.checked = all.checked;
      syncGroupState(all, boxes);
    });
    for (const box of boxes) {
      box.addEventListener("change", () => syncGroupState(all, boxes));
    }

    group.append(head, chips);
    return group;
  });

  $("sections").replaceChildren(...groups);
}

function syncGroupState(toggle, boxes) {
  const checked = boxes.filter((box) => box.checked).length;
  toggle.checked = boxes.length > 0 && checked === boxes.length;
  toggle.indeterminate = checked > 0 && checked < boxes.length;
}

function renderAnswer(result) {
  const callout = $("answer-callout");
  const prose = $("answer-prose");

  if (result.rejected) {
    const title = document.createElement("p");
    title.className = "callout-title";
    title.textContent = "未作答";
    const body = document.createElement("p");
    body.textContent = result.answer;
    callout.replaceChildren(title, body);
    prose.replaceChildren();
    show(callout, true);
    show(prose, false);
    return;
  }

  // 唯一一处 innerHTML：renderMarkdown 的输入已整体转义，输出只含它自己生成的标签
  prose.innerHTML = renderMarkdown(result.answer);
  callout.replaceChildren();
  show(callout, false);
  show(prose, true);
}

function buildCitationCard(citation) {
  const item = document.createElement("li");
  item.className = "citation-card";
  item.id = `cite-${citation.index}`;

  const head = document.createElement("div");
  head.className = "citation-head";
  const index = document.createElement("span");
  index.className = "citation-index";
  index.textContent = `[${citation.index}]`;
  const title = document.createElement("span");
  title.className = "citation-title";
  title.textContent = citation.title;
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = citation.section_label;
  head.append(index, title, badge);
  item.append(head);

  if (citation.heading_path) {
    const path = document.createElement("p");
    path.className = "citation-path";
    path.textContent = citation.heading_path;
    item.append(path);
  }

  // 只放行 http(s)：来源地址来自语料 frontmatter，不能让别的协议进 href
  if (/^https?:\/\//i.test(citation.source_url)) {
    const link = document.createElement("a");
    link.className = "citation-link";
    link.href = citation.source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = citation.source_url;
    item.append(link);
  }

  // 引用原文与答案共用同一条渲染链路。用 div 而不是 p：渲染后可能含 table/ul 这类块级元素。
  const excerpt = document.createElement("div");
  excerpt.className = "excerpt clamped";
  excerpt.innerHTML = renderMarkdown(citation.excerpt, {
    // 语料里的图片相对文章所在目录，用它作基准拼出可加载的地址
    resolveImage: (src) => citationImageUrl(citation.page_id, src),
    // 站内相对链接（/ai/rag/1.html）以引用自身的出处地址为基准补成完整 URL
    resolveLink: (href) => {
      try {
        const url = new URL(href, citation.source_url);
        return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
      } catch {
        return null;
      }
    },
  });
  item.append(excerpt);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "expand";
  toggle.textContent = "展开全文";
  toggle.addEventListener("click", () => {
    const collapsed = excerpt.classList.toggle("clamped");
    toggle.textContent = collapsed ? "展开全文" : "收起";
  });
  item.append(toggle);

  return item;
}

function renderCitations(citations) {
  const section = $("citations-section");
  if (!citations.length) {
    $("citations-list").replaceChildren();
    show(section, false);
    return;
  }
  $("citations-list").replaceChildren(...citations.map(buildCitationCard));
  show(section, true);
}

function renderDebug(debug) {
  const section = $("debug-section");
  // 只有用户开了开关才展示：此时 debug 里才有逐条分数
  if (!state.debugOn || !debug || !Object.keys(debug).length) {
    $("debug-output").textContent = "";
    show(section, false);
    return;
  }
  $("debug-output").textContent = JSON.stringify(debug, null, 2);
  show(section, true);
}

function renderResult(result) {
  renderAnswer(result);
  renderCitations(result.citations);
  renderDebug(result.debug);
}

/* ---------- 事件编排 ---------- */

const LOADING_TEXT = "正在检索并生成答案…";

function syncSubmit() {
  const button = $("submit");
  button.disabled = state.busy || !$("question").value.trim();
  button.textContent = state.busy ? "提问中…" : "提问";
}

function setBusy(busy) {
  state.busy = busy;
  const status = $("status");
  status.textContent = busy ? LOADING_TEXT : "";
  show(status, busy);
  syncSubmit();
}

/** 空态提示不该与「正在检索」同时出现，一提交就撤掉它。 */
function clearPlaceholder() {
  const prose = $("answer-prose");
  if (prose.querySelector(".placeholder")) prose.replaceChildren();
}

function showError(message) {
  const box = $("form-error");
  box.textContent = message || "";
  show(box, Boolean(message));
}

function selectedSections() {
  return [...$("sections").querySelectorAll("input:checked")].map((input) => input.value);
}

async function submit() {
  if (state.busy) return;
  const question = $("question").value.trim();
  if (!question) return;

  showError("");
  clearPlaceholder();
  setBusy(true);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        sections: selectedSections(),
        debug: state.debugOn,
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload && typeof payload.detail === "string" ? payload.detail : "";
      showError(detail || `请求失败（HTTP ${response.status}）`);
      return;
    }
    state.result = payload;
    renderResult(payload);
  } catch {
    // 网络层失败：行内提示，不弹窗
    showError("无法连接到服务，请确认 xiaolinrag serve 仍在运行。");
  } finally {
    setBusy(false);
  }
}

async function init() {
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.meta = await response.json();
    renderMeta(state.meta);
  } catch {
    showError("读取知识库元信息失败，请确认服务已启动且索引已构建。");
  }
  setBusy(false);
}

if (typeof document !== "undefined") {
  const question = $("question");
  const debugToggle = $("debug-toggle");

  question.addEventListener("input", syncSubmit);
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });
  debugToggle.addEventListener("change", () => {
    state.debugOn = debugToggle.checked;
  });

  // 图片加载失败时换成可读文本，不留裂图。error 事件不冒泡，必须在捕获阶段接。
  $("citations-list").addEventListener(
    "error",
    (event) => {
      const img = event.target;
      if (!img || img.tagName !== "IMG") return;
      const fallback = document.createElement("span");
      fallback.className = "image-fallback";
      fallback.textContent = `[图片加载失败：${img.alt || "未命名"}]`;
      img.replaceWith(fallback);
    },
    true,
  );

  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  init();
}
