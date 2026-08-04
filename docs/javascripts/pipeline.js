(function () {
  const ICON_HOOK_PREFIX = "pipeline-icon-";
  const COLOR_HOOK_PREFIX = "pipeline-color-";

  // ---------- parsing ----------

  function parsePipeline(text) {
    const nodes = new Map();
    const edges = [];

    // Full block form:
    // A[
    // Label text
    // ]
    // ---
    // link: ...
    // tooltip: ...
    // icon: ...
    // color: ...
    // type: ...
    // ---
    const blockRegex = /(\w+)\[(.*?)\]\s*\n---\s*\n([\s\S]*?)\n---/gs;
    const directionMatch = text.match(/^direction:\s*(LR|TB)\s*$/m);
    const direction = directionMatch ? directionMatch[1] : "LR";
    let match;
    while ((match = blockRegex.exec(text)) !== null) {
      const [, id, labelRaw, yamlRaw] = match;
      let meta = {};
      try {
        meta = window.jsyaml ? window.jsyaml.load(yamlRaw) || {} : {};
      } catch (e) {
        console.error(`pipeline.js: failed to parse YAML for node "${id}"`, e);
      }
      nodes.set(id, {
        id,
        label: labelRaw.trim(),
        link: meta.link || null,
        tooltip: meta.tooltip || null,
        icon: meta.icon || null,
        color: meta.color || null,
        type: meta.type || "process",
      });
    }

    // Shorthand form: A[Label]: optional/link
    const simpleRegex = /^(\w+)\[(.+?)\](?:\s*:\s*(.+))?$/gm;
    let simpleMatch;
    while ((simpleMatch = simpleRegex.exec(text)) !== null) {
      const [, id, label, link] = simpleMatch;
      if (!nodes.has(id)) {
        nodes.set(id, {
          id,
          label: label.trim(),
          link: link ? link.trim() : null,
          tooltip: null,
          icon: null,
          color: null,
          type: "process",
        });
      }
    }

    // Edges: A --> B
    const edgeRegex = /^(\w+)\s*-->\s*(\w+)$/gm;
    let edgeMatch;
    while ((edgeMatch = edgeRegex.exec(text)) !== null) {
      edges.push({ from: edgeMatch[1], to: edgeMatch[2] });
    }

    return { nodes, edges, direction };
  }

  // ---------- layout ----------

  function computeLevels(nodes, edges) {
    const incoming = new Map();
    nodes.forEach((_, id) => incoming.set(id, []));
    edges.forEach((e) => {
      if (incoming.has(e.to)) incoming.get(e.to).push(e.from);
    });

    const level = new Map();
    function getLevel(id, seen = new Set()) {
      if (level.has(id)) return level.get(id);
      if (seen.has(id)) return 0;
      seen.add(id);
      const preds = incoming.get(id) || [];
      const lvl = preds.length
        ? Math.max(...preds.map((p) => getLevel(p, seen))) + 1
        : 0;
      level.set(id, lvl);
      return lvl;
    }
    nodes.forEach((_, id) => getLevel(id));
    return level;
  }

  function layout(nodes, edges, direction) {
    const level = computeLevels(nodes, edges);
    const byLevel = new Map();

    nodes.forEach((node, id) => {
      const lvl = level.get(id);
      if (!byLevel.has(lvl)) byLevel.set(lvl, []);
      byLevel.get(lvl).push(node);
    });

    const colWidth = 260;
    const rowHeight = 100;

    const maxCount = Math.max(...[...byLevel.values()].map(a => a.length));

    let width, height;

    if (direction === "TB") {
      width = maxCount * colWidth + 40;
      height = byLevel.size * rowHeight + 40;

      byLevel.forEach((arr, lvl) => {
        const startX = (width - arr.length * colWidth) / 2;

        arr.forEach((node, i) => {
          node.w = Math.max(140, node.label.length * 8 + 40);
          node.h = 60;

          // LEFT EDGE
          node.x = startX + i * colWidth + (colWidth - node.w) / 2;
          // CENTER
          node.y = 40 + lvl * rowHeight + rowHeight / 2;
        });
      });
    }
    else {
      width = byLevel.size * colWidth + 40;
      height = maxCount * rowHeight + 40;

      byLevel.forEach((arr, lvl) => {
        const startY = (height - arr.length * rowHeight) / 2;

        arr.forEach((node, i) => {
          node.w = Math.max(140, node.label.length * 8 + 40);
          node.h = 60;

          node.x = 40 + lvl * colWidth;
          node.y = startY + i * rowHeight + rowHeight / 2;
        });
      });
    }

    return { width, height };
  }

  // ---------- shape rendering per type ----------

  function shapeForType(ns, node) {
    const { x, y, w, h, type } = node;
    const top = y - h / 2;

    if (type === "decision") {
      const poly = document.createElementNS(ns, "polygon");
      const points = [
        [x + w / 2, top],
        [x + w, y],
        [x + w / 2, top + h],
        [x, y],
      ]
        .map((p) => p.join(","))
        .join(" ");
      poly.setAttribute("points", points);
      return poly;
    }

    if (type === "io") {
      const skew = 18;
      const poly = document.createElementNS(ns, "polygon");
      const points = [
        [x + skew, top],
        [x + w, top],
        [x + w - skew, top + h],
        [x, top + h],
      ]
        .map((p) => p.join(","))
        .join(" ");
      poly.setAttribute("points", points);
      return poly;
    }

    if (type === "terminal") {
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", x);
      rect.setAttribute("y", top);
      rect.setAttribute("width", w);
      rect.setAttribute("height", h);
      rect.setAttribute("rx", h / 2);
      return rect;
    }

    // default: "process"
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", top);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("rx", 12);
    return rect;
  }

  // ---------- tooltip ----------

  let tooltipEl = null;

  function getTooltipEl() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "pipeline-tooltip";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }

  function showTooltip(text, x, y) {
    const el = getTooltipEl();
    el.textContent = text;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.classList.add("pipeline-tooltip--visible");
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.classList.remove("pipeline-tooltip--visible");
  }

  // ---------- svg build ----------

  function renderSVG(nodes, edges, direction) {
    const { width, height } = layout(nodes, edges, direction);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "pipeline-svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("xmlns", ns);

    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML = `
      <marker id="pipeline-arrow" markerWidth="10" markerHeight="10"
              refX="8" refY="3" orient="auto">
        <path d="M0,0 L0,6 L8,3 z" fill="currentColor"/>
      </marker>`;
    svg.appendChild(defs);

    edges.forEach(({ from, to }) => {
      const a = nodes.get(from);
      const b = nodes.get(to);
      if (!a || !b) return;

      let sx, sy, tx, ty;
      let path;

      if (direction === "TB") {
        sx = a.x + a.w / 2;
        sy = a.y + a.h / 2;

        tx = b.x + b.w / 2;
        ty = b.y - b.h / 2;

        const midY = (sy + ty) / 2;

        path =
          `M${sx},${sy}
          C${sx},${midY}
            ${tx},${midY}
            ${tx},${ty}`;
      } 
      else {
        sx = a.x + a.w;
        sy = a.y;

        tx = b.x;
        ty = b.y;

        const midX = (sx + tx) / 2;

        path =
          `M${sx},${sy}
          C${midX},${sy}
            ${midX},${ty}
            ${tx},${ty}`;
      }

      const pathEl = document.createElementNS(ns, "path");
      pathEl.setAttribute("d", path);
      pathEl.setAttribute("class", "pipeline-arrow");
      pathEl.setAttribute("marker-end", "url(#pipeline-arrow)");
      svg.appendChild(pathEl);
    });

    nodes.forEach((node) => {
      const wrapper = node.link
        ? document.createElementNS(ns, "a")
        : document.createElementNS(ns, "g");
      if (node.link) wrapper.setAttribute("href", node.link);

      let className = "pipeline-node";
      if (node.color) className += ` ${COLOR_HOOK_PREFIX}${node.color}`;
      if (node.icon) className += ` ${ICON_HOOK_PREFIX}${node.icon}`;
      wrapper.setAttribute("class", className);

      if (node.tooltip) {
        wrapper.addEventListener("mouseenter", (e) => {
          const rect = wrapper.getBoundingClientRect();
          showTooltip(node.tooltip, rect.left + rect.width / 2, rect.top + window.scrollY);
        });
        wrapper.addEventListener("mousemove", (e) => {
          const rect = wrapper.getBoundingClientRect();
          showTooltip(node.tooltip, rect.left + rect.width / 2, rect.top + window.scrollY);
        });
        wrapper.addEventListener("mouseleave", hideTooltip);
      }

      wrapper.appendChild(shapeForType(ns, node));

      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", node.x + node.w / 2);
      text.setAttribute("y", node.y);
      text.textContent = node.label;
      wrapper.appendChild(text);

      svg.appendChild(wrapper);
    });

    return svg;
  }

  // ---------- mount ----------

  function renderAll() {
    document.querySelectorAll(".pipeline-diagram").forEach((div) => {
      if (div.dataset.pipelineRendered) return;
      const { nodes, edges, direction } = parsePipeline(div.textContent);
      const svg = renderSVG(nodes, edges, direction);
      div.replaceWith(svg);
    });
  }

  if (window.document$) {
    document$.subscribe(renderAll);
  } else {
    document.addEventListener("DOMContentLoaded", renderAll);
  }
})();