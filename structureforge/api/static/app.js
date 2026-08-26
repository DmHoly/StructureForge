"use strict";

const state = {
  materials: [],
  materialColors: {},
  recipes: { deposition: [], etch: [] },
  steps: [],
  frames: [],
  frameIndex: 0,
  globalBounds: null, // {x0,x1,y0,y1} across every frame, for a stable overview scale
};

const $ = (id) => document.getElementById(id);

function optionsHtml(names) {
  return names.map((n) => `<option value="${n}">${n}</option>`).join("");
}

async function loadLibraries() {
  const [materials, recipes] = await Promise.all([
    fetch("/api/materials").then((r) => r.json()),
    fetch("/api/recipes").then((r) => r.json()),
  ]);
  state.materials = materials;
  state.recipes = recipes;
  for (const m of materials) state.materialColors[m.name] = m.color;

  const materialNames = materials.map((m) => m.name);
  for (const id of ["substrate-material", "dep-material", "pla-stop-material", "litho-material", "strip-material"]) {
    $(id).innerHTML = optionsHtml(materialNames);
  }
  $("litho-material").value = materialNames.includes("Photoresist") ? "Photoresist" : materialNames[0];
  $("strip-material").value = materialNames.includes("Photoresist") ? "Photoresist" : materialNames[0];

  $("dep-recipe").innerHTML = optionsHtml(recipes.deposition.map((r) => r.name));
  $("etch-recipe").innerHTML = optionsHtml(recipes.etch.map((r) => r.name));
}

function switchStepKindFields() {
  const kind = $("step-kind").value;
  for (const el of document.querySelectorAll(".step-fields")) {
    el.classList.toggle("active", el.id === `fields-${kind}`);
  }
}

function switchPlanarizationMode() {
  const mode = $("pla-mode").value;
  $("pla-stop-wrap").style.display = mode === "stop_material" ? "flex" : "none";
  $("pla-level-wrap").style.display = mode === "target_level" ? "flex" : "none";
}

function parseOpenings(text) {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((pair) => {
      const [a, b] = pair.split("-").map((v) => parseFloat(v.trim()));
      return [a, b];
    });
}

function buildStepFromForm() {
  const kind = $("step-kind").value;
  const name = $("step-name").value.trim() || kind;

  if (kind === "deposition") {
    return {
      kind, name,
      material: $("dep-material").value,
      recipe: $("dep-recipe").value,
      thickness: { value: parseFloat($("dep-thickness").value), unit: "nm" },
    };
  }
  if (kind === "etch") {
    return {
      kind, name,
      recipe: $("etch-recipe").value,
      depth: { value: parseFloat($("etch-depth").value), unit: "nm" },
    };
  }
  if (kind === "planarization") {
    const mode = $("pla-mode").value;
    const step = { kind, name };
    if (mode === "stop_material") step.stop_material = $("pla-stop-material").value;
    else step.target_level = { value: parseFloat($("pla-level").value), unit: "nm" };
    return step;
  }
  if (kind === "lithography") {
    return {
      kind, name,
      resist_material: $("litho-material").value,
      thickness: { value: parseFloat($("litho-thickness").value), unit: "nm" },
      openings: parseOpenings($("litho-openings").value),
    };
  }
  if (kind === "resist_strip") {
    return { kind, name, material: $("strip-material").value };
  }
  if (kind === "chemical") {
    return { kind, name, description: $("chem-description").value || null };
  }
  throw new Error(`unknown step kind ${kind}`);
}

function renderStepList() {
  const list = $("step-list");
  list.innerHTML = "";
  state.steps.forEach((step, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span><span class="step-kind">${step.kind}</span>${step.name}</span>`;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "Supprimer";
    removeBtn.addEventListener("click", () => {
      state.steps.splice(i, 1);
      renderStepList();
    });
    li.appendChild(removeBtn);
    list.appendChild(li);
  });
}

function showSimError(message) {
  const el = $("sim-error");
  if (!message) {
    el.hidden = true;
    el.textContent = "";
  } else {
    el.hidden = false;
    el.textContent = message;
  }
}

async function runSimulation() {
  showSimError(null);
  const body = {
    substrate: {
      material: $("substrate-material").value,
      domain_width: { value: parseFloat($("domain-width").value), unit: "nm" },
      thickness: { value: parseFloat($("substrate-thickness").value), unit: "nm" },
    },
    steps: state.steps,
  };
  let response;
  try {
    response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    showSimError(`Erreur reseau: ${err}`);
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    showSimError(data.detail ? JSON.stringify(data.detail) : "Erreur de simulation");
    return;
  }
  state.frames = data.frames;
  state.globalBounds = computeGlobalBounds(data.frames);
  const slider = $("frame-slider");
  slider.max = String(data.frames.length - 1);
  slider.value = String(data.frames.length - 1);
  setFrame(data.frames.length - 1);
  fitZoomToStructure();
}

function computeGlobalBounds(frames) {
  let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
  let any = false;
  for (const frame of frames) {
    x1 = Math.max(x1, frame.domain_width_nm);
    for (const layer of frame.layers) {
      for (const ring of layer.rings) {
        for (const [x, y] of ring.exterior) {
          if (!any) { y0 = y; y1 = y; any = true; }
          x0 = Math.min(x0, x); x1 = Math.max(x1, x);
          y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        }
      }
    }
  }
  if (!any) { y0 = -10; y1 = 10; }
  const pad = Math.max((y1 - y0) * 0.05, 1);
  return { x0, x1, y0: y0 - pad, y1: y1 + pad };
}

function setFrame(index) {
  state.frameIndex = index;
  const frame = state.frames[index];
  $("frame-label").textContent = `${index}/${state.frames.length - 1} - ${frame.step_kind}: ${frame.step_name}`;
  drawView($("view-overview"), frame, state.globalBounds);
  drawZoomView();
}

function drawZoomView() {
  const frame = state.frames[state.frameIndex];
  if (!frame) return;
  const box = {
    x0: parseFloat($("zoom-x0").value),
    x1: parseFloat($("zoom-x1").value),
    y0: parseFloat($("zoom-y0").value),
    y1: parseFloat($("zoom-y1").value),
  };
  drawView($("view-zoom"), frame, box);
}

function fitZoomToStructure() {
  if (!state.globalBounds) return;
  const b = state.globalBounds;
  $("zoom-x0").value = b.x0;
  $("zoom-x1").value = b.x1;
  $("zoom-y0").value = b.y0;
  $("zoom-y1").value = b.y1;
  drawZoomView();
}

function ringToPathD(ring) {
  const toSeg = (pts) => "M " + pts.map(([x, y]) => `${x},${y}`).join(" L ") + " Z";
  let d = toSeg(ring.exterior);
  for (const hole of ring.holes) d += " " + toSeg(hole);
  return d;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function drawView(svg, frame, box) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!box || box.x1 <= box.x0 || box.y1 <= box.y0) return;
  svg.setAttribute("viewBox", `${box.x0} ${-box.y1} ${box.x1 - box.x0} ${box.y1 - box.y0}`);

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("transform", "scale(1,-1)");
  svg.appendChild(g);

  for (const layer of frame.layers) {
    let d = "";
    for (const ring of layer.rings) d += ringToPathD(ring) + " ";
    if (!d.trim()) continue;
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d.trim());
    path.setAttribute("fill", state.materialColors[layer.material] || "#999");
    path.setAttribute("fill-rule", "evenodd");
    path.setAttribute("stroke", "rgba(0,0,0,0.25)");
    path.setAttribute("stroke-width", String((box.x1 - box.x0) * 0.002));
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = layer.material;
    path.appendChild(title);
    g.appendChild(path);
  }
}

function wireEvents() {
  $("step-kind").addEventListener("change", switchStepKindFields);
  $("pla-mode").addEventListener("change", switchPlanarizationMode);
  $("add-step-btn").addEventListener("click", () => {
    try {
      state.steps.push(buildStepFromForm());
      renderStepList();
      $("step-name").value = "";
    } catch (err) {
      showSimError(String(err));
    }
  });
  $("simulate-btn").addEventListener("click", () => {
    runSimulation().catch((err) => showSimError(String(err)));
  });
  $("frame-slider").addEventListener("input", (e) => setFrame(parseInt(e.target.value, 10)));
  $("zoom-fit-btn").addEventListener("click", fitZoomToStructure);
  for (const id of ["zoom-x0", "zoom-x1", "zoom-y0", "zoom-y1"]) {
    $(id).addEventListener("change", drawZoomView);
  }
}

async function init() {
  wireEvents();
  switchStepKindFields();
  switchPlanarizationMode();
  await loadLibraries();
}

init();
