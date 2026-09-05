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

function openModal(modalId) {
  document.querySelectorAll(".modal").forEach((m) => {
    m.hidden = m.id !== modalId;
  });
  $("modal-backdrop").hidden = false;
}

function closeModals() {
  $("modal-backdrop").hidden = true;
}

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
  for (const id of ["substrate-material", "dep-material", "epi-material", "pla-stop-material", "litho-material", "strip-material"]) {
    $(id).innerHTML = optionsHtml(materialNames);
  }
  $("litho-material").value = materialNames.includes("Photoresist") ? "Photoresist" : materialNames[0];
  $("strip-material").value = materialNames.includes("Photoresist") ? "Photoresist" : materialNames[0];
  $("epi-material").value = materialNames.includes("GaN") ? "GaN" : materialNames[0];

  $("dep-recipe").innerHTML = optionsHtml(recipes.deposition.map((r) => r.name));
  $("etch-recipe").innerHTML = optionsHtml(recipes.etch.map((r) => r.name));
  renderRecipeLists();
}

function switchStepKindFields() {
  const kind = $("step-kind").value;
  for (const el of document.querySelectorAll(".step-fields")) {
    el.classList.toggle("active", el.id === `fields-${kind}`);
  }
}

function switchEpiOrientation() {
  const orientation = $("epi-orientation").value;
  $("epi-angle-wrap").style.display = orientation === "semi_polar" ? "flex" : "none";
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
  if (kind === "epitaxial_growth") {
    const orientation = $("epi-orientation").value;
    const seedText = $("epi-seed-materials").value.trim();
    const seedMaterials = seedText
      ? seedText.split(",").map((s) => s.trim()).filter(Boolean)
      : [];
    const step = {
      kind, name,
      material: $("epi-material").value,
      thickness: { value: parseFloat($("epi-thickness").value), unit: "nm" },
      orientation,
      seed_materials: seedMaterials,
    };
    if (orientation === "semi_polar") {
      step.angle_deg = parseFloat($("epi-angle").value);
    }
    return step;
  }
  if (kind === "chemical") {
    return { kind, name, description: $("chem-description").value || null };
  }
  throw new Error(`unknown step kind ${kind}`);
}

function renderStepList() {
  const list = $("step-list");
  list.innerHTML = "";
  $("step-list-empty").hidden = state.steps.length > 0;
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

function buildSubstrateSpec() {
  return {
    material: $("substrate-material").value,
    domain_width: { value: parseFloat($("domain-width").value), unit: "nm" },
    thickness: { value: parseFloat($("substrate-thickness").value), unit: "nm" },
  };
}

async function runSimulation() {
  showSimError(null);
  const body = { substrate: buildSubstrateSpec(), steps: state.steps };
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

function showFollowMessage(kind, message) {
  const errorEl = $("follow-error");
  const successEl = $("follow-success");
  errorEl.hidden = true;
  successEl.hidden = true;
  if (!message) return;
  const el = kind === "error" ? errorEl : successEl;
  el.hidden = false;
  el.textContent = message;
}

async function exportToFollow() {
  showFollowMessage(null);
  const repoPath = $("follow-repo-path").value.trim();
  const title = $("follow-title").value.trim();
  const intent = $("follow-intent").value.trim();
  if (!repoPath || !title || !intent) {
    showFollowMessage("error", "Chemin du depot, titre et intention sont obligatoires.");
    return;
  }
  const body = {
    substrate: buildSubstrateSpec(),
    steps: state.steps,
    repo_path: repoPath,
    branch: $("follow-branch").value.trim() || "main",
    title,
    intent,
  };
  let response;
  try {
    response = await fetch("/api/export_follow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    showFollowMessage("error", `Erreur reseau: ${err}`);
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    showFollowMessage("error", data.detail ? JSON.stringify(data.detail) : "Erreur d'export");
    return;
  }
  showFollowMessage("success", `Experience Follow ${data.experiment_id} committee sur la branche "${data.branch}".`);
}

const RECIPE_MODES = {
  deposition: ["conformal", "directional"],
  etch: ["isotropic", "directional"],
};

function populateRecipeModeOptions() {
  const kind = $("recipe-kind").value;
  $("recipe-mode").innerHTML = optionsHtml(RECIPE_MODES[kind]);
  switchRecipeFormFields();
}

function switchRecipeFormFields() {
  const kind = $("recipe-kind").value;
  const mode = $("recipe-mode").value;
  $("recipe-angle-wrap").style.display = mode === "directional" ? "flex" : "none";
  $("recipe-etch-fields").classList.toggle("active", kind === "etch");
}

function parseFactorMap(text) {
  const out = {};
  for (const part of text.split(",").map((s) => s.trim()).filter(Boolean)) {
    const [name, factor] = part.split(":").map((s) => s.trim());
    if (name && factor !== undefined) out[name] = parseFloat(factor);
  }
  return out;
}

function factorMapToText(map) {
  return Object.entries(map || {}).map(([k, v]) => `${k}:${v}`).join(", ");
}

function buildRecipeFromForm() {
  const kind = $("recipe-kind").value;
  const name = $("recipe-name").value.trim();
  if (!name) throw new Error("Le nom de la recette est obligatoire.");
  const mode = $("recipe-mode").value;
  const angle_deg = mode === "directional" ? parseFloat($("recipe-angle").value) : 0.0;
  const notes = $("recipe-notes").value.trim() || null;

  if (kind === "deposition") {
    return { name, mode, angle_deg, notes };
  }
  return {
    name,
    mode,
    angle_deg,
    default_factor: parseFloat($("recipe-default-factor").value),
    selectivity_by_material: parseFactorMap($("recipe-sel-material").value),
    selectivity_by_category: parseFactorMap($("recipe-sel-category").value),
    notes,
  };
}

function clearRecipeForm() {
  $("recipe-name").value = "";
  $("recipe-angle").value = "0";
  $("recipe-default-factor").value = "1.0";
  $("recipe-sel-material").value = "";
  $("recipe-sel-category").value = "";
  $("recipe-notes").value = "";
  populateRecipeModeOptions();
}

function loadRecipeIntoForm(kind, recipe) {
  $("recipe-kind").value = kind;
  populateRecipeModeOptions();
  $("recipe-name").value = recipe.name;
  $("recipe-mode").value = recipe.mode;
  $("recipe-angle").value = recipe.angle_deg;
  if (kind === "etch") {
    $("recipe-default-factor").value = recipe.default_factor;
    $("recipe-sel-material").value = factorMapToText(recipe.selectivity_by_material);
    $("recipe-sel-category").value = factorMapToText(recipe.selectivity_by_category);
  }
  $("recipe-notes").value = recipe.notes || "";
  switchRecipeFormFields();
}

function showRecipeMessage(kind, message) {
  const errorEl = $("recipe-error");
  const successEl = $("recipe-success");
  errorEl.hidden = true;
  successEl.hidden = true;
  if (!message) return;
  const el = kind === "error" ? errorEl : successEl;
  el.hidden = false;
  el.textContent = message;
}

function renderRecipeLists() {
  for (const kind of ["deposition", "etch"]) {
    const list = $(`recipe-list-${kind}`);
    list.innerHTML = "";
    for (const recipe of state.recipes[kind]) {
      const li = document.createElement("li");
      const angleText = recipe.mode === "directional" ? ` - ${recipe.angle_deg}deg` : "";
      const badgeClass = recipe.is_custom ? "badge custom" : "badge";
      const badgeText = recipe.is_custom ? "personnalisee" : "integree";
      li.innerHTML = `
        <div class="recipe-row">
          <strong>${recipe.name}</strong>
          <span class="${badgeClass}">${badgeText}</span>
        </div>
        <span>${recipe.mode}${angleText}</span>
      `;
      li.addEventListener("click", () => loadRecipeIntoForm(kind, recipe));
      if (recipe.is_custom) {
        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.textContent = "Supprimer";
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteRecipe(kind, recipe.name).catch((err) => showRecipeMessage("error", String(err)));
        });
        li.appendChild(delBtn);
      }
      list.appendChild(li);
    }
  }
}

async function saveRecipe() {
  showRecipeMessage(null);
  const kind = $("recipe-kind").value;
  let body;
  try {
    body = buildRecipeFromForm();
  } catch (err) {
    showRecipeMessage("error", String(err));
    return;
  }
  const response = await fetch(`/api/recipes/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    showRecipeMessage("error", data.detail ? JSON.stringify(data.detail) : "Erreur d'enregistrement");
    return;
  }
  state.recipes = data;
  renderRecipeLists();
  $("dep-recipe").innerHTML = optionsHtml(data.deposition.map((r) => r.name));
  $("etch-recipe").innerHTML = optionsHtml(data.etch.map((r) => r.name));
  showRecipeMessage("success", `Recette "${body.name}" enregistree.`);
}

async function deleteRecipe(kind, name) {
  const response = await fetch(`/api/recipes/${kind}/${encodeURIComponent(name)}`, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) {
    showRecipeMessage("error", data.detail ? JSON.stringify(data.detail) : "Erreur de suppression");
    return;
  }
  state.recipes = data;
  renderRecipeLists();
  $("dep-recipe").innerHTML = optionsHtml(data.deposition.map((r) => r.name));
  $("etch-recipe").innerHTML = optionsHtml(data.etch.map((r) => r.name));
  showRecipeMessage("success", `Recette "${name}" supprimee.`);
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

function niceStep(range) {
  const rough = range / 6;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / magnitude;
  if (residual > 5) return 10 * magnitude;
  if (residual > 2) return 5 * magnitude;
  if (residual > 1) return 2 * magnitude;
  return magnitude;
}

function formatAxisValue(v) {
  return Math.abs(v) < 1e-9 ? "0" : String(Math.round(v * 100) / 100);
}

function drawGrid(svg, box, stepX, stepY) {
  const gridGroup = document.createElementNS(SVG_NS, "g");
  gridGroup.setAttribute("transform", "scale(1,-1)");

  const xStart = Math.ceil(box.x0 / stepX) * stepX;
  for (let x = xStart; x <= box.x1; x += stepX) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", -box.y0);
    line.setAttribute("y2", -box.y1);
    line.setAttribute("class", Math.abs(x) < stepX / 1e6 ? "axis-line" : "axis-grid");
    gridGroup.appendChild(line);
  }
  const yStart = Math.ceil(box.y0 / stepY) * stepY;
  for (let y = yStart; y <= box.y1; y += stepY) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("y1", -y);
    line.setAttribute("y2", -y);
    line.setAttribute("x1", box.x0);
    line.setAttribute("x2", box.x1);
    line.setAttribute("class", Math.abs(y) < stepY / 1e6 ? "axis-line" : "axis-grid");
    gridGroup.appendChild(line);
  }
  svg.appendChild(gridGroup);
}

function drawAxisLabels(svg, box, stepX, stepY) {
  // Direct children of <svg> (not the flipped group used for the grid/geometry) so the text
  // itself renders upright; placing them at (x, -y) reuses the same y-flip the viewBox encodes.
  const fontSize = (box.x1 - box.x0) * 0.022;
  const labelGroup = document.createElementNS(SVG_NS, "g");
  const xStart = Math.ceil(box.x0 / stepX) * stepX;
  for (let x = xStart; x <= box.x1; x += stepX) {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", x);
    t.setAttribute("y", -box.y0 + fontSize * 1.3);
    t.setAttribute("class", "axis-label");
    t.setAttribute("font-size", fontSize);
    t.setAttribute("text-anchor", "middle");
    t.textContent = formatAxisValue(x);
    labelGroup.appendChild(t);
  }
  const yStart = Math.ceil(box.y0 / stepY) * stepY;
  for (let y = yStart; y <= box.y1; y += stepY) {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", box.x0 + fontSize * 0.4);
    t.setAttribute("y", -y - fontSize * 0.35);
    t.setAttribute("class", "axis-label");
    t.setAttribute("font-size", fontSize);
    t.textContent = formatAxisValue(y);
    labelGroup.appendChild(t);
  }
  svg.appendChild(labelGroup);
}

function drawView(svg, frame, box) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!box || box.x1 <= box.x0 || box.y1 <= box.y0) return;
  // A small margin around the true data box, so the axis tick labels (drawn just outside it)
  // have room instead of being clipped by the viewBox edge.
  const marginX = (box.x1 - box.x0) * 0.06;
  const marginTop = (box.y1 - box.y0) * 0.06;
  const marginBottom = (box.y1 - box.y0) * 0.16;
  svg.setAttribute(
    "viewBox",
    `${box.x0 - marginX} ${-(box.y1 + marginTop)} ${box.x1 - box.x0 + 2 * marginX} ${box.y1 - box.y0 + marginTop + marginBottom}`
  );

  const stepX = niceStep(box.x1 - box.x0);
  const stepY = niceStep(box.y1 - box.y0);
  drawGrid(svg, box, stepX, stepY);

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

  drawAxisLabels(svg, box, stepX, stepY);
}

function wireEvents() {
  $("step-kind").addEventListener("change", switchStepKindFields);
  $("epi-orientation").addEventListener("change", switchEpiOrientation);
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
  $("export-follow-btn").addEventListener("click", () => {
    exportToFollow().catch((err) => showFollowMessage("error", String(err)));
  });
  $("recipe-kind").addEventListener("change", populateRecipeModeOptions);
  $("recipe-mode").addEventListener("change", switchRecipeFormFields);
  $("save-recipe-btn").addEventListener("click", () => {
    saveRecipe().catch((err) => showRecipeMessage("error", String(err)));
  });
  $("clear-recipe-form-btn").addEventListener("click", clearRecipeForm);
  $("frame-slider").addEventListener("input", (e) => setFrame(parseInt(e.target.value, 10)));
  $("zoom-fit-btn").addEventListener("click", fitZoomToStructure);
  for (const id of ["zoom-x0", "zoom-x1", "zoom-y0", "zoom-y1"]) {
    $(id).addEventListener("change", drawZoomView);
  }

  $("open-recipes-btn").addEventListener("click", () => openModal("modal-recipes"));
  $("open-follow-btn").addEventListener("click", () => openModal("modal-follow"));
  document.querySelectorAll("[data-close-modal]").forEach((btn) => btn.addEventListener("click", closeModals));
  $("modal-backdrop").addEventListener("click", (e) => {
    if (e.target.id === "modal-backdrop") closeModals();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModals();
  });
}

async function init() {
  wireEvents();
  switchStepKindFields();
  switchEpiOrientation();
  switchPlanarizationMode();
  populateRecipeModeOptions();
  await loadLibraries();
}

init();
