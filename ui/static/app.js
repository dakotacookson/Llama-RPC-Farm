/* Llama RPC Farm UI */
const $ = (sel) => document.querySelector(sel);

/* Safe binder: one missing/stale element must never kill the rest of
   the UI (a stale cached index.html used to throw here and left every
   button below it dead with zero feedback). */
function on(sel, evt, fn) {
  const el = document.querySelector(sel);
  if (!el) { console.warn("[ui] missing element, skipped:", sel); return; }
  el.addEventListener(evt, fn);
}
console.log("[ui] app.js loaded");

/* Access-key identity: persisted locally, attached to every API call.
   Set once via the key modal (or pre-seeded by pasting the generated
   token). Keys scope model visibility server-side; loopback console
   needs none. */
let _accessKey = null;
try { _accessKey = localStorage.getItem("farm-access-key") || null; } catch (_) {}

function setAccessKey(k) {
  _accessKey = (k || "").trim() || null;
  try {
    if (_accessKey) localStorage.setItem("farm-access-key", _accessKey);
    else localStorage.removeItem("farm-access-key");
  } catch (_) {}
}

async function api(method, url, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (_accessKey) opts.headers["Authorization"] = `Bearer ${_accessKey}`;
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || res.statusText);
  }
  return res.json();
}

const fmtBytes = (b) => b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB";

/* Accordion section state: keeps which sections are open across re-renders
   (e.g. the scan list re-renders on a timer) AND across page loads (persisted
   to the backend config/accordion_state.json). State lives on the container
   element so it survives innerHTML replacement. */
let archPersisted = {};   // key -> open, loaded from backend
let archPersistLoaded = false;
async function loadArchPersisted() {
  try { archPersisted = await api("GET", "/api/accordion"); }
  catch (_) { archPersisted = {}; }
  archPersistLoaded = true;
}
function archIsOpen(container, key) {
  if (container && container._archOpen && key in container._archOpen) {
    return !!container._archOpen[key];
  }
  // fall back to persisted state
  return !!archPersisted[key];
}
function archToggle(container, key) {
  if (!container) { container = {}; }
  if (!container._archOpen) container._archOpen = {};
  container._archOpen[key] = !container._archOpen[key];
  // persist to backend (fire-and-forget)
  archPersisted[key] = container._archOpen[key];
  api("POST", "/api/accordion", { key, open: container._archOpen[key] }).catch(() => {});
  return container._archOpen[key];
}
function makeArchSection(container, key, title, count) {
  const sec = document.createElement("div");
  sec.classList.add("arch-section");
  const header = document.createElement("button");
  header.type = "button";
  header.classList.add("arch-header");
  const open = archIsOpen(container, key);
  header.setAttribute("aria-expanded", open ? "true" : "false");
  header.innerHTML = `<span class="arch-chevron"><i class="fa-solid fa-chevron-right"></i></span><span class="arch-title">${escHtml(title)}</span><span class="arch-count">${count}</span>`;
  const archBody = document.createElement("div");
  archBody.classList.add("arch-body");
  archBody.hidden = !open;
  header.addEventListener("click", () => {
    const now = archToggle(container, key);
    if (now) {
      // opening: unhide + opt into the slide-fade (poll rebuilds never animate)
      archBody.classList.remove("arch-closing");
      archBody.hidden = false;
      archBody.classList.add("arch-anim");
    } else {
      // closing: play the fade, then hide (mirrors the details interceptor)
      archBody.classList.add("arch-closing");
      setTimeout(() => {
        archBody.classList.remove("arch-closing");
        archBody.hidden = true;
      }, 170);
    }
    header.setAttribute("aria-expanded", now ? "true" : "false");
  });
  sec.appendChild(header);
  sec.appendChild(archBody);
  return { sec, archBody };
}

/* Toast: replaces native alert()/confirm(), which browsers block or
   suppress (that silently killed the delete/status buttons). */
function toast(msg, kind = "info", ms = 6000) {
  let host = $("#toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "toast-host";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = msg;
  host.appendChild(el);
  while (host.children.length > 5) host.firstChild.remove();
  setTimeout(() => el.remove(), ms);
  console.log("[ui]", kind, msg);
}

/* ---------------- tabs ---------------- */
function triggerTabEnter(tabEl) {
  tabEl.classList.add("entering");
  // force reflow so animation restarts if re-entering same tab quickly
  void tabEl.offsetWidth;
  clearTimeout(tabEl._enteringTimer);
  tabEl._enteringTimer = setTimeout(() => tabEl.classList.remove("entering"), 520);
}

// programmatic tab switch (same code path as a nav click)
function switchToTab(name) {
  const btn = document.querySelector(`nav button[data-tab="${name}"]`);
  if (btn) btn.click();
}

document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    const tabEl = $(`#tab-${btn.dataset.tab}`);
    tabEl.classList.add("active");
    triggerTabEnter(tabEl);
    if (btn.dataset.tab === "dashboard") startAutoRefresh();
    else stopAutoRefresh();
    if (btn.dataset.tab === "machines") { loadMachines(); discoverInit(); }
    if (btn.dataset.tab === "groups") { loadGroups(); renderGroupMachines([]); }
    if (btn.dataset.tab === "visualizer") { initFlowVisualizer(); }
    if (btn.dataset.tab === "logs") { initLogsTab(); }
    if (btn.dataset.tab === "models") {
      loadModels();
      getAppSettings().then(s => {
        $("#scan-path").value = s.model_dir || "store://llm-models";
        const lbl = $("#scan-path-label");
        if (lbl) lbl.textContent = `scanning ${s.model_dir || "store://llm-models"}`;
        if (scanDirty || !$("#scan-results").querySelector(".scan-model")) runScan();
        scanDirty = false;
      });
      startScanRefresh();
    }
    else stopScanRefresh();
    if (btn.dataset.tab === "downloads") {
      initHfBrowse();
      loadDownloads();
    }
    if (btn.dataset.tab === "dljobs") loadDownloads();
    if (btn.dataset.tab === "chat") loadChat();
    if (btn.dataset.tab === "settings") { loadSettings(); renderThemeGrid(); }
    if (btn.dataset.tab === "launch") loadLaunch();
    if (btn.dataset.tab === "servegroups") { loadServeGroups(); renderServeGroupMembers([]); }
    if (btn.dataset.tab === "access") loadAccess();
    if (btn.dataset.tab === "user") {
      loadUser();
      clearInterval(_userTimer);
      _userTimer = setInterval(_userRefreshVisible, 10000);
    } else {
      clearInterval(_userTimer);
      _userTimer = null;
    }
  });
});

let _userTimer = null;

/* ---------------- user page (session + logins) ---------------- */
async function loadUser() {
  const me = window.__me || await api("GET", "/api/auth/me").catch(() => null);
  const line = $("#user-session-line");
  const heroName = $("#user-hero-name");
  const heroSub = $("#user-hero-sub");
  const roleBadge = $("#user-hero-role");
  if (heroName) heroName.textContent = me?.user || me?.name || "console";
  if (heroSub) heroSub.textContent = !me
    ? "not signed in"
    : (me.trust_via === "session-cookie"
        ? `cookie session as "${me.user}"`
        : me.trust_via === "key"
          ? "access key identity (Bearer header)"
          : (me.trust_via === "loopback" ? "console — loopback, no login" : "unauthenticated"));
  if (roleBadge) {
    if (me?.admin) {
      roleBadge.textContent = "admin";
      roleBadge.style.display = "";
      roleBadge.style.background = "var(--accent-dim)";
      roleBadge.style.color = "var(--accent)";
      roleBadge.title = "admin identity — control surfaces open";
    } else if (me?.name && me?.name !== "console") {
      roleBadge.textContent = me.scoped ? "strict user" : "user";
      roleBadge.style.display = "";
      roleBadge.style.background = "var(--surface-dim,#0001)";
      roleBadge.style.color = "var(--muted)";
      roleBadge.title = "identity without admin rights";
    } else roleBadge.style.display = "none";
  }
  // my-model-access card: the LIST AS THIS IDENTITY SEES IT — the strict
  // posture live (admins in strict mode see exactly their grants here)
  try {
    const vis = me ? await api("GET", "/api/models") : [];
    const all = (me && me.admin)
      ? await api("GET", "/api/models?full=1").catch(() => vis) : vis;
    const card = $("#user-models");
    const lineA = $("#user-access-line");
    const hiddenCount = (all.length || 0) - (vis.length || 0);
    if (lineA) lineA.innerHTML = me
      ? `you see <b>${vis.length}</b> of <b>${all.length}</b> catalog models` +
        (hiddenCount > 0 ? ` — <span style="color:var(--muted)">${hiddenCount} hidden by your key's grants / strict chip</span>` : "")
      : "no identity — no models";
    if (card) {
      card.innerHTML = (vis && vis.length)
        ? vis.map(m => `
        <div class="server-chip" title="${escHtml(m.model_id)}">
          <span class="chip-name">${escHtml((m.name || m.model_id || "").slice(0, 24))}</span>
          ${m.decision_type ? '<span class="badge badge-master">dec</span>' : ""}
        </div>`).join("")
        : '<span class="hint">no models granted</span>';
    }
  } catch (err) {
    const lineA = $("#user-access-line");
    if (lineA) lineA.textContent = "model list failed: " + err.message;
  }
  // usage cards (all-time totals + series graph)
  loadUsageTotals();
  loadUsageSeries("day");

  // my API key card (only when the session resolves to a real key)
  loadMyKey();
  // admin extras (logins CRUD) — hidden unless identity says so
}

async function loadUsageByModel(light = false) {
  /* per-model deep view: one card per model — dial = compute hours
     (de-duplicated vs the decode-speed tab's tok/s dial); totals,
     true avg tok/s, 30-day bars, last-used. light: patch in place. */
  const host = $("#ubm-cards");
  if (!host) return;
  const grid = host.querySelector(".launch-grid");
  if (light && grid && grid.children.length) {
    try {
      const r = await api("GET", "/api/usage/by-model?days=30");
      const mods = {};
      for (const [k, v] of Object.entries(r)) {
        if (k !== "identity" && v && (v.days || v.total)) mods[k] = v;
      }
      const cards = Object.entries(mods).sort((a, b) =>
        (b[1].total?.requests || 0) - (a[1].total?.requests || 0));
      // same count → patch values in place; count change → full rebuild
      if (cards.length === grid.children.length) {
        const maxH = Math.max(0.01, ...cards.map(([, x]) =>
          x.total?.hours || 0)) * 1.15;
        cards.forEach(([m, d], i) => {
          const card = grid.children[i];
          const hours = d.total?.hours || 0;
          const sub = card.querySelectorAll(".hint")[0];
          if (sub) sub.innerHTML =
            `${d.total?.requests||0} req · ${_fmtTok(d.total?.tok_out||0)} out · ${_fmtTok(d.total?.tok_in||0)} in${d.avg_spd ? ` · <b>${d.avg_spd} tok/s</b>` : ""}${d.total?.errors ? ` · <span style="color:var(--warn)">${d.total.errors} err</span>` : ""}`;
          const val = card.querySelector("text.gauge-val");
          if (val && _gaugePatch(card, hours, maxH, _fmtHours)) {
            const nameEl = card.firstElementChild;
            if (nameEl) nameEl.title = m;
          }
        });
        return;
      }
    } catch (err) { /* fall through to full render */ }
  }
  host.innerHTML = "loading…";
  try {
    const r = await api("GET", "/api/usage/by-model?days=30");
    const mods = {};
    for (const [k, v] of Object.entries(r)) {
      if (k === "identity") continue;
      if (v && (v.days || v.total)) mods[k] = v;
    }
    const cards = Object.entries(mods).sort((a, b) =>
      (b[1].total?.requests || 0) - (a[1].total?.requests || 0));
    if (!cards.length) {
      host.innerHTML = '<div class="card gauge-card accent"><span class="hint">no usage recorded yet — every model you use through the gateway gets a card here</span></div>';
      return;
    }
    host.innerHTML = `<div class="launch-grid" style="grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:var(--s-3); align-items:stretch">` +
      cards.map(([m, d]) => {
        const t = d.total || {};
        const hours = t.hours || 0;
        const maxH = Math.max(0.01, ...cards.map(([, x]) => x.total?.hours || 0)) * 1.15;
        return `<div class="card gauge-card accent" style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:8px 10px">
          <div style="font-size:0.95em; font-weight:600; width:100%; text-align:center; overflow:hidden; text-overflow:ellipsis" title="${escHtml(m)}">${escHtml(m.length > 30 ? m.slice(0,29) + "…" : m)}</div>
          ${_gaugeSkel(hours, maxH, "accent", _fmtHours, "compute time", "")}
          <div class="hint" style="width:100%; text-align:center; margin-top:2px">${t.requests||0} req · ${_fmtTok(t.tok_out||0)} out · ${_fmtTok(t.tok_in||0)} in${d.avg_spd ? ` · <b>${d.avg_spd} tok/s</b>` : ""}${t.errors ? ` · <span style="color:var(--warn)">${t.errors} err</span>` : ""}</div>
        </div>`;
      }).join("") + `</div>`;
  } catch (err) {
    host.textContent = "by-model failed: " + err.message;
  }
}

function _userRefreshVisible() {
  if (!document.querySelector("#tab-user")?.classList.contains("active"))
    return;
  const which = {
    speed: () => loadUsageTotals(true),
    alltime: () => loadUsageTotals(true),
    overtime: () => loadUsageSeries(usagePeriod, true),
    bymodel: () => loadUsageByModel(true),
  };
  for (const [n, fn] of Object.entries(which)) {
    const p = $("#user-pane-" + n);
    if (p && !p.hidden) { fn(); break; }
  }
  loadUserAccessChipsLite();
}

/* LIGHT update = write into EXISTING nodes only (textContent/attrs);
   the caller's full renderer keeps building first time. To keep this
   maintainable each renderer owns its light path: same data, then
   patch values via data-field hooks. Gauge cards carry data-field
   markers on the value text/arc/peak/sub nodes. */
function _gaugePatch(card, value, max, fmt) {
  const svg = card.querySelector("svg");
  if (!svg) return false;
  // value text (3rd text node class gauge-val)
  const val = svg.querySelector("text.gauge-val");
  if (val) val.textContent = fmt(value ?? 0);
  // arc path = 2nd path (track first)
  const paths = svg.querySelectorAll("path");
  if (paths.length === 2) {
    const frac = Math.max(0, Math.min(1, value / Math.max(max, 1e-9)));
    // recompute d for the 270° sweep
    const W = 300, H = 175, cx = W/2, cy = H*0.78, R = 105;
    const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
    const r1 = a0 + (a1 - a0) * frac;
    const x1 = cx + R * Math.cos(r1), y1 = cy + R * Math.sin(r1);
    const large = (r1 - a0) > Math.PI ? 1 : 0;
    const x0 = cx + R * Math.cos(a0), y0 = cy + R * Math.sin(a0);
    if (frac <= 0) {
      paths[1].setAttribute("opacity", "0");
    } else {
      paths[1].setAttribute("opacity", "0.95");
      paths[1].setAttribute("d", `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${R} ${R} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`);
    }
  }
  const peak = [...svg.querySelectorAll("text")].find(t => t.textContent.startsWith("peak"));
  if (peak) peak.textContent = `peak ${fmt(max)}`;
  return true;
}

async function loadUserAccessChipsLite() {
  try {
    const vis = await api("GET", "/api/models");
    const line = $("#user-access-line");
    if (line) line.textContent =
      `${vis.length} models granted`;
    const grid = $("#user-models");
    if (grid) {
      const want = new Set(vis.map(m => m.model_id));
      const have = new Set([...grid.querySelectorAll("[data-acc-model]")
        .map(c => c.dataset.accModel)]);
      const same = want.size === have.size &&
        [...want].every(x => have.has(x));
      if (!same) {   // grants changed: rebuild chips only (no flicker)
        grid.innerHTML = vis.map(m => `
        <div class="server-chip" data-acc-model="${escHtml(m.model_id)}" title="${escHtml(m.model_id)}">
          <span class="chip-name">${escHtml((m.name || m.model_id || "").slice(0, 24))}</span>
          ${m.decision_type ? '<span class="badge badge-master">dec</span>' : ""}
        </div>`).join("") || '<span class="hint">no models granted</span>';
      }
    }
  } catch (_) {}
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) _userRefreshVisible();
});

function userSubTab(name) {
  for (const n of ["info", "speed", "alltime", "overtime", "bymodel"]) {
    const btn = document.querySelector(`#user-subtabs [data-utab="${n}"]`);
    if (btn) btn.classList.toggle("active", n === name);
    const pane = $("#user-pane-" + n);
    if (pane) pane.hidden = n !== name;
  }
  // lazy per-pane loads on first open
  if (name === "speed" || name === "alltime") loadUsageTotals();
  if (name === "speed") loadDecodeSpeed();
  if (name === "overtime") loadUsageSeries(usagePeriod);
  if (name === "bymodel") loadUsageByModel();
}
document.querySelectorAll("#user-subtabs [data-utab]").forEach(b => {
  b.addEventListener("click", () => userSubTab(b.dataset.utab));
});

on("#user-logout", "click", async () => {
  try { await api("POST", "/api/auth/logout", {}); } catch (_) {}
  window.location.href = "/";
});

/* ---------------- access (per-IP model visibility) ----------------
   My-key card: the admin session's raw access key + copy button — the
   list endpoints mask tokens by design, so this is the one sanctioned
   copy path (server refuses console-only/non-admin asks with 404). */
async function loadMyKey() {
  const card = $("#acc-my-key-card");
  if (!card) return;
  try {
    const k = await api("GET", "/api/auth/my-key");
    $("#acc-my-key").textContent = k.key;
    card.hidden = false;
  } catch (_) {
    // console-only session or non-admin: the raw key isn't yours to copy
    card.hidden = true;
  }
}

on("#acc-edit-key-copy", "click", async () => {
  try {
    await navigator.clipboard.writeText($("#acc-edit-key").textContent);
    toast("key copied", "ok");
  } catch (_) { toast("select the key and Ctrl+C", "bad"); }
});
on("#acc-my-key-copy", "click", async () => {
  try {
    await navigator.clipboard.writeText($("#acc-my-key").textContent);
    toast("key copied", "ok");
  } catch (_) { toast("clipboard unavailable — click the key text and Ctrl+C", "bad"); }
});
   // selected model tokens (model_id for whole-model, model_id--variant
// for a single variant) — mirrors the launch-grid card selection idiom
let accessModelSel = new Set();

function renderAccessModelGrid(models, selectedTokens) {
  const grid = $("#acc-model-grid");
  if (!grid) return;
  accessModelSel = new Set(selectedTokens || []);
  const cells = [];
  for (const m of models) {
    // EXACT-grant semantics: the model card grants the base entry ONLY;
    // each variant card grants that exact variant (variants visible to
    // the view iff their own token is in the allowlist)
    const mid = m.model_id;
    const on = accessModelSel.has(mid);
    cells.push(`
    <div class="server-chip acc-model-card${on ? " sel" : ""}" data-acc-model="${escHtml(mid)}"${!m.variants?.length ? '' : ' data-has-variants="1"'}>
      <input type="checkbox" ${on ? "checked" : ""} tabindex="-1">
      <span class="chip-name">${escHtml(m.name || mid)}</span>
        <span class="chip-sub">${escHtml((m.base_path||"").split("/").pop().slice(0, 22))}${(m.variants||[]).length ? ` · ${(m.variants).length} variants` : ""}</span>
    </div>`);
    for (const v of (m.variants||[])) {
      const lab = (v.label||"").trim(); if (!lab) continue;
      const tok = `${mid}--${lab}`;
      const von = accessModelSel.has(tok);
      cells.push(`
      <div class="server-chip acc-model-card${von ? " sel" : ""}" data-acc-variant="${escHtml(tok)}" style="margin-left:14px; opacity:0.9">
        <input type="checkbox" ${von ? "checked" : ""} tabindex="-1">
        <span class="chip-name" title="${escHtml(tok)}">${escHtml(v.name || tok)}</span>
        <span class="chip-sub">variant · ${escHtml(lab)}</span>
      </div>`);
    }
  }
  grid.innerHTML = cells.join("") || '<span class="hint">no models configured</span>';
  grid.querySelectorAll("[data-acc-model]").forEach(card => {
    card.onclick = () => {
      const tok = card.dataset.accModel;
      const cb = card.querySelector("input");
      // EXACT grants now: toggling the model card does NOT touch its
      // variant picks — each card is its own independent grant
      if (cb.checked) { accessModelSel.delete(tok); cb.checked = false; card.classList.remove("sel"); }
      else { accessModelSel.add(tok); cb.checked = true; card.classList.add("sel"); }
    };
  });
  grid.querySelectorAll("[data-acc-variant]").forEach(card => {
    card.onclick = () => {
      const tok = card.dataset.accVariant;
      const cb = card.querySelector("input");
      // EXACT grants: variant toggle only adds/removes its own token —
      // the base model card is unaffected either way
      if (cb.checked) {
        accessModelSel.delete(tok); cb.checked = false; card.classList.remove("sel");
      } else {
        accessModelSel.add(tok); cb.checked = true; card.classList.add("sel");
      }
    };
  });
}

function accessViewPayload() {
  const ips = $("#acc-vw-ips").value.split("\n").map(s => s.trim()).filter(Boolean);
  return { ips, models: [...accessModelSel] };
}

let accessLoaded = false;
async function loadAccess() {
  const who = $("#access-whoami");
  let w = null;
  try { w = await api("POST", "/api/views/whoami", {}); } catch (_) {}
  if (w) {
    who.innerHTML = w.admin
      ? `key <b>${escHtml(w.name || "?")}</b> — admin (control access${w.scoped ? `, scoped to ${(w.models||[]).length} models` : ", unscoped"})`
      : w.scoped
        ? `key <b>${escHtml(w.name || "?")}</b> — scoped to ${(w.models||[]).length} models`
        : w.name
          ? `key <b>${escHtml(w.name)}</b> — unscoped (sees everything)`
          : `no key — ${w.trusted ? "console (loopback)" : "open access"}`;
  } else who.textContent = "whoami failed";
  // reveal the tab for trusted consoles AND admin keys
  const btn = $("#tab-btn-access");
  if (btn) btn.hidden = !(w && (w.trusted || w.admin));
  const kr = await api("GET", "/api/views").catch(() => ({ keys: [] }));
  accessAllModels = await api("GET", "/api/models?full=1").catch(() => []);
  await renderAccessKeys(kr.keys || []);
  loadMyKey();
  accessLoaded = true;
}

let accessAllModels = [];

async function renderAccessKeys(keys) {
  const host = $("#acc-views");
  if (!host) return;
  host.innerHTML = (keys || []).length
    ? keys.map(k => `
    <div class="server-chip" style="display:flex; gap:8px; align-items:center; margin-bottom:6px">
      <span class="chip-name"><b>${escHtml(k.name)}</b>${k.admin ? ' <span class="badge" style="background:var(--accent-dim);color:var(--accent)">admin</span>' : ""}</span>
      <span class="chip-sub" title="${escHtml((k.models||[]).join(", "))}">${escHtml(k.token_masked)} · ${(k.models||[]).length ? (k.models).length + " model grants" : "unscoped"} · <b>${k.unrestricted === false ? "strict" : "unrestricted"}</b></span>
      <button class="small" data-k-id="${escHtml(k.token_masked)}" data-k-name="${escHtml(k.name)}" data-k-admin="${k.admin ? 1 : 0}" data-k-models="${escHtml((k.models||[]).join(","))}" data-k-unrestricted="${k.unrestricted === false ? 0 : 1}">Edit</button>
      <button class="small danger" data-k-revoke="${escHtml(k.token_masked)}" data-k-name="${escHtml(k.name)}">Revoke</button>
    </div>`).join("")
    : '<span class="hint">no keys yet — generate one per person/service</span>';
  host.querySelectorAll("[data-k-id]").forEach(btn => {
    btn.onclick = async () => {
      accessLoadIntoEditor({
        token_masked: btn.dataset.kId, name: btn.dataset.kName,
        admin: btn.dataset.kAdmin === "1",
        models: (btn.dataset.kModels || "").split(",").filter(Boolean),
        unrestricted: btn.dataset.kUnrestricted !== "0",
      });
      // reveal + copy affordance (admins hand keys to users/rotate)
      try {
        const r = await api("POST", "/api/views/keys/reveal",
                            {name: btn.dataset.kName});
        const el = $("#acc-edit-key");
        const row = $("#acc-edit-key-row");
        if (el) el.textContent = r.key;
        if (row) row.classList.remove("hidden");
      } catch (_) {}
    };
  });
  host.querySelectorAll("[data-k-revoke]").forEach(btn => {
    btn.onclick = async () => {
      if (!armed(btn, "Revoke")) return;
      try {
        await api("POST", "/api/views/keys/revoke", { name: btn.dataset.kRevoke });
        toast(`key ${btn.dataset.kName} revoked — its clients lose access on next request`, "ok");
        try { sessionStorage.setItem("acc-mode-saved", "1"); } catch (_) {}
        setTimeout(() => window.location.reload(), 600);
      } catch (err) { toast(err.message, "bad"); }
    };
  });
}

/* editing state: masked token "" = new key */
let accessEditingMasked = "";

function accessLoadIntoEditor(k) {
  accessEditingMasked = k.token_masked || "";
  $("#acc-edit-key-row")?.classList.add("hidden");
  $("#acc-vw-name").value = k.name || "";
  $("#acc-key-admin").value = k.admin ? "true" : "false";
  $("#acc-key-unrestricted").value =
    (k.unrestricted === undefined ? true : !!k.unrestricted) ? "true" : "false";
  syncKeyEditorChips();
  renderAccessModelGrid(accessAllModels, k.models || []);
  $("#acc-key-generate").textContent = accessEditingMasked ? "Save changes" : "Generate key";
  $("#acc-key-generate").hidden = false;
  toast(accessEditingMasked ? `editing key ${k.name}` : "new key: pick models, then Generate", "ok");
}

/* key editor chip pairs — Admin/User + Unrestricted/Strict, the 2x2
   identity grid; hidden inputs carry the state for save */
function syncKeyEditorChips() {
  const admin = $("#acc-key-admin")?.value === "true";
  const unres = $("#acc-key-unrestricted")?.value !== "false";
  document.querySelectorAll("#chips-key-admin .hw-chip").forEach(b =>
    b.classList.toggle("active",
      b.dataset.value === (admin ? "admin" : "user")));
  document.querySelectorAll("#chips-key-visibility .hw-chip").forEach(b =>
    b.classList.toggle("active",
      b.dataset.value === (unres ? "unrestricted" : "strict")));
  // relevance hint: model grants only matter in strict mode
  const grantsCard = $("#acc-model-grid")?.closest(".card");
  if (grantsCard) grantsCard.style.opacity = (unres && !admin) || (!unres) ? "" : "0.45";
}

function wireKeyEditorChips() {
  document.querySelectorAll("#chips-key-admin .hw-chip").forEach(b => {
    b.addEventListener("click", () => {
      const el = $("#acc-key-admin");
      if (el) el.value = b.dataset.value === "admin" ? "true" : "false";
      syncKeyEditorChips();
    });
  });
  document.querySelectorAll("#chips-key-visibility .hw-chip").forEach(b => {
    b.addEventListener("click", () => {
      const el = $("#acc-key-unrestricted");
      if (el) el.value = b.dataset.value === "unrestricted" ? "true" : "false";
      syncKeyEditorChips();
    });
  });
}
/* EXACT-grant grid: model card = base entry token; variant card = its
   own token; independent selections, no cascading */
function renderAccessModelGrid(models, selectedTokens) {
  const grid = $("#acc-model-grid");
  if (!grid) return;
  accessModelSel = new Set(selectedTokens || []);
  const cells = [];
  for (const m of models) {
    const mid = m.model_id;
    const on = accessModelSel.has(mid);
    cells.push(`
    <div class="server-chip acc-model-card${on ? " sel" : ""}" data-acc-model="${escHtml(mid)}">
      <input type="checkbox" ${on ? "checked" : ""} tabindex="-1">
      <span class="chip-name">${escHtml(m.name || mid)}</span>
        <span class="chip-sub">${escHtml((m.base_path||"").split("/").pop().slice(0, 22))}${(m.variants||[]).length ? ` · ${(m.variants).length} variants` : ""}</span>
    </div>`);
    for (const v of (m.variants||[])) {
      const lab = (v.label||"").trim(); if (!lab) continue;
      const tok = `${mid}--${lab}`;
      const von = accessModelSel.has(tok);
      cells.push(`
      <div class="server-chip acc-model-card${von ? " sel" : ""}" data-acc-variant="${escHtml(tok)}" style="margin-left:14px; opacity:0.9">
        <input type="checkbox" ${von ? "checked" : ""} tabindex="-1">
        <span class="chip-name" title="${escHtml(tok)}">${escHtml(v.name || tok)}</span>
        <span class="chip-sub">variant · ${escHtml(lab)}</span>
      </div>`);
    }
  }
  grid.innerHTML = cells.join("") || '<span class="hint">no models configured</span>';
  grid.querySelectorAll("[data-acc-model]").forEach(card => {
    card.onclick = () => {
      const tok = card.dataset.accModel;
      const cb = card.querySelector("input");
      if (cb.checked) { accessModelSel.delete(tok); cb.checked = false; card.classList.remove("sel"); }
      else { accessModelSel.add(tok); cb.checked = true; card.classList.add("sel"); }
    };
  });
  grid.querySelectorAll("[data-acc-variant]").forEach(card => {
    card.onclick = () => {
      const tok = card.dataset.accVariant;
      const cb = card.querySelector("input");
      if (cb.checked) { accessModelSel.delete(tok); cb.checked = false; card.classList.remove("sel"); }
      else { accessModelSel.add(tok); cb.checked = true; card.classList.add("sel"); }
    };
  });
}

on("#acc-key-generate", "click", async () => {
  const name = $("#acc-vw-name").value.trim();
  const models = [...accessModelSel];
  const admin = $("#acc-key-admin")?.value === "true";
  const unrestricted = $("#acc-key-unrestricted")?.value !== "false";
  try {
    if (accessEditingMasked) {
      await api("POST", `/api/views/keys/${encodeURIComponent(accessEditingMasked)}/update`, { name, models, admin, unrestricted });
      toast(`key ${name} updated — refreshing…`, "ok");
    } else {
      const r = await api("POST", "/api/views/keys/generate", { name, models, admin, unrestricted });
      // the ONE time the raw token is shown — modal with copy button
      $("#acc-new-token").textContent = r.token;
      $("#acc-token-modal").classList.remove("hidden");
      $("#acc-vw-name").value = r.name;
      toast(`key ${r.name} generated`, "ok");
    }
    try { sessionStorage.setItem("acc-mode-saved", "1"); } catch (_) {}
    setTimeout(() => window.location.reload(), accessEditingMasked ? 600 : 60);
  } catch (err) { toast(err.message, "bad"); }
});

on("#acc-token-copy", "click", async () => {
  try { await navigator.clipboard.writeText($("#acc-new-token").textContent); toast("token copied", "ok"); }
  catch (_) { toast("clipboard unavailable — select and copy manually", "bad"); }
});

on("#acc-token-close", "click", () => $("#acc-token-modal").classList.add("hidden"));

/* key modal: paste an existing key to adopt it into this browser */
on("#acc-key-set", "click", () => {
  const el = document.getElementById("acc-key-input");
  setAccessKey(el ? el.value : "");
  toast("key stored in this browser", "ok");
  setTimeout(() => window.location.reload(), 400);
});

on("#acc-test-go", "click", async () => {
  const tok = $("#acc-test-ip").value.trim();
  const out = $("#acc-test-out");
  if (!tok) { toast("key required", "bad"); return; }
  out.hidden = false;
  out.textContent = "checking…";
  try {
    // exact visibility decision server-side for that KEY: whoami's
    // X-Diag-IP header now carries a key to simulate (never forwarded)
    const r = await fetch("/api/views/whoami", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Diag-IP": tok },
      body: "{}",
    });
    const j = await r.json();
    if (j.unknown_key) {
      out.textContent = `key ${tok.slice(0, 8)}… → unknown (client gets: private gate closed, all non-private models visible)`;
    } else if (j.admin) {
      const grants = j.scoped
        ? ` (visibility scoped to:\n${(j.models||[]).map(m => "    - " + m).join("\n")})`
        : ", unscoped";
      out.textContent = `key ${j.name} → admin${grants}`;
    } else if (j.scoped) {
      const list = (j.models || []).map(m => "    - " + m).join("\n");
      out.textContent = `key ${j.name} → scoped:\n  sees ONLY:\n${list || "    (nothing)"}`;
    } else {
      out.textContent = `key ${j.name || tok.slice(0, 8) + "…"} → unscoped (sees everything)`;
    }
  } catch (err) { out.textContent = "ERR " + err.message; }
});

/* ── sidebar section accordions ─────────────────────────────────
   Collapse any nav section; state persists per section (localStorage).
   The section holding the ACTIVE tab is always auto-expanded. */
function navCollapseInit() {
  document.querySelectorAll(".nav-sec.collapsible").forEach(sec => {
    const btn = sec.querySelector(".nav-collapse-btn");
    if (!btn) return;
    const label = sec.querySelector(".nav-label")?.textContent.trim() || "";
    const KEY = "nav-sec-" + label;
    let open = (localStorage.getItem(KEY) ?? "1") === "1";
    if ([...document.querySelectorAll("nav .nav-sec.collapsible")]
        .some(s => s.contains(
          document.querySelector("nav button[data-tab].active"))))
      if (sec.querySelector("button[data-tab].active")) open = true;
    const body = sec.querySelector(".nav-sec-body");
    const paint = () => {
      body.style.display = open ? "" : "none";
      const chev = btn.querySelector(".nav-chev");
      if (chev) chev.style.transform = open ? "" : "rotate(180deg)";
      btn.setAttribute("aria-expanded", String(open));
    };
    paint();
    btn.addEventListener("click", (e) => {
      open = !open;
      localStorage.setItem(KEY, open ? "1" : "0");
      paint();
    });
  });
  // when a nav tab activates, force its section open (never trap a view)
  document.querySelectorAll("nav button[data-tab]").forEach(b => {
    b.addEventListener("click", () => {
      const sec = b.closest(".nav-sec.collapsible");
      if (!sec) return;
      const body = sec.querySelector(".nav-sec-body");
      if (getComputedStyle(body).display === "none") {
        sec.querySelector(".nav-collapse-btn")?.click();
      }
    }, true);   // capture: run before the tab-switch handler
  });
}
navCollapseInit();

/* ---------------- dashboard ---------------- */
function barColor(pct) { return pct >= 90 ? "red" : pct >= 70 ? "yellow" : "green"; }
function valColor(pct) { return pct >= 90 ? "var(--bad)" : pct >= 70 ? "var(--warn)" : "var(--ok)"; }

function statCardHTML(m) {
  const online = m.status === "online";
  const gpus = m.gpus || [];
  const usedVram = gpus.reduce((a, g) => a + g.mem_used_gb, 0);
  const totalVram = gpus.reduce((a, g) => a + g.mem_total_gb, 0);
  const ramPct = m.ram_total_gb ? Math.round(m.ram_used_gb / m.ram_total_gb * 100) : 0;
  const vramPct = totalVram ? Math.round(usedVram / totalVram * 100) : 0;
  const osLabel = (m.os || "linux").charAt(0).toUpperCase() + (m.os || "linux").slice(1);
  // DEC-16: engine-correct card subtitle. Under llama_cpp verbatim
  // (RPC up/down + RDMA/TCP); under vllm there are no RPC peers, so
  // hidden-not-translated: host facts only (IP · load) — same
  // manifestHasSection idiom as the DEC-10 model rows.
  const vllmMode = manifestHasSection("vllm_section");
  // DEC-17: under vllm, machines the server says are vLLM-incapable
  // (vllm_capable === false — e.g. Windows hosts) render dimmed with an
  // explicit "not available under vLLM" line. The verdict comes from the
  // /api/stats payload stamped by the backend (B14 — no client-side
  // os/vendor re-derivation). Under llama_cpp the fields are ignored
  // entirely: cards render verbatim as before.
  const vllmBlind = vllmMode && m.vllm_capable === false;
  const vllmReason = m.vllm_capability_reason || "vLLM is not usable on this host";
  const cardSub = vllmMode
    ? `${m.ip}${m.load ? ` — load ${m.load}` : ""}`
    : `${m.ip} — RPC <span class="${m.rpc_running ? "status-ok" : "status-warn"}">${m.rpc_running ? "up" : "down"}</span>${m.rdma ? ' — <span class="badge-rdma" style="padding:1px 6px;border-radius:6px">RDMA</span>' : " — TCP"}${m.load ? ` — load ${m.load}` : ""}`;
  return `
    <div class="machine-status-card ${online ? "online" : "offline"}${vllmBlind ? " vllm-unavailable" : ""}" data-machine="${escHtml(m.name)}" title="${vllmBlind ? escHtml(vllmReason) : "Click for details"}">
      ${vllmBlind ? '<div class="vllm-unavailable-tag"><i class="fa-solid fa-ban"></i> not available under vLLM</div>' : ""}
      <div class="card-header">
        <span class="card-label">
          <span class="status-dot ${online ? "on" : "off"}"></span>
          <i class="fa-solid fa-server card-label-icon"></i>
          ${m.name}
          ${m.is_master ? '<span class="badge badge-master">master</span>' : ""}
          ${m.is_local ? '<span class="badge badge-local">this machine</span>' : ""}
        </span>
        <div class="card-actions">
          <span class="os-badge">${osLabel}</span>
        </div>
      </div>
      <div class="card-sub">${cardSub}</div>
      ${gpus.length ? `
        <table class="gpu-table">
          <thead><tr><th>Idx</th><th>GPU</th><th>Util</th><th>VRAM</th><th>Temp</th></tr></thead>
          <tbody>
            ${gpus.map(g => `
              <tr>
                <td>${g.index}</td>
                <td class="gpu-name">${g.name}</td>
                <td class="gpu-util">${g.util}%</td>
                <td class="gpu-vram">${g.mem_used_gb}/${g.mem_total_gb}GB</td>
                <td class="gpu-temp ${g.temp >= 75 ? "hot" : ""}">${g.temp}°C</td>
              </tr>`).join("")}
          </tbody>
        </table>` : `<div class="no-gpus">no GPUs</div>`}
      <div class="card-metrics">
        <div class="card-metric">
          <div class="card-metric-label"><i class="fa-solid fa-gauge-high"></i> CPU</div>
          <div class="card-metric-value" style="color:${valColor(m.cpu_pct)}">${Math.round(m.cpu_pct || 0)}%${m.cores ? ` · ${m.cores} cores` : ""}</div>
          <div class="metric-bar"><div class="metric-fill" style="width:${m.cpu_pct || 0}%; background:${valColor(m.cpu_pct)}"></div></div>
          ${m.cores ? `<div class="card-metric-label">${((m.cpu_pct || 0) / 100 * m.cores).toFixed(1)} / ${m.cores} cores</div>` : ""}
        </div>
        <div class="card-metric">
          <div class="card-metric-label"><i class="fa-solid fa-memory"></i> RAM</div>
          <div class="card-metric-value" style="color:${valColor(ramPct)}">${m.ram_used_gb}/${m.ram_total_gb}GB</div>
          <div class="metric-bar"><div class="metric-fill" style="width:${ramPct}%; background:${valColor(ramPct)}"></div></div>
        </div>
        <div class="card-metric">
          <div class="card-metric-label"><i class="fa-solid fa-microchip"></i> VRAM</div>
          <div class="card-metric-value" style="color:${valColor(vramPct)}">${usedVram.toFixed(1)}/${totalVram.toFixed(1)}GB</div>
          <div class="metric-bar"><div class="metric-fill" style="width:${vramPct}%; background:${valColor(vramPct)}"></div></div>
        </div>
      </div>
    </div>`;
}

/* ---- machine detail modal (dashboard card click) ---- */
// per-machine detail cache: render the last result INSTANTLY on reopen,
// then swap in fresh data when the fetch lands (same pattern the
// overview cards use — no dead spinner on a repeat visit)
const _detailCache = {};
function _detailCacheGet(name) {
  if (_detailCache[name]) return _detailCache[name];
  try {
    const raw = localStorage.getItem("machine-detail-" + name);
    if (raw) { const d = JSON.parse(raw); _detailCache[name] = d; return d; }
  } catch (_) { /* corrupt entry: treat as absent */ }
  return null;
}
function _detailCachePut(name, d) {
  _detailCache[name] = d;
  try { localStorage.setItem("machine-detail-" + name, JSON.stringify(d)); }
  catch (_) { /* storage full/private mode: memory cache still works */ }
}
function openMachineDetail(name) {
  const modal = $("#machine-detail-modal");
  if (!modal) return;
  const body = $("#machine-detail-body");
  const cached = _detailCacheGet(name);
  if (cached) {
    renderMachineDetail(body, cached);          // instant paint from cache
    body.classList.add("detail-refreshing");     // subtle "updating" hint
  } else {
    body.innerHTML = `<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i><span>Loading machine details…</span></div>`;
  }
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  api("GET", `/api/machines/${encodeURIComponent(name)}/detail`).then(d => {
    _detailCachePut(name, d);
    if ($("#machine-detail-modal")?.classList.contains("hidden")) return; // closed meanwhile
    renderMachineDetail(body, d);
  }).catch(e => {
    if (!body.classList.contains("detail-refreshing"))
      body.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    // fetch failed but cache shown: keep it, drop the hint
  }).finally(() => {
    body.classList.remove("detail-refreshing");
  });
}
function closeMachineDetail() {
  const modal = $("#machine-detail-modal");
  if (modal) { modal.classList.add("hidden"); modal.setAttribute("aria-hidden", "true"); }
}
function renderMachineDetail(body, d) {
  const s = d.stats || {};
  const online = s.status === "online";
  const gpus = s.gpus || [];
  const usedVram = gpus.reduce((a, g) => a + g.mem_used_gb, 0);
  const totalVram = gpus.reduce((a, g) => a + g.mem_total_gb, 0);
  const ramPct = s.ram_total_gb ? Math.round(s.ram_used_gb / s.ram_total_gb * 100) : 0;
  const vramPct = totalVram ? Math.round(usedVram / totalVram * 100) : 0;
  const osLabel = (d.os || "linux").charAt(0).toUpperCase() + (d.os || "linux").slice(1);
  const kv = (k, v) => `<div class="md-row"><span class="md-key">${k}</span><span class="md-val">${v || "—"}</span></div>`;
  const gpuRows = gpus.length ? gpus.map(g => `
    <tr>
      <td>${g.index}</td>
      <td class="gpu-name">${g.name}</td>
      <td class="gpu-util">${g.util}%
        <div class="mini-bar"><div class="mini-fill" style="width:${Math.min(g.util,100)}%; background:${valColor(g.util)}"></div></div>
      </td>
      <td class="gpu-vram">${g.mem_used_gb}/${g.mem_total_gb}GB
        <div class="mini-bar"><div class="mini-fill" style="width:${Math.min(g.mem_pct || (g.mem_total_gb ? g.mem_used_gb / g.mem_total_gb * 100 : 0),100)}%; background:${valColor(g.mem_pct || (g.mem_total_gb ? g.mem_used_gb / g.mem_total_gb * 100 : 0))}"></div></div>
      </td>
      <td class="gpu-temp ${g.temp >= 75 ? "hot" : ""}">${g.temp}°C</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">no GPUs</td></tr>`;
  const net = d.net || {};
  const disk = d.disk || {};
  // disk usage % from free/total strings ("86 GB", "1.2 TB")
  let diskPct = null;
  if (disk.free && disk.total) {
    const toGB = (s) => { const m = String(s).match(/([\d.]+)\s*(GB|TB)/i); if (!m) return null; const v = parseFloat(m[1]); return m[2].toUpperCase() === "TB" ? v * 1024 : v; };
    const freeGB = toGB(disk.free), totalGB = toGB(disk.total);
    if (freeGB != null && totalGB > 0) diskPct = Math.round((1 - freeGB / totalGB) * 100);
  }
  // DEC-16: engine-correct hero subtitle. Under llama_cpp verbatim
  // ("RPC up/down · role:"); under vllm RPC status is hidden-not-
  // translated — host facts + role only, same manifestHasSection idiom.
  const mdVllm = manifestHasSection("vllm_section");
  const mdHeroSub = mdVllm
    ? `${escHtml(d.ip)}${s.load ? ` · load ${escHtml(s.load)}` : ""}${d.role ? ` · role: ${escHtml(d.role)}` : ""}`
    : `${escHtml(d.ip)} · RPC <span class="${s.rpc_running ? "status-ok" : "status-warn"}">${s.rpc_running ? "up" : "down"}</span> · role: ${escHtml(d.role)}`;
  body.innerHTML = `
    <div class="md-hero">
      <div class="md-hero-title">
        <span class="status-dot ${online ? "on" : "off"}"></span>
        <strong>${escHtml(d.name)}</strong>
        ${d.is_master ? '<span class="badge badge-master">master</span>' : ""}
        ${d.is_local ? '<span class="badge badge-local">this machine</span>' : ""}
        <span class="os-badge">${osLabel}</span>
      </div>
      <div class="md-hero-sub">${mdHeroSub}</div>
    </div>
    <div class="md-grid">
      <div class="md-card">
        <h4><i class="fa-solid fa-microchip"></i> System</h4>
        ${kv("OS", osLabel)}
        ${kv("Kernel", d.kernel)}
        ${kv("CPU", d.cpu_model)}
        ${kv("Cores", s.cores)}
        ${kv("Uptime", d.uptime)}
        ${kv("llama.cpp", d.llama_version)}
        ${kv("llama bin", d.llama_bin)}
        <div class="md-row"><span class="md-key">update</span>
          <span class="md-val">
            <button class="small" id="md-update-llamacpp" data-mid="${escHtml(d.name)}">Update llama.cpp</button>
            <span class="hint" id="md-update-latest"></span>
          </span>
        </div>
      </div>
      <div class="md-card">
        <h4><i class="fa-solid fa-microchip"></i> CPU</h4>
        ${kv("Load", `${s.cpu_pct}%`)}
        ${kv("Cores", d.cores || s.cores)}
        ${kv("Load avg", s.load)}
        <div class="cpu-core-bars">
          ${(d.cpu_cores || s.cpu_cores || []).map((c, i) => `
            <div class="cpu-core" title="Core ${i}: ${c}%">
              <span class="cpu-core-label">${i}</span>
              <div class="cpu-core-bar"><div class="cpu-core-fill" style="width:${Math.min(c,100)}%; background:${valColor(c)}"></div></div>
            </div>`).join("") || '<span class="hint">no per-core data</span>'}
        </div>
      </div>
      <div class="md-card">
        <h4><i class="fa-solid fa-memory"></i> Memory</h4>
        ${kv("RAM", `${s.ram_used_gb}/${s.ram_total_gb} GB (${ramPct}%)`)}
        <div class="metric-bar"><div class="metric-fill" style="width:${ramPct}%; background:${valColor(ramPct)}"></div></div>
      </div>
      <div class="md-card">
        <h4><i class="fa-solid fa-network-wired"></i> Network</h4>
        ${kv("IP address", net.ip)}
        ${kv("MAC address", d.mac)}
        ${kv("Uplink speed", net.speed)}
      </div>
      <div class="md-card">
        <h4><i class="fa-solid fa-bolt"></i> RDMA Transport</h4>
        ${(() => {
          const r = d.rdma || {};
          const enabled = r.enabled;
          const badge = enabled
            ? '<span class="badge badge-rdma">RDMA</span>'
            : '<span class="badge">TCP</span>';
          let details = "";
          if (enabled) {
            details = kv("Mode", r.transport);
            if (r.ip) details += kv("RoCE IP", r.ip);
            if (r.dev) details += kv("Device", r.dev);
            if (r.gid) details += kv("GID index", r.gid);
          } else {
            details = kv("Mode", "TCP only (GGML_RPC_NO_RDMA)");
          }
          return badge + details;
        })()}
      </div>
      <div class="md-card">
        <h4><i class="fa-solid fa-hard-drive"></i> Disk</h4>
        ${kv("Free space", disk.free)}
        ${kv("Total", disk.total)}
        ${kv("Mount", disk.mount)}
        ${diskPct != null ? `<div class="metric-bar"><div class="metric-fill" style="width:${diskPct}%; background:${valColor(diskPct)}"></div></div>` : ""}
      </div>
      <div class="md-card md-card-wide">
        <h4><i class="fa-solid fa-microchip"></i> GPUs</h4>
        ${kv("VRAM total", `${usedVram.toFixed(1)}/${totalVram.toFixed(1)} GB (${vramPct}%)`)}
        <div class="metric-bar"><div class="metric-fill" style="width:${vramPct}%; background:${valColor(vramPct)}"></div></div>
        <table class="gpu-table">
          <thead><tr><th>Idx</th><th>GPU</th><th>Util</th><th>VRAM</th><th>Temp</th></tr></thead>
          <tbody>${gpuRows}</tbody>
        </table>
      </div>
    </div>`;
  mdUpdateBadge(d);   // version badge needs the latest tag — async fill
}

/* ---- per-machine llama.cpp update (machine detail) ----
   Badge: current llama_version build vs newest upstream release; the
   button runs a polled background job that stops the machine's llama
   processes first (Windows locks running exes), installs the newest
   prebuilt, and resurrects llama-swap when the machine hosts it. */
let _llamaLatestCache = { t: 0, v: null };
function _llamaLatest() {
  const now = Date.now();
  if (_llamaLatestCache.v && now - _llamaLatestCache.t < 300000)
    return Promise.resolve(_llamaLatestCache.v);
  return api("GET", "/api/llama-cpp-latest").then(r => {
    _llamaLatestCache = { t: now, v: r };
    return r;
  });
}
function mdUpdateBadge(d) {
  const el = $("#md-update-latest");
  const btn = $("#md-update-llamacpp");
  if (!el || !btn || btn.dataset.mid !== d.name) return;
  el.textContent = "";
  btn.disabled = false;
  btn.textContent = "Update llama.cpp";
  _llamaLatest().then(latest => {
    if (btn.dataset.mid !== d.name) return;    // modal re-rendered meanwhile
    if (latest.latest_build == null) {
      el.textContent = latest.detail ? "latest unknown" : "";
      return;
    }
    const cur = String(d.llama_version || "").match(/build\s+(\d+)/);
    const curN = cur ? parseInt(cur[1], 10) : null;
    if (curN == null) { el.textContent = `latest: ${latest.latest}`; return; }
    if (curN >= latest.latest_build) {
      el.textContent = `up to date (${latest.latest})`;
      btn.disabled = true;
    } else {
      el.innerHTML = `${curN} → ${latest.latest_build} <span class="status-warn">update available</span>`;
    }
  }).catch(() => { /* GitHub hiccup: badge stays empty, button still works */ });
}
/* Shared poll loop for the per-machine llama.cpp update job. setState
   receives "stopping"/"installing" as the job progresses; resolves with
   the finished job (state "done" or "failed"). Used by both the machine
   grid's inline Update button and the detail modal's. */
async function runLlamaCppUpdate(mid, setState = () => {}) {
  const start = await api("POST", `/api/machines/${encodeURIComponent(mid)}/update-llama-cpp`);
  let job = { state: "stopping", log: [] };
  for (;;) {
    job = await api("GET", start.status_url);
    if (["stopping", "installing", "updating"].includes(job.state)) {
      setState(job.state, job);
      await new Promise(r => setTimeout(r, 2000));
    } else {
      return job;
    }
  }
}
on("#machine-detail-body", "click", async (e) => {
  const btn = e.target.closest("#md-update-llamacpp");
  if (!btn) return;
  const mid = btn.dataset.mid;
  const hint = $("#md-update-latest");
  if (!mid) return;
  if (!window.confirm(`Update llama.cpp on ${mid}?\n\nSource builds are rebuilt in place (custom flags kept, running models keep serving); prebuilt installs stop that machine's llama processes first. Continue?`)) return;
  btn.disabled = true;
  try {
    const job = await runLlamaCppUpdate(mid, (state, j) => {
      btn.textContent = state === "stopping" ? "Stopping processes…"
        : state === "installing" ? "Downloading & installing…"
        : state === "updating" ? "Rebuilding from source…" : btn.textContent;
      if (hint) hint.textContent = (j.log || []).length
        ? String(j.log[j.log.length - 1]).slice(0, 90) : "";
    });
    if (job.state === "done") {
      btn.textContent = "Updated";
      toast(`llama.cpp updated on ${mid}`, "ok");
      setTimeout(() => openMachineDetail(mid), 1200);  // repaint version row
    } else {
      btn.textContent = "Update failed";
      if (hint) hint.innerHTML = `<span class="status-bad">${escHtml((job.result && job.result.detail) || "failed")}</span>`;
      toast(`llama.cpp update failed on ${mid}`, "bad");
      btn.disabled = false;
    }
  } catch (err) {
    btn.textContent = "Update llama.cpp";
    btn.disabled = false;
    if (hint) hint.innerHTML = `<span class="status-bad">${escHtml(err.message)}</span>`;
    toast(err.message, "bad");
  }
});
on("#machine-detail-backdrop", "click", closeMachineDetail);
on("#machine-detail-close", "click", closeMachineDetail);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#machine-detail-modal")?.classList.contains("hidden")) closeMachineDetail();
});

function renderGroupedStats(machines, groups) {
  const byName = Object.fromEntries(machines.map(m => [m.name, m]));
  const claimed = new Set();
  const container = $("#status-grid");
  const frag = document.createDocumentFragment();
  for (const g of groups || []) {
    const members = (g.machines || []).map(n => byName[n]).filter(Boolean);
    (g.machines || []).forEach(n => claimed.add(n));
    const onlineCount = members.filter(m => m.status === "online").length;
    const totVram = members.reduce((a, m) => a + (m.gpus || []).reduce((x, gpu) => x + gpu.mem_total_gb, 0), 0);
    const sub = `${onlineCount}/${members.length} online · ${totVram ? totVram.toFixed(0) + " GB VRAM · " : ""}${g.description || ""}`;
    const { sec, archBody } = makeArchSection(container, "dash:" + g.name, g.name, members.length);
    const grid = document.createElement("div");
    grid.className = "grid group-members";
    grid.innerHTML = members.map(statCardHTML).join("");
    archBody.appendChild(grid);
    // append the online/sub line under the header
    const subEl = document.createElement("div");
    subEl.className = "group-card-sub";
    subEl.textContent = sub;
    sec.querySelector(".arch-header").appendChild(subEl);
    frag.appendChild(sec);
  }
  const rest = machines.filter(m => !claimed.has(m.name));
  if (rest.length) {
    const sub = `${rest.filter(m => m.status === "online").length}/${rest.length} online`;
    const { sec, archBody } = makeArchSection(container, "dash:Ungrouped", "Ungrouped", rest.length);
    const grid = document.createElement("div");
    grid.className = "grid group-members";
    grid.innerHTML = rest.map(statCardHTML).join("");
    archBody.appendChild(grid);
    const subEl = document.createElement("div");
    subEl.className = "group-card-sub";
    subEl.textContent = sub;
    sec.querySelector(".arch-header").appendChild(subEl);
    frag.appendChild(sec);
  }
  return frag;
}

function loadingSpinner(label) {
  return `<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i><span>${escHtml(label || "Loading…")}</span></div>`;
}

// dashboard card click -> detail modal (delegated so it survives re-renders)
on("#status-grid", "click", (e) => {
  const card = e.target.closest(".machine-status-card");
  if (card && card.dataset.machine) openMachineDetail(card.dataset.machine);
});

let _loadStatusInFlight = false;
// last-good /api/stats payload: on the FIRST paint of a session (or after a
// failed poll) we render from this cache instantly, then swap in fresh
// data when the fetch lands — exactly what the overview totals already do.
let _lastStats = null;
try {
  const raw = sessionStorage.getItem("last-stats");
  if (raw) _lastStats = JSON.parse(raw);
} catch (_) { /* corrupt: cold start */ }
let _lastGroups = null;

async function loadStatus() {
  if (_loadStatusInFlight) return; // never pile up overlapping polls
  _loadStatusInFlight = true;
  try {
    await _loadStatusInner();
  } finally {
    _loadStatusInFlight = false;
  }
}

function _paintStats(r, groups) {
  const totals = r.totals;
  $("#machine-count").textContent = r.data.length;
  _paintTotals(totals);
  const grid = $("#status-grid");
  grid.innerHTML = "";
  grid.appendChild(renderGroupedStats(r.data, groups || []));
}

function _paintTotals(totals) {
  if (!totals) return;
  for (const k of ["vram", "ram"]) {
    const t = totals[k];
    if (!t) continue;
    $(`#ov-${k}-used`).textContent = `${t.used} GB`;
    $(`#ov-${k}-total`).textContent = `${t.total} GB`;
    $(`#ov-${k}-pct`).textContent = `${t.pct}%`;
    const bar = $(`#ov-${k}-bar`);
    const newWidth = `${Math.min(t.pct, 100)}%`;
    const newColor = barColor(t.pct);
    // update color class without touching 'bar' base class
    bar.classList.remove('green', 'yellow', 'red');
    bar.classList.add(newColor);
    // set width after class change to avoid transition fighting
    bar.style.width = newWidth;
  }
  $("#ov-cpu-pct").textContent = `${totals.cpu.pct}%`;
  $("#ov-cpu-sub").textContent = `${totals.cpu.busy} / ${totals.cpu.total} cores`;
  const cpuBar = $("#ov-cpu-bar");
  const cpuWidth = `${Math.min(totals.cpu.pct, 100)}%`;
  const cpuColor = barColor(totals.cpu.pct);
  cpuBar.classList.remove('green', 'yellow', 'red');
  cpuBar.classList.add(cpuColor);
  cpuBar.style.width = cpuWidth;
}

// per-card in-place refresh from the NDJSON stream: replace the one
// card's markup, keeping it inside its group section. Nothing else
// repaints — a slow host can't delay other cards' updates.
let _paintedGrouping = null;
function _upsertMachineCard(m) {
  // keep the streamed machine in the local cache for fallback/repaint
  if (!_streamedData[m.name] || JSON.stringify(_streamedData[m.name]) !== JSON.stringify(m))
    _streamedData[m.name] = m;
  const grid = $("#status-grid");
  if (!grid) return;
  const old = grid.querySelector(`.machine-status-card[data-machine="${CSS.escape(m.name)}"]`);
  if (!old) return;  // grouping not painted yet — full repaint covers it
  const tmp = document.createElement("div");
  tmp.innerHTML = statCardHTML(m);
  const fresh = tmp.firstElementChild;
  // bar animation: fresh nodes start at width 0 (or the OLD card's width)
  // and glide to the new value — otherwise every poll is a jump-cut.
  // Copy each old fill's width+color onto the fresh node, force a reflow
  // so the browser registers the starting state, THEN apply the new
  // widths. Works for metric bars, mini GPU bars, and the detail card's
  // cpu-core bars alike (any .metric-fill/.mini-fill/.cpu-core-fill).
  const oldFills = old.querySelectorAll(".metric-fill, .mini-fill, .cpu-core-fill");
  const freshFills = fresh.querySelectorAll(".metric-fill, .mini-fill, .cpu-core-fill");
  if (oldFills.length && oldFills.length === freshFills.length) {
    for (let i = 0; i < oldFills.length; i++) {
      freshFills[i].style.width = oldFills[i].style.width;
      freshFills[i].style.background = oldFills[i].style.background;
    }
  }
  old.replaceWith(fresh);
  // next frame: glide to the new values (transition triggers on change)
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const vals = cardBarValues(m);
    const fills = fresh.querySelectorAll(".metric-fill, .mini-fill, .cpu-core-fill");
    fills.forEach((f, i) => {
      const v = vals[i];
      if (!v) return;
      f.style.width = `${Math.min(v.pct, 100)}%`;
      f.style.background = v.color;
    });
  }));
}
// bar values in the same order the fills appear in statCardHTML.
// statCardHTML's GPU table has NO mini-bars (values-only cells), so this
// must list ONLY the three .metric-fill bars: CPU, RAM, VRAM. Pushing
// per-GPU values here shifted them onto CPU/RAM/VRAM fills — the card
// briefly rendered 0%-width bars after every stream upsert.
function cardBarValues(m) {
  const gpus = m.gpus || [];
  const vals = [];
  const usedVram = gpus.reduce((a, g) => a + g.mem_used_gb, 0);
  const totalVram = gpus.reduce((a, g) => a + g.mem_total_gb, 0);
  const vramPct = totalVram ? Math.round(usedVram / totalVram * 100) : 0;
  const ramPct = m.ram_total_gb ? Math.round(m.ram_used_gb / m.ram_total_gb * 100) : 0;
  vals.push({ pct: m.cpu_pct || 0, color: valColor(m.cpu_pct) });
  vals.push({ pct: ramPct, color: valColor(ramPct) });
  vals.push({ pct: vramPct, color: valColor(vramPct) });
  return vals;
}
const _streamedData = {};
function _streamedCards() {
  return Object.values(_streamedData);
}

async function _loadStatusInner() {
  // hidden tab: skip the whole poll — DOM churn with nothing watching
  if (document.hidden) return;
  const grid = $("#status-grid");
  // cold start with a cache: paint the last-good cards IMMEDIATELY, then
  // refresh in place when the fetch returns (no dead spinner, no flicker)
  if (_lastStats && !grid.querySelector(".arch-section") && !grid.querySelector(".machine-status-card"))
    _paintStats(_lastStats, _lastGroups);
  // only show the spinner when we have NOTHING to show
  else if (!grid.querySelector(".arch-section") && !grid.querySelector(".machine-status-card"))
    grid.innerHTML = loadingSpinner("Loading cluster status…");
  try {
    // Per-machine stream (NDJSON): each machine's line paints its card
    // THE MOMENT that machine's collection finishes — one slow or offline
    // host can no longer delay every other card. A final {totals} line
    // updates the overview bars. Falls back to the bulk endpoint if the
    // stream can't be read.
    const resp = await fetch("/api/stats/stream");
    if (!resp.ok || !resp.body) throw new Error("stream unavailable");
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let totals = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        let ev;
        try { ev = JSON.parse(line); } catch (_) { continue; }
        if (ev.totals) { totals = ev.totals; continue; }
        if (ev.name) _upsertMachineCard(ev);
      }
    }
    $("#machine-count").textContent = _streamedCards().length;
    // update the persisted cache from the streamed data
    if (totals) {
      _lastStats = { data: _streamedCards(), totals };
      try { sessionStorage.setItem("last-stats", JSON.stringify(_lastStats)); } catch (_) {}
    }
    const groups = await api("GET", "/api/groups").catch(() => []);
    _lastGroups = groups;
    // full repaint only when grouping changed (per-card updates above
    // already refreshed each card in place)
    const wanted = (groups || []).map(g => g.name).join(",");
    if (wanted !== _paintedGrouping) {
      _paintedGrouping = wanted;
      if (_lastStats) _paintStats(_lastStats, groups);
    }
    if (totals) _paintTotals(totals);
  } catch (e) {
    // stream failed (proxy, older backend): fall back to the bulk endpoint
    try {
      const r = await api("GET", "/api/stats");
      _lastStats = r;
      try { sessionStorage.setItem("last-stats", JSON.stringify(r)); } catch (_) {}
      const groups = await api("GET", "/api/groups").catch(() => []);
      _lastGroups = groups;
      _paintedGrouping = (groups || []).map(g => g.name).join(",");
      _paintStats(r, groups);
    } catch (e2) {
      const grid2 = $("#status-grid");
      if (!grid2.querySelector(".machine-status-card"))
        grid2.innerHTML = `<p class="status-bad">${e2.message}</p>`;
      else
        dashStatus(`stats refresh failed: ${e2.message}`);
    }
  }
  try {
    const s = await api("GET", "/api/swap/status");
    const pill = $("#swap-pill");
    if (pill) pill.innerHTML = `<span class="status-dot ${s.running ? "on" : "off"}"></span> swap: ${s.running ? `up (${s.host}:${s.port})` : "down"}`;
  } catch (e) { /* stats already rendered; swap pill keeps last state */ }
  try {
    const l = await api("GET", "/api/swap/loaded");
    const banner = $("#ov-loaded-model");
    if (!banner) return;
    if (l.loaded && l.loaded.length) {
      banner.className = "loaded-model-banner";
      banner.textContent = `▶ ${l.loaded.join(" + ")}`;
    } else if (l.running) {
      banner.className = "loaded-model-banner idle";
      banner.innerHTML = `<i class="fa-solid fa-cube banner-icon"></i> no model loaded`;
    } else {
      banner.className = "loaded-model-banner down";
      banner.innerHTML = `<i class="fa-solid fa-cube banner-icon"></i> llama-swap down`;
    }
  } catch (e) { /* banner keeps last state */ }
}

let statsTimer = null;
let autoRefreshOn = () => localStorage.getItem("auto-refresh") !== "0";
function paintRefreshChip() {
  const chip = $("#auto-refresh-chip");
  if (chip) {
    chip.setAttribute("aria-pressed", String(autoRefreshOn()));
    // spin the knob only while the poll loop actually runs
    chip.classList.toggle("spinning", autoRefreshOn() && statsTimer !== null);
  }
}
function startAutoRefresh() {
  clearInterval(statsTimer);
  statsTimer = null;
  if (autoRefreshOn()) statsTimer = setInterval(loadStatus, 1000);
  paintRefreshChip();
  loadStatus();
}
// pause ALL polling while the tab is hidden — background tabs kept
// rebuilding DOM + holding WebSockets alive forever, which is how a
// WebUI tab ballooned to 64GB on winbox. Resume on visibility.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearInterval(statsTimer);
    statsTimer = null;
  } else {
    startAutoRefresh();
  }
});
function stopAutoRefresh() {
  clearInterval(statsTimer);
  statsTimer = null;
  paintRefreshChip();
}
on("#auto-refresh-chip", "click", () => {
  localStorage.setItem("auto-refresh", autoRefreshOn() ? "0" : "1");
  startAutoRefresh();
});

/* Two-click arm (no native confirm()/alert() — those get blocked or
   suppressed and leave dead buttons with zero feedback). First click
   arms, second click within 6s executes. Result goes inline. */
function dashStatus(msg) {
  console.log("[ui]", msg);
  const el = $("#dash-action-status");
  if (el) el.textContent = msg;
}
function armed(btn, label) {
  if (!btn.dataset.armed) {
    btn.dataset.armed = "1";
    btn.textContent = "Sure?";
    dashStatus(`${label}: click again to confirm`);
    setTimeout(() => {
      delete btn.dataset.armed;
      if (!btn.disabled) btn.textContent = label;
    }, 6000);
    return false;
  }
  delete btn.dataset.armed;
  return true;
}

on("#stop-swap", "click", async (e) => {
  const btn = e.currentTarget;
  dashStatus("stop-all clicked");
  if (!armed(btn, "Stop All")) return;
  btn.disabled = true; btn.textContent = "stopping…";
  dashStatus("stopping llama-swap + all RPC…");
  try {
    const r = await api("POST", "/api/swap/stop", {});
    dashStatus(`swap stop response: ${JSON.stringify(r)}`);
    let msg = `${r.host}: ${r.stopped ? "stopped" : "STILL RUNNING"}`;
    // G-3: rpc-server cleanup is a llama.cpp-only concept — under vLLM
    // there are no ggml-rpc-server processes; skip the leg entirely.
    if ((appSettings && appSettings.engine) === "vllm") {
      msg += " || rpc: n/a (vLLM mode)";
    } else {
      try {
        const q = await api("POST", "/api/rpc/stop-all");
        dashStatus(`rpc stop-all response: ${JSON.stringify(q)}`);
        const failed = q.filter(x => !x.killed).map(x => x.machine);
        msg += failed.length
          ? ` || rpc FAILED on: ${failed.join(", ")}`
          : ` || rpc: ${q.map(x => `${x.machine}: cleared`).join(" | ")}`;
      } catch (e2) { msg += ` || rpc stop failed: ${e2.message}`; }
    }
    dashStatus(msg);
  } catch (err) {
    console.error("[ui] stop-all failed:", err);
    dashStatus(`Stop All failed: ${err.message}`);
  }
  btn.disabled = false; btn.textContent = "Stop All";
  loadStatus();
});

on("#restart-app", "click", async (e) => {
  const btn = e.currentTarget;
  dashStatus("restart-app clicked");
  if (!armed(btn, "Restart App")) return;
  btn.disabled = true; btn.textContent = "restarting…";
  dashStatus("stopping llama-swap + RPC, restarting web app (3s)…");
  try {
    const r = await api("POST", "/api/restart", {});
    dashStatus(`${r.message} — page will reload in 3s`);
    setTimeout(() => location.reload(), 3500);
  } catch (err) {
    console.error("[ui] restart-app failed:", err);
    dashStatus(`Restart failed: ${err.message}`);
    btn.disabled = false; btn.textContent = "Restart App";
  }
});

on("#start-all", "click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true; btn.dataset.busy = "1"; btn.textContent = "starting…";
  dashStatus("starting RPC on all machines + llama-swap (takes a minute+)…");
  try {
    const r = await api("POST", "/api/start-all");
    dashStatus(`start-all raw: ${JSON.stringify(r)}`);
    // G-3: under vLLM the backend skips the RPC phase and reports
    // rpc: [{skipped, reason}] — show an honest summary, not "DOWN".
    const rpcPart = (r.rpc || []).length === 1 && r.rpc[0].skipped
      ? r.rpc[0].reason
      : (r.rpc || [])
          .map(x => `${x.machine}: ${x.running ? `up (:${x.port})` : "DOWN"}`)
          .join(" | ");
    const sw = r.swap || {};
    dashStatus(`rpc: ${rpcPart} || swap: `
      + (sw.failed ? `FAILED (${sw.error || "see Launch tab"})`
                   : `live on :${sw.port} (${sw.host})`));
  } catch (err) {
    console.error("[ui] start-all failed:", err);
    dashStatus(`Start All failed: ${err.message}`);
  }
  // FIX-A7: restore the ENGINE-AWARE label, not a hardcoded "Start All" —
  // under vLLM the button reads "Start llama-swap" (applyEngineHints set it).
  // While the daemon is up the button stays disabled (re-launch would pile
  // up another instance) — sync from live status, not from intent.
  delete btn.dataset.busy;
  const engine = (appSettings && appSettings.engine) || "llama_cpp";
  btn.textContent = engine === "vllm" ? "Start llama-swap" : "Start All";
  try {
    const st = await api("GET", "/api/swap/status");
    syncStartAll(st);
    if (!st.running) { btn.disabled = false; }
    else btn.textContent = engine === "vllm" ? "Start llama-swap" : "Start All";
  } catch (_) { btn.disabled = false; }
  loadStatus();
});

/* ---------------- machines ---------------- */
const EMPTY_MACHINE = { name: "", ip: "", username: "zazi", password: "", port: 22, role: "peer", os: "linux", rpc_port: 50052, gpu: "", llama_bin: "", rdma: false, rdma_ip: "", rdma_dev: "", rdma_gid: "", rdma_cm: false };

async function defaultLlamaBin() {
  const s = await getAppSettings();
  return s.llama_bin || "~/llama.cpp/build/bin";
}

function machineFormLoad(m, isEdit) {
  // isEdit=true only when opening a SAVED machine via Edit. Discovery fills
  // (discoveredFill) also carry a suggested name but are still a new machine —
  // conflating the two hid the wizard on step 3 (the reported bug).
  // Resets the edit-wizard override; callers that need it set it afterwards.
  wizForceWizard = false;
  const editing = isEdit ?? Boolean(m.name);
  $("#machine-original-name").value = editing ? (m.name || "") : "";
  $("#m-name").value = m.name || "";
  $("#m-ip").value = m.ip || "";
  $("#m-username").value = m.username || "zazi";
  $("#m-password").value = ""; // never fill the real password — hint shows state
  const pwHint = $("#m-password-hint");
  if (pwHint) pwHint.textContent = (editing && m.password_set)
    ? "Password saved in vault — leave blank to keep, type to replace."
    : (editing ? "No password saved — type one to set it." : "Stored encrypted in the vault (.env).");
  $("#m-port").value = m.port ?? 22;
  $("#m-role").value = m.role || "peer";
  $("#m-os").value = m.os || "linux";
  $("#m-gpu-vendor").value = m.gpu_vendor || "nvidia";
  $("#m-rpc-port").value = m.rpc_port ?? 50052;
  $("#m-gpu").value = m.gpu || "";
  $("#m-ram").value = m.ram_total_gb || "";
  $("#m-llama-bin").value = m.llama_bin || "";
  // hint shows global default when no override — machine-specific wins when set
  defaultLlamaBin().then(d => {
    const hint = $("#m-llama-bin-hint");
    if (hint) hint.textContent = m.llama_bin ? `override — will use this, not global (${d})` : `empty → inherits global default: ${d}`;
    if (!m.llama_bin) $("#m-llama-bin").placeholder = d + " (global default)";
    else $("#m-llama-bin").placeholder = d + " (global)";
  });
  const ex = m.expose || {};
  $("#m-expose-mode").value = ex.mode || "both";
  $("#m-expose-gpus").value = (ex.gpus || []).join(",");
  const avail = (m.gpu || "").split(",").filter(Boolean);
  $("#m-expose-gpus").placeholder = avail.length ? `available: ${avail.join(",")}` : "all (not detected)";
  $("#m-rdma").checked = Boolean(m.rdma);
  $("#m-rdma-ip").value = m.rdma_ip || "";
  $("#m-rdma-ip").disabled = !$("#m-rdma").checked;
  $("#m-rdma-dev").value = m.rdma_dev || "";
  $("#m-rdma-gid").value = m.rdma_gid || "";
  $("#m-rdma-cm").checked = Boolean(m.rdma_cm);
  $("#m-rdma-dev").disabled = !$("#m-rdma").checked;
  $("#m-rdma-gid").disabled = !$("#m-rdma").checked;
  $("#m-rdma-cm").disabled = !$("#m-rdma").checked;
  $("#machine-cancel").hidden = !editing;
  // add mode = 3-step wizard (Scan → Select → Configure), one step at a
  // time; edit mode skips the wizard and shows Configure directly
  wizShow(editing ? 3 : 1);
  if (!editing) {
    discoverInit();
  }
}

/* ---- add-machine wizard: Scan → Select → Configure ---- */
let wizStep = 1;
let wizSelection = null;  // ip picked in step 2 (row click or Select →)
let wizForceWizard = false;  // edit flow: keep wizard chrome visible even with original-name set
function wizEditing() { return Boolean($("#machine-original-name").value) && !wizForceWizard; }
function wizShow(n) {
  wizStep = Math.min(3, Math.max(1, n));
  const editing = wizEditing();
  for (const s of [1, 2, 3]) {
    const el = $("#wiz-step-" + s);
    if (el) el.hidden = editing ? (s !== 3) : (s !== wizStep);
  }
  document.querySelectorAll("#wiz-indicator li").forEach(li => {
    const s = parseInt(li.dataset.s, 10);
    li.classList.toggle("active", !editing && s === wizStep);
    li.classList.toggle("done", !editing && s < wizStep);
  });
  $("#wiz-indicator").style.display = editing ? "none" : "";
  $("#wiz-nav").style.display = editing ? "none" : "";
  const editBar = $("#machine-edit-actions");
  if (editBar) editBar.hidden = !editing;
  $("#wiz-back").disabled = wizStep === 1;
  // Continue on the last step acts as Save (the separate Save button is retired)
  $("#wiz-next").hidden = false;
  $("#wiz-next").textContent = wizStep === 3 ? "Save ✓" : "Continue →";
  $("#wiz-save").hidden = true;
  const origName = $("#machine-original-name").value;
  if (!origName) $("#machine-form-title").textContent = `Add machine — step ${wizStep} of 3`;
  else if (editing) $("#machine-form-title").textContent = `Edit ${origName}`;
  else $("#machine-form-title").textContent = `Edit ${origName} — step ${wizStep} of 3`;
  // fingerprint check every time Configure is reached (new machine: TOFU
  // notice; known host: shows pinned key + mismatch warning)
  if (wizStep === 3) showHostFingerprint();
}
function wizClearPick() {
  wizSelection = null;
  document.querySelectorAll("#d-results tr.wiz-selected").forEach(tr => tr.classList.remove("wiz-selected"));
  const man = $("#wiz-manual-ip");
  if (man) man.value = "";
}
function wizPickedIp() {
  if (wizSelection) return wizSelection;
  const man = ($("#wiz-manual-ip") || {}).value;
  return (man || "").trim();
}
function wizNext() {
  if (wizEditing()) return;
  if (wizStep >= 3) { $("#machine-form").requestSubmit(); return; }
  if (wizStep === 1) { wizShow(2); return; }
  // step 2 → 3 needs one machine: a selected row or a manual IP
  // (edit flow with no new pick just keeps the loaded values)
  const ip = wizPickedIp();
  if (!ip) {
    if ($("#machine-original-name").value) { wizShow(3); return; }
    toast("Select a discovered machine or enter an IP manually", "bad"); return;
  }
  const h = DISCOVERED[ip];
  if (h) { discoveredFill(h); return; }  // fills form + advances to 3
  $("#m-ip").value = ip;
  wizShow(3);
  toast("Verify the details, then Save (Save verifies SSH before storing)", "ok");
}
on("#wiz-next", "click", wizNext);
on("#wiz-back", "click", () => { if (!wizEditing() && wizStep > 1) wizShow(wizStep - 1); });
on("#wiz-cancel", "click", () => { wizClearPick(); machineFormLoad(EMPTY_MACHINE, false); });

/* ---- SSH host-key fingerprint (first-time login handling) ----
   On entering Configure for a NEW machine, fetch the host key pre-auth
   and show it. First contact = trusted on first use (stored server-side);
   later visits show the stored fingerprint and warn on mismatch. */
async function showHostFingerprint() {
  const box = $("#m-fingerprint");
  if (!box) return;
  const ip = $("#m-ip").value.trim();
  if (!ip) { box.textContent = ""; return; }
  const port = parseInt($("#m-port").value, 10) || 22;
  box.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> checking host key…';
  try {
    const r = await api("POST", "/api/machines/fingerprint",
      { ip, port, username: $("#m-username").value.trim() });
    const mismatch = r.known && r.stored_fingerprint && r.stored_fingerprint !== r.fingerprint;
    box.innerHTML = mismatch
      ? `<span class="status-bad"><i class="fa-solid fa-triangle-exclamation"></i> Host key CHANGED for ${escHtml(ip)}! stored: <code>${escHtml(r.stored_fingerprint)}</code> · now: <code>${escHtml(r.fingerprint)}</code> — verify before trusting</span>`
      : r.known
        ? `<i class="fa-solid fa-fingerprint"></i> Host key (trusted): <code>${escHtml(r.fingerprint)}</code> <small class="hint">(${escHtml(r.type)}, saved on first login)</small>`
        : `<i class="fa-solid fa-fingerprint"></i> First login — host key <code>${escHtml(r.fingerprint)}</code> <small class="hint">(${escHtml(r.type)}) will be trusted on save and pinned for future logins</small>`;
  } catch (e) {
    box.innerHTML = `<span class="status-warn"><i class="fa-solid fa-triangle-exclamation"></i> Host key check failed: ${escHtml(e.message)}</span>`;
  }
}

function machineFormRead() {
  return {
    name: $("#m-name").value.trim(),
    ip: $("#m-ip").value.trim(),
    username: $("#m-username").value.trim(),
    password: $("#m-password").value,
    port: parseInt($("#m-port").value, 10),
    role: $("#m-role").value,
    os: $("#m-os").value,
    gpu_vendor: $("#m-gpu-vendor").value,
    rpc_port: parseInt($("#m-rpc-port").value, 10),
    gpu: $("#m-gpu").value.trim(),
    ram_total_gb: parseFloat($("#m-ram").value) || null,
    llama_bin: $("#m-llama-bin").value.trim(),
    expose: {
      mode: $("#m-expose-mode").value,
      gpus: $("#m-expose-gpus").value.split(",").map(s => s.trim()).filter(Boolean),
    },
    rdma: $("#m-rdma").checked,
    rdma_ip: $("#m-rdma-ip").value.trim(),
    rdma_dev: $("#m-rdma-dev").value.trim(),
    rdma_gid: $("#m-rdma-gid").value.trim(),
    rdma_cm: $("#m-rdma-cm").checked,
  };
}

async function loadMachines() {
  const body = $("#machines-list");
  if (body && !body.querySelector(".arch-section") && !body.querySelector(".scan-model"))
    body.innerHTML = loadingSpinner("Loading machines…");
  const machines = await api("GET", "/api/machines");
  let groups = [];
  try { groups = await api("GET", "/api/groups"); } catch (_) {}
  body.innerHTML = "";
  // build machine -> groups map
  const machineGroups = new Map();
  for (const g of groups) {
    for (const mn of (g.machines || [])) {
      if (!machineGroups.has(mn)) machineGroups.set(mn, []);
      machineGroups.get(mn).push(g.name);
    }
  }
  // group machines by their group membership; ungrouped go in their own section
  const sections = new Map();  // sectionTitle -> [machines]
  for (const m of machines) {
    const gs = machineGroups.get(m.name) || [];
    if (gs.length) {
      for (const g of gs) {
        if (!sections.has(g)) sections.set(g, []);
        sections.get(g).push(m);
      }
    } else {
      if (!sections.has("Ungrouped")) sections.set("Ungrouped", []);
      sections.get("Ungrouped").push(m);
    }
  }
  const sortedTitles = Array.from(sections.keys()).sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase()));
  for (const title of sortedTitles) {
    const secMachines = sections.get(title).sort((a, b) =>
      (a.name || "").toLowerCase().localeCompare((b.name || "").toLowerCase()));
    const { sec, archBody: secBody } = makeArchSection(body, "machines:" + title, title, secMachines.length);
    for (const m of secMachines) {
      // DEC-17: under vllm, machines the server declares vLLM-incapable
      // (vllm_capable === false from /api/machines — B14, no client-side
      // OS re-derivation) render dimmed with a "not available under
      // vLLM" line. Under llama_cpp this is dead code — full function.
      const isVllm = (appSettings && appSettings.engine) === "vllm";
      const vllmBlind = isVllm && m.vllm_capable === false;
      const vllmReason = m.vllm_capability_reason || "vLLM is not usable on this host";
      const tr = document.createElement("div");
      tr.innerHTML = `
      <div class="scan-main${vllmBlind ? " vllm-unavailable" : ""}"><div class="scan-title">${osIcon(m.os)}${m.name}${m.role === "master" ? ' <span class="badge badge-master">master</span>' : ""}</div>
        <small>${m.ip} · ${m.role} (${m.os}) · GPUs: ${m.gpu || "-"} · ${exposeHTML(m)}</small>
        ${detectedHTML(m)}
        ${vllmBlind ? `<span class="vllm-unavailable-tag"><i class="fa-solid fa-ban"></i> not available under vLLM — ${escHtml(vllmReason)}</span>` : ""}</div>
      <div class="scan-model-actions">
        <button class="small" data-act="edit">Edit</button>
        <button class="small" data-act="detect">Detect</button>
        <button class="small" data-act="update">Update</button>
        <button class="small danger" data-act="delete">Delete</button>
      </div>`;
      tr.classList.add("scan-model");
      tr.querySelector('[data-act="update"]').onclick = async (e) => {
        const btn = e.currentTarget;
        if (!window.confirm(`Update llama.cpp on ${m.name}?\n\nSource builds are rebuilt in place (custom flags kept, running models keep serving); prebuilt installs stop that machine's llama processes first. Continue?`)) return;
        btn.disabled = true; btn.textContent = "Starting…";
        try {
          const job = await runLlamaCppUpdate(m.name, (state) => {
            btn.textContent = state === "stopping" ? "Stopping…"
              : state === "installing" ? "Installing…"
              : state === "updating" ? "Rebuilding…" : btn.textContent;
          });
          if (job.state === "done") {
            btn.textContent = "Updated";
            toast(`llama.cpp updated on ${m.name}`, "ok");
          } else {
            btn.textContent = "Failed";
            toast(`llama.cpp update failed on ${m.name}: ${(job.result && job.result.detail) || "failed"}`, "bad");
            btn.disabled = false;
          }
        } catch (err) {
          btn.textContent = "Update"; btn.disabled = false;
          toast(err.message, "bad");
        }
      };
      tr.querySelector('[data-act="detect"]').onclick = async (e) => {
        const btn = e.target;
        btn.disabled = true; btn.textContent = "detecting...";
        try {
          await api("POST", `/api/machines/${m.name}/detect`);
          loadMachines();
        } catch (err) { toast(`detect failed on ${m.name}: ${err.message}`, "bad"); btn.disabled = false; btn.textContent = "Detect"; }
      };
      tr.querySelector('[data-act="delete"]').onclick = async (e) => {
        const btn = e.currentTarget;
        if (!armed(btn, "Delete")) return;
        btn.disabled = true;
        try {
          await api("DELETE", `/api/machines/${encodeURIComponent(m.name)}`);
          toast(`Deleted machine ${m.name}`, "ok");
          loadMachines();
        } catch (err) { toast(`Delete failed: ${err.message}`, "bad"); btn.disabled = false; btn.textContent = "Delete"; }
      };
      tr.querySelector('[data-act="edit"]').onclick = () => {
        // edit walks the same Scan → Select → Configure wizard, opening at
        // Select with the saved config loaded; Save PUTs to the original name
        machineFormLoad(m, true);
        wizForceWizard = true;
        wizClearPick();
        discoverInit();
        wizShow(2);
        document.querySelector("#machine-form").scrollIntoView({ behavior: "smooth", block: "start" });
      };
      secBody.appendChild(tr);
      fillInlineUpdateState(tr, m.name);
    }
    body.appendChild(sec);
  }
}

/* Inline Update button state: when the machine is already on the newest
   upstream release, the button becomes a disabled 'on latest' chip (the
   update job would short-circuit anyway); behind latest it shows the
   current→latest build jump it will perform. Falls back to a plain
   Update button when the version can't be probed (offline machine). */
function fillInlineUpdateState(tr, name) {
  const btn = tr.querySelector('[data-act="update"]');
  if (!btn) return;
  api("GET", `/api/machines/${encodeURIComponent(name)}/llama-cpp-version`)
    .then(v => {
      if (!btn.isConnected) return;          // grid re-rendered meanwhile
      if (v.on_latest) {
        btn.disabled = true;
        btn.textContent = `on latest (${v.latest})`;
        btn.title = "already on the newest llama.cpp release";
      } else if (v.build != null && v.latest_build != null && v.build < v.latest_build) {
        btn.textContent = `Update ${v.build}→${v.latest_build}`;
        btn.title = `current build ${v.build}, latest ${v.latest} (${v.latest_build})`;
      }
    }).catch(() => { /* unprobed/offline: keep plain Update */ });
}

function detectedHTML(m) {
  const d = m.detected;
  if (!d) return '<span class="status-warn">not detected</span>';
  return `<div class="detected">
    <div>${(d.gpu_names || []).length} GPU(s), ${d.vram_total_gb} GB VRAM</div>
    <small>${(d.gpu_names || []).join(", ")}</small>
    <small>${d.cores} cores, ${d.ram_total_gb} GB RAM</small>
  </div>`;
}

function exposeHTML(m) {
  const ex = m.expose || {};
  const mode = ex.mode || "both";
  const gpus = ex.gpus || [];
  const gpuNote = gpus.length ? ` (${gpus.join(",")})` : " (all)";
  return `${mode}${gpuNote}`;
}

on("#detect-all-btn", "click", async (e) => {
  const btn = e.target;
  btn.disabled = true; btn.textContent = "Detecting...";
  try {
    const r = await api("POST", "/api/machines/detect-all");
    const n = (r.synced_models || []).length;
    toast(`Detected hardware on: ${(r.updated || []).join(", ") || "none (all offline?)"}`
      + (n ? ` — synced ${n} model config${n === 1 ? "" : "s"} to new hardware` : ""), "ok");
    loadMachines(); loadModels();
  } catch (err) { toast(err.message, "bad"); }
  btn.disabled = false; btn.textContent = "Detect Hardware (all machines)";
});

/* ---------------- network discovery (inside Add-machine form) ---------------- */
const DISCOVERED = {};   // ip -> discovery result entry

async function discoverInit() {
  try {
    const r = await api("GET", "/api/discover/subnets");
    const sel = $("#d-subnet");
    if (!sel.options.length)
      sel.innerHTML = (r.subnets || []).map(s =>
        `<option value="${escHtml(s)}"${s === r.default ? " selected" : ""}>${escHtml(s)}${s === r.default ? " (this network)" : ""}</option>`
      ).join("");
  } catch (_) { /* dropdown stays empty; scan will error visibly */ }
}

function gpuPrefix(vendor) {
  const v = (vendor || "").toLowerCase();
  return v === "amd" ? "ROCm" : v === "apple" ? "Metal" : v === "intel" ? "SYCL" : "CUDA";
}
function osIcon(os) {
  const o = (os || "").toLowerCase();
  if (o === "windows") return '<i class="fa-brands fa-windows os-icon" title="Windows"></i>';
  if (o === "macos" || o === "darwin") return '<i class="fa-brands fa-apple os-icon" title="macOS"></i>';
  if (o === "linux") return '<i class="fa-brands fa-linux os-icon" title="Linux"></i>';
  return '<i class="fa-solid fa-server os-icon" title="Unknown OS"></i>';
}
// model icon cache (permanent, backed by config/model_icons.json)
const modelIconCache = {};
// gguf basename -> model_id (built in loadModels) so scan rows reuse
// configured models' icons instead of resolving separately
let scanFileIconKey = {};
// load all cached icons once at boot so they're reused everywhere
// (backend sanitizes; proxiedImgSrc also strips legacy '&quot;...' pollution per use)
api("GET", "/api/models/icons").then(icons => {
  for (const [k, v] of Object.entries(icons || {}))
    modelIconCache[k] = typeof v === "string" ? v : "";
}).catch(() => {});
// Route external image URLs through the backend's same-origin image proxy —
// HF CDN avatars sometimes trip OpaqueResponseBlocking in the browser.
function proxiedImgSrc(url) {
  if (typeof url !== "string") return "";
  url = url.replace(/&amp;/g, "&").trim();
  // strip legacy polluted cache values: URL + '&quot;,...' JSON leakage
  const cut = url.search(/&quot;|["'<\s]/);
  if (cut !== -1) url = url.slice(0, cut);
  if (!url) return "";
  if (!/^https?:\/\//i.test(url)) {
    if (url.startsWith("/") || url.startsWith(`${location.origin}/`)) return url;
    return ""; // never hand non-http(s) values (objects, data:, file:) to <img>
  }
  return `/img-proxy?url=${encodeURIComponent(url)}`;
}
function modelIconHTML(modelId) {
  const url = proxiedImgSrc(modelIconCache[modelId]);
  if (url) return `<img class="model-avatar" src="${escHtml(url)}" alt="" loading="lazy" onerror="this.outerHTML='<span class=&quot;model-avatar model-avatar-ph&quot; data-model=&quot;${escHtml(modelId)}&quot;><i class=&quot;fa-solid fa-cube&quot;></i></span>'">`;
  // placeholder until resolved
  return `<span class="model-avatar model-avatar-ph" data-model="${escHtml(modelId)}"><i class="fa-solid fa-cube"></i></span>`;
}
async function resolveModelIcon(modelId) {
  if (modelIconCache[modelId]) return;
  try {
    const r = await api("GET", `/api/models/${encodeURIComponent(modelId)}/icon`);
    modelIconCache[modelId] = r.icon;
    // swap any placeholder for this model (empty icon -> keep the cube)
    if (r.icon) {
      document.querySelectorAll(`.model-avatar-ph[data-model="${CSS.escape(modelId)}"]`).forEach(el => {
        el.outerHTML = `<img class="model-avatar" src="${escHtml(proxiedImgSrc(r.icon))}" alt="" loading="lazy" onerror="this.outerHTML='<span class=&quot;model-avatar model-avatar-ph&quot; data-model=&quot;${escHtml(modelId)}&quot;><i class=&quot;fa-solid fa-cube&quot;></i></span>'">`;
      });
    }
  } catch (_) { /* leave placeholder */ }
}

// icon for a SCAN row (unconfigured GGUF file). Reuses the icon of any
// configured model that points at the same file (instant), else resolves
// via the ?path= endpoint (cached server-side under file:<basename> so
// the whole quant/shard family shares one entry + one HF lookup).
function scanModelIconHTML(basePath) {
  const fname = (basePath || "").split("/").pop();
  // configured model with the same file? reuse its icon key
  const byFile = scanFileIconKey[fname];
  if (byFile && modelIconCache[byFile]) return modelIconHTML(byFile);
  const key = `file:${fname}`;
  if (modelIconCache[key] !== undefined) return modelIconHTML(key);
  // unknown yet: placeholder now, resolve async then swap placeholders
  modelIconCache[key] = ""; // mark in-progress so we don't re-request
  api("GET", `/api/models/${encodeURIComponent("file:" + fname)}/icon?path=${encodeURIComponent(basePath)}`)
    .then(r => {
      modelIconCache[key] = r.icon || "";
      if (r.icon) {
        document.querySelectorAll(`.model-avatar-ph[data-model="${CSS.escape(key)}"]`).forEach(el => {
          el.outerHTML = `<img class="model-avatar" src="${escHtml(proxiedImgSrc(r.icon))}" alt="" loading="lazy" onerror="this.outerHTML='<span class=&quot;model-avatar model-avatar-ph&quot; data-model=&quot;${escHtml(key)}&quot;><i class=&quot;fa-solid fa-cube&quot;></i></span>'">`;
        });
      }
    }).catch(() => {});
  return modelIconHTML(key);
}

function discoveredFill(h) {
  // preserve an in-progress edit (target + wizard mode) across the refill
  const orig = $("#machine-original-name").value;
  const fw = wizForceWizard;
  const pref = gpuPrefix(h.gpu_vendor);
  machineFormLoad({
    name: h.hostname || h.ip.replace(/\./g, "-"),
    ip: h.ip,
    username: h.username || $("#m-username").value.trim() || "zazi",
    password: $("#m-password").value,
    port: h.port || 22,
    os: h.os || "linux",
    gpu_vendor: h.gpu_vendor || "nvidia",
    gpu: (h.gpus || []).length ? h.gpus.map((g, i) => `${pref}${i}`).join(",") : "",
  }, false);  // suggested name, but still a NEW machine — keep wizard visible
  if (orig) $("#machine-original-name").value = orig;
  wizForceWizard = fw;
  wizClearPick();
  wizShow(3);
  toast("Review the details, then Save (Save verifies SSH before storing)", "ok");
}
// IP / SSH port edits change which host we fingerprint — re-check live
on("#m-ip", "input", () => {
  const box = $("#m-fingerprint");
  if (box) box.textContent = "";
});
on("#m-port", "change", () => { if (wizStep === 3) showHostFingerprint(); });

on("#d-scan", "click", async (e) => {
  const btn = e.currentTarget;
  const res = $("#d-results");
  btn.disabled = true; btn.textContent = "scanning… (~1 min)";
  res.innerHTML = '<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Pinging subnet, sweeping for SSH port, grabbing banners…</p>';
  let scanned = false;
  try {
    const r = await api("POST", "/api/discover", {
      subnet: $("#d-subnet").value,
      port: parseInt($("#d-port").value, 10) || 22,
      username: $("#m-username").value.trim(),
      password: $("#m-password").value,
      ping: true,
    });
    r.results.forEach(h => DISCOVERED[h.ip] = h);
    $("#discovered-ips").innerHTML = r.results.map(h => `<option value="${escHtml(h.ip)}">`).join("");
    scanned = true;
    if (!r.results.length) {
      res.innerHTML = `<p class="hint">Nothing identifiable on ${escHtml(r.subnet)} — ${r.alive || 0} alive, ${r.ssh_hosts || 0} with SSH port ${r.port}${r.unknown ? `, ${r.unknown} without a recognizable banner (hidden)` : ""}.</p>`;
    } else {
      res.innerHTML = `<p class="hint">${escHtml(r.subnet)}: ${r.alive} alive, ${r.ssh_hosts} with SSH port ${r.port} — ${r.results.length} candidate(s)${r.unknown ? `, ${r.unknown} unknown hidden` : ""}. Click a row to select it, then Continue — or Select → to jump straight to Configure.</p>` +
        '<table><thead><tr><th>IP</th><th>Ping</th><th>SSH</th><th>OS (banner)</th><th>Hostname</th><th>GPUs</th><th></th></tr></thead><tbody></tbody></table>';
      const tbody = res.querySelector("tbody");
      for (const h of r.results) {
        const tr = document.createElement("tr");
        const pingTxt = h.ping_alive == null ? "–" : (h.ping_alive ? '<span class="status-ok">alive</span>' : '<span class="status-warn">no reply</span>');
        const sshTxt = h.ssh_port_open
          ? (h.ssh_ok ? '<span class="status-ok">login OK</span>'
             : (h.auth_failed ? '<span class="status-warn">auth failed</span>' : '<span class="status-warn">closed/no auth</span>'))
          : '<span class="status-warn">no SSH</span>';
        const osTxt = h.os
          ? `${osIcon(h.os)}${escHtml(h.os)}${h.banner ? ` <small class="hint" title="${escHtml(h.banner)}">${escHtml(h.banner.split("$").pop() || "")}</small>` : ""}`
          : '<span class="status-warn">unknown</span>';
        tr.innerHTML = `
          <td>${escHtml(h.ip)}</td>
          <td>${pingTxt}</td>
          <td>${sshTxt}</td>
          <td>${osTxt}</td>
          <td>${escHtml(h.hostname || "-")}</td>
          <td>${escHtml((h.gpus || []).join(", ") || "-")}</td>
          <td>${h.already_added
            ? '<small class="hint">already added</small>'
            : (h.os ? `<button type="button" class="small" data-act="add">Select →</button>` : "")}</td>`;
        const addBtn = tr.querySelector('[data-act="add"]');
        if (addBtn) addBtn.onclick = () => { wizSelection = h.ip; discoveredFill(h); };
        // clicking the row selects it (Continue uses the selection)
        tr.onclick = (ev) => {
          if (ev.target.closest("button")) return;
          wizSelection = h.ip;
          document.querySelectorAll("#d-results tr.wiz-selected").forEach(x => x.classList.remove("wiz-selected"));
          tr.classList.add("wiz-selected");
        };
        tbody.appendChild(tr);
      }
    }
  } catch (err) {
    res.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
  }
  btn.disabled = false; btn.textContent = "Scan";
  // scan finished → move the user to step 2 (Select); errors stay on step 1
  if (scanned && !wizEditing() && wizStep === 1) wizShow(2);
});

on("#machine-form", "submit", async (e) => {
  e.preventDefault();
  // wizard guard: Enter/Save before step 3 advances instead of saving
  if (!wizEditing() && wizStep < 3) { wizNext(); return; }
  try {
    const original = $("#machine-original-name").value;
    const data = machineFormRead();
    if (original) await api("PUT", `/api/machines/${original}`, data);
    else await api("POST", "/api/machines", data);
    wizClearPick();
    machineFormLoad(EMPTY_MACHINE, false);
    loadMachines();
  } catch (err) { toast(err.message, "bad"); }
});
on("#machine-cancel", "click", () => machineFormLoad(EMPTY_MACHINE, false));

// picking a discovered IP from the datalist auto-fills name/OS/port
on("#m-ip", "change", () => {
  const h = DISCOVERED[$("#m-ip").value.trim()];
  if (!h) return;
  if (!$("#m-name").value.trim()) $("#m-name").value = h.hostname || h.ip.replace(/\./g, "-");
  if (h.os) $("#m-os").value = h.os;
  if (h.gpu_vendor) $("#m-gpu-vendor").value = h.gpu_vendor;
  $("#m-port").value = h.port || 22;
  if ((h.gpus || []).length && !$("#m-gpu").value.trim())
    $("#m-gpu").value = h.gpus.map((g, i) => `${gpuPrefix(h.gpu_vendor)}${i}`).join(",");
});

on("#m-detect-form", "click", async (e) => {
  const btn = e.currentTarget;
  const ip = $("#m-ip").value.trim();
  if (!ip) { toast("enter an IP first", "bad"); return; }
  btn.disabled = true; btn.textContent = "detecting…";
  try {
    const m = await api("POST", "/api/machines/probe", {
      ip,
      username: $("#m-username").value.trim() || "zazi",
      password: $("#m-password").value,
      port: parseInt($("#m-port").value, 10) || 22,
      os: $("#m-os").value,
      rpc_port: parseInt($("#m-rpc-port").value, 10) || 50052,
    });
    $("#m-gpu").value = m.gpu || "";
    $("#m-ram").value = m.ram_total_gb || "";
    $("#m-expose-gpus").placeholder = m.gpu ? `available: ${m.gpu}` : "all (not detected)";
    toast(`Detected ${ip}: ${((m.detected || {}).gpu_names || []).length} GPU(s), ${m.ram_total_gb || "?"} GB RAM`, "ok");
  } catch (err) { toast(`detect failed: ${err.message}`, "bad"); }
  btn.disabled = false; btn.textContent = "Detect hardware (fills GPU + RAM)";
});

/* ---------------- groups ---------------- */
function grpModalShow(n){
  const modal = $("#grp-modal");
  if (!modal) return;
  modal._grpstep = Math.min(2, Math.max(1, n));
  for (const st of [1, 2]) {
    const el = modal.querySelector(".grpwiz-step-" + st);
    if (el) el.hidden = st !== modal._grpstep;
  }
  modal.querySelectorAll(".grpwiz-indicator li").forEach(li => {
    const st = parseInt(li.dataset.s, 10);
    li.classList.toggle("active", st === modal._grpstep);
    li.classList.toggle("done", st < modal._grpstep);
  });
  const back = modal.querySelector(".grpwiz-back");
  const next = modal.querySelector(".grpwiz-next");
  const save = $("#grp-modal-save");
  if (back) back.disabled = modal._grpstep === 1;
  if (next) next.hidden = modal._grpstep === 2;
  if (save) save.hidden = modal._grpstep !== 2;
  if (modal._grpstep === 2) grpRenderReview();
}
function grpRenderReview(){
  const box = $("#grp-review");
  if (!box) return;
  const name = $("#g-name").value.trim();
  const desc = $("#g-description").value.trim();
  const machines = readGroupMachines();
  const chips = machines.map(m => `<span class="sg-member-chip"><i class="fa-solid fa-server" style="color:var(--accent)"></i>${escHtml(m)}</span>`).join("");
  // DEC-17: engine-appropriate group vocabulary — under vllm a machine
  // group is not an RPC peer cluster, so "deploy across machines" hides
  // (hidden-not-translated; same manifestHasSection idiom as DEC-16).
  const grpVllmReview = manifestHasSection("vllm_section");
  const behaviorLine = grpVllmReview
    ? `<span>${machines.length} machine(s)</span>`
    : `<span>${machines.length} machine(s) — models assigned to this group deploy across them</span>`;
  box.innerHTML = `
    <div class="sg-group-title" style="margin-bottom:6px"><i class="fa-solid fa-layer-group" style="color:var(--accent)"></i> ${escHtml(name || "(unnamed)")}</div>
    ${desc ? `<div class="sg-group-desc">${escHtml(desc)}</div>` : ""}
    <div class="sg-group-members">${chips || '<span class="hint">no machines</span>'}</div>
    <div class="sg-group-behavior" style="margin-top:6px">${behaviorLine}</div>`;
}
function grpModalReset(){
  $("#grp-modal formless-reset") /* noop */;
  $("#g-name").value = "";
  $("#g-description").value = "";
  $("#group-original-name").value = "";
  $("#grp-modal-title").innerHTML = '<i class="fa-solid fa-layer-group"></i> New group';
  renderGroupMachines([]);
  grpModalShow(1);
}
on("#flow-debug-active", "change", (e) => {
  localStorage.setItem("flow-debug-all-active", e.target.checked ? "1" : "0");
  flowLastSvg = null;   // force redraw with the new state
  requestFlowRender();
});
// restore persisted debug state on boot so checkbox matches localStorage
(() => {
  try {
    const dbg = document.getElementById("flow-debug-active");
    if (dbg) dbg.checked = localStorage.getItem("flow-debug-all-active") === "1";
  } catch (_) {}
})();
on("#grp-add", "click", async () => {
  $("#group-original-name").value = "";
  $("#g-name").value = ""; $("#g-description").value = "";
  $("#grp-modal-title").innerHTML = '<i class="fa-solid fa-layer-group"></i> New group';
  await renderGroupMachines([]);
  const modal = $("#grp-modal");
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  grpModalShow(1);
});
on("#grp-modal-close", "click", () => { $("#grp-modal").classList.add("hidden"); $("#grp-modal").setAttribute("aria-hidden", "true"); });
on("#grp-modal-backdrop", "click", () => { $("#grp-modal").classList.add("hidden"); $("#grp-modal").setAttribute("aria-hidden", "true"); });
on("#grp-modal .grpwiz-back", "click", () => grpModalShow(($("#grp-modal")._grpstep || 1) - 1));
on("#grp-modal .grpwiz-next", "click", async () => {
  const modal = $("#grp-modal");
  if (!$("#g-name").value.trim()) { toast("Name the group first", "bad"); $("#g-name").focus(); return; }
  if (!readGroupMachines().length) { toast("Pick at least one machine", "bad"); return; }
  const next = modal.querySelector(".grpwiz-next");
  const orig = next.textContent;
  next.disabled = true; next.textContent = "…";
  try { await renderGroupMachines(readGroupMachines()); } finally { next.disabled = false; next.textContent = orig; }
  grpModalShow(2);
});
on("#grp-modal-save", "click", async () => {
  const data = {
    name: $("#g-name").value.trim(),
    description: $("#g-description").value.trim(),
    machines: readGroupMachines(),
  };
  if (!data.name) { toast("name the group", "bad"); return; }
  if (!data.machines.length) { toast("pick at least one machine", "bad"); return; }
  const original = $("#group-original-name").value;
  try {
    if (original) await api("PUT", `/api/groups/${encodeURIComponent(original)}`, data);
    else await api("POST", "/api/groups", data);
    $("#grp-modal").classList.add("hidden");
    $("#grp-modal").setAttribute("aria-hidden", "true");
    loadGroups();
  } catch (err) { toast(err.message, "bad"); }
});

async function loadGroups() {
  const body = $("#groups-list");
  if (body && !body.querySelector(".sg-group-card"))
    body.innerHTML = loadingSpinner("Loading groups…");
  let groups = [];
  try { groups = await api("GET", "/api/groups"); }
  catch (e) { body.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`; return; }
  body.innerHTML = "";
  if (!groups.length) body.innerHTML = '<p class="hint">No groups yet — use <b>New group</b> to bundle machines that deploy together.</p>';
  for (const g of groups) {
    const card = document.createElement("div");
    card.classList.add("sg-group-card");
    const chips = (g.machines || []).map(m =>
      `<span class="sg-member-chip"><i class="fa-solid fa-server" style="color:var(--accent)"></i>${escHtml(m)}</span>`).join("") || '<span class="hint">no machines</span>';
    card.innerHTML = `
      <div class="sg-group-head">
        <div class="sg-group-title">
          <i class="fa-solid fa-layer-group" style="color:var(--accent)"></i>
          ${escHtml(g.name)}
          <span class="badge" style="background:var(--accent-dim);color:var(--accent)">${(g.machines || []).length} machine(s)</span>
        </div>
        <div class="scan-model-actions">
          <button class="small" data-act="edit">Edit</button>
          <button class="small" data-act="status">Status</button>
          <button class="small danger" data-act="delete">Delete</button>
        </div>
      </div>
      ${g.description ? `<div class="sg-group-desc">${escHtml(g.description)}</div>` : ""}
      <div class="sg-group-members">${chips}</div>
      <div class="sg-group-behavior group-status"><span class="hint">status not checked yet</span></div>`;
    card.querySelector('[data-act="edit"]').onclick = async () => {
      $("#group-original-name").value = g.name;
      $("#g-name").value = g.name;
      $("#g-description").value = g.description || "";
      $("#grp-modal-title").innerHTML = `<i class="fa-solid fa-layer-group"></i> Edit ${escHtml(g.name)}`;
      await renderGroupMachines(g.machines);
      const modal = $("#grp-modal");
      modal.classList.remove("hidden");
      modal.setAttribute("aria-hidden", "false");
      grpModalShow(1);
    };
    card.querySelector('[data-act="status"]').onclick = async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true; btn.textContent = "…";
      const cell = card.querySelector(".group-status");
      cell.innerHTML = '<span class="hint">checking…</span>';
      try {
        const st = await api("GET", `/api/groups/${encodeURIComponent(g.name)}/status`);
        // DEC-16: engine-correct group status line — RPC up/down is
        // llama.cpp-only vocabulary; under vllm show reachable only.
        const grpVllm = manifestHasSection("vllm_section");
        cell.innerHTML = st.map(s =>
          `<span class="${s.reachable && (grpVllm || s.rpc_running) ? "status-ok" : "status-warn"}">${escHtml(s.name)}: ${s.reachable ? "online" : "offline"}${grpVllm ? "" : `, rpc ${s.rpc_running ? "up" : "down"}`}</span>`).join(" · ");
      } catch (err) { cell.innerHTML = `<span class="status-bad">${escHtml(err.message)}</span>`; }
      btn.disabled = false; btn.textContent = "Status";
    };
    card.querySelector('[data-act="delete"]').onclick = async (e) => {
      const btn = e.currentTarget;
      if (!armed(btn, "Delete")) return;
      btn.disabled = true;
      try {
        await api("DELETE", `/api/groups/${encodeURIComponent(g.name)}`);
        toast(`Deleted group ${g.name}`, "ok");
        loadGroups();
      } catch (err) { toast(`Delete failed: ${err.message}`, "bad"); btn.disabled = false; btn.textContent = "Delete"; }
    };
    body.appendChild(card);
  }
}

// Engine-aware machine chips (B8 + B14): machines the vLLM engine cannot
// use are greyed out in vllm mode. Capability comes from the BACKEND
// payload (vllm_capable / vllm_capability_reason / vllm_capability_info,
// HW-PARITY audit §3b — windows/macos/intel/cpu) — the client must NOT
// re-derive os/gpu_vendor rules (B8's old client-side isWindows check
// could drift from the server's decision). Capable-but-caveated hosts
// (ROCm wheels, unknown gpu_vendor) stay checkable and show an info note.

async function renderGroupMachines(selected) {
  const chipList = $("#g-machines-chip-list");
  if (!chipList) return;
  let machines = [];
  try { machines = await api("GET", "/api/machines"); }
  catch (_) { chipList.innerHTML = '<span class="status-bad">machines unavailable</span>'; return; }
  const settings = await getAppSettings();
  const isVllm = (settings && settings.engine) === "vllm";
  const chosen = new Set(selected || []);
  chipList.innerHTML = machines.map(mac => {
    // B14: consume the server's hardware verdict — no client-side re-derivation
    const capable = mac.vllm_capable !== false;
    const reason = mac.vllm_capability_reason || "vLLM is not usable on this host";
    const info = mac.vllm_capability_info || "";
    const disabled = isVllm && !capable;
    const chipClass = `hw-chip${mac.role === "master" ? " ram-chip" : ""}${disabled ? " vllm-disabled" : ""}`;
    // info tooltip = proper in-chip info section (platform + why + what to
    // use instead), not just title text
    const tooltip = disabled
      ? `<span class="vllm-capability-info" role="note">`
          + `<span class="vllm-cap-platform">${escHtml((mac.os || "linux").toLowerCase())}</span>`
          + ` — ${escHtml(reason)}`
          + ` <span class="vllm-cap-alt">Use the llama.cpp engine for this machine.</span>`
          + `</span>`
      : "";
    const warnNote = (!disabled && isVllm && info)
      ? `<span class="chip-sub" style="color:var(--warn)"> · ${escHtml(info)}</span>`
      : "";
    const disabledSub = disabled
      ? `<span class="chip-sub" style="color:var(--warn)"> · ${escHtml((mac.os || "linux").toLowerCase())} — ${escHtml(reason)}</span>`
      : "";
    return `<label class="${chipClass}"${disabled ? ` title="${escHtml(reason)}"` : ""}>
      <input type="checkbox" value="${escHtml(mac.name)}" ${disabled ? "disabled" : ""} ${chosen.has(mac.name) ? "checked" : ""}>
      <i class="fa-solid fa-server chip-icon"></i>
      <span class="chip-name">${escHtml(mac.name)}${mac.role === "master" ? ' <span class="badge badge-master">master</span>' : ""}</span>
      <span class="chip-sub">${escHtml(mac.ip)} · ${((mac.detected || {}).gpu_names || []).length || "?"} GPU</span>
      ${disabledSub}
      ${warnNote}
      ${tooltip}
    </label>`;
  }).join("");
}

function readGroupMachines() {
  return [...document.querySelectorAll("#g-machines-chip-list input:checked")].map(cb => cb.value);
}


/* ---------------- serve groups (llama-swap model co-serving) ---------------- */
// Deployments = base models + --variant configs, the actual routable units.
async function listDeployments() {
  let models = [];
  try { models = await api("GET", "/api/models"); } catch (_) { return []; }
  const deps = [];
  for (const m of models) {
    deps.push({ label: m.model_id, title: m.name || m.model_id, servers: m.servers || [], variant: null, model_id: m.model_id, device_order: m.device_order || "" });
    for (const v of (m.variants || [])) {
      const lab = (v.label || "").trim();
      if (!lab) continue;
      deps.push({ label: `${m.model_id}--${lab}`, title: `${m.name || m.model_id} (${lab})`, servers: v.servers || m.servers || [], variant: lab, model_id: m.model_id, device_order: v.device_order || m.device_order || "" });
    }
  }
  return deps;
}

async function renderServeGroupMembers(selected, editingName) {
  const chipList = $("#sg-members-chip-list");
  if (!chipList) return;
  const [deps, machines, sgroups] = await Promise.all([
    listDeployments(),
    api("GET", "/api/machines").catch(() => []),
    api("GET", "/api/swap-groups").catch(() => []),
  ]);
  if (!deps.length) { chipList.innerHTML = '<span class="hint">no models configured yet — add some on the Models tab first</span>'; return; }
  const vramByMachine = {};
  for (const m of machines) vramByMachine[m.name] = (m.detected && m.detected.vram_total_gb) || "?";
  // deployments already claimed by another serve group (llama-swap: one group each)
  const claimedBy = {};
  for (const g of sgroups) {
    if (g.name === editingName) continue;
    for (const mem of (g.members || [])) claimedBy[mem] = g.name;
  }
  const chosen = new Set(selected || []);
  chipList.innerHTML = deps.map(d => {
    const sel = chosen.has(d.label);
    const claimed = claimedBy[d.label];
    const servers = (d.servers || []).map(s =>
      `<span class="os-badge">${escHtml(s)} · ${escHtml(String(vramByMachine[s] ?? "?"))} GB</span>`).join(" ") || '<span class="hint">no servers</span>';
    return `
    <div class="dep-card${sel ? " selected" : ""}" data-label="${escHtml(d.label)}" role="checkbox" aria-checked="${sel ? "true" : "false"}" tabindex="0" title="click to ${sel ? "remove from" : "add to"} this group">
      <input type="checkbox" value="${escHtml(d.label)}" ${sel ? "checked" : ""} hidden>
      <div class="dep-title">${escHtml(d.title)}${d.variant ? ' <span class="badge" style="background:var(--accent-dim);color:var(--accent)">variant</span>' : ""}</div>
      <div class="dep-servers">${servers}</div>
      <div class="dep-foot">${claimed ? `<span class="hint">in group: ${escHtml(claimed)}</span>` : `<span class="hint">serves via :8080 as <code>${escHtml(d.label)}</code></span>`}</div>
    </div>`;}).join("");
  chipList.querySelectorAll(".dep-card").forEach(card => {
    const toggle = () => {
      const cb = card.querySelector("input[type=checkbox]");
      cb.checked = !cb.checked;
      card.classList.toggle("selected", cb.checked);
      card.setAttribute("aria-checked", cb.checked ? "true" : "false");
      card.title = `click to ${cb.checked ? "remove from" : "add to"} this group`;
      updateServeGroupExplainer();
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); }
    });
  });
  updateServeGroupExplainer();
}

function readServeGroupMembers() {
  return [...document.querySelectorAll("#sg-members-chip-list input:checked")].map(cb => cb.value);
}

/* Plain-language serve-group explainer: what the flags MEAN for the
   current form selection, plus fit + sharing analysis. No jargon. */
let sgExplainerState = { depsAt: 0, deps: [], statsAt: 0, stats: null, est: new Map() };
async function sgCachedDeps() {
  const now = Date.now();
  if (now - sgExplainerState.depsAt < 15000 && sgExplainerState.deps.length) return sgExplainerState.deps;
  sgExplainerState.deps = await listDeployments();
  sgExplainerState.depsAt = now;
  return sgExplainerState.deps;
}
async function sgCachedStats() {
  const now = Date.now();
  if (now - sgExplainerState.statsAt < 30000 && sgExplainerState.stats) return sgExplainerState.stats;
  try { sgExplainerState.stats = await api("GET", "/api/stats"); } catch (_) { /* stay silent, hints are best-effort */ }
  sgExplainerState.statsAt = now;
  return sgExplainerState.stats;
}
async function sgEstimateGb(modelId) {
  const hit = sgExplainerState.est.get(modelId);
  if (hit && Date.now() - hit.at < 300000) return hit;
  try {
    const e = await api("GET", `/api/models/${encodeURIComponent(modelId)}/tune/estimate`);
    const v = { at: Date.now(), gb: (e.weights_gb || 0) + (e.kv_at_ctx_gb || 0) };
    sgExplainerState.est.set(modelId, v);
    return v;
  } catch (_) { return null; }
}
// "aibox2-CUDA0" -> ["aibox2", "CUDA0"] (split on LAST hyphen, mirrors backend)
function sgSplitChip(chip) {
  const i = chip.lastIndexOf("-");
  return i < 0 ? [chip, ""] : [chip.slice(0, i), chip.slice(i + 1)];
}
function sgPeerResources(dep, machinesByName) {
  // per-peer-machine GPU sets from the deployment's device order;
  // "" order = unknown selection -> excluded from sharing analysis
  if (!dep.device_order) return null;
  const master = sgMasterOf(dep, machinesByName);
  const per = {};
  for (const chip of dep.device_order.split(",").map(s => s.trim()).filter(Boolean)) {
    const [mach, res] = sgSplitChip(chip);
    if (!mach || mach === master || res === "" || res === "RAM") continue;
    (per[mach] = per[mach] || []).push(res);
  }
  return per;
}
function sgMasterOf(dep, machinesByName) {
  const servers = dep.servers || [];
  const m = servers.find(s => (machinesByName[s] || {}).role === "master");
  return m || servers[0] || "";
}
function describeServeGroup(memberLabels, flags, ctx) {
  // ctx: {depByLabel, otherGroupNames, ungroupedLabels}
  const n = memberLabels.length;
  const names = memberLabels.map(l => (ctx.depByLabel[l] || {}).title || l);
  const head = n === 0 ? "Pick members above and this box explains the group in plain language."
    : n === 1 ? `<b>${escHtml(names[0])}</b> is on its own here.`
    : `<b>${names.map(escHtml).join("</b>, <b>")}</b> are in this group.`;
  const parts = [head];
  if (n >= 2) parts.push(flags.swap
    ? "They <b>take turns</b> — only one runs at a time; loading one parks the others."
    : "They <b>run side by side</b>, sharing VRAM.");
  if (flags.exclusive) parts.push("Loading a member <b>clears the stage</b> — models outside this group get unloaded.");
  if (flags.persistent) parts.push("This group <b>always stays loaded</b> — nothing else can auto-unload it (you can still evict it yourself).");
  if (!flags.swap && !flags.exclusive && !flags.persistent && n > 0) parts.push("No special behavior — plain co-serving.");
  const notes = [];
  if (n === 1 && flags.swap) notes.push({ level: "warn", text: "Take turns needs 2+ members — with one member it does nothing yet." });
  if (flags.exclusive && ctx.otherGroupNames.length === 0 && ctx.ungroupedLabels) {
    notes.push(ctx.ungroupedLabels.length
      ? { level: "info", text: "This is the only group, so clear-the-stage only affects models with no group." }
      : { level: "warn", text: "Every deployment is inside this group, so clear-the-stage currently has nothing to clear." });
  }
  if (n === 1 && flags.persistent) notes.push({ level: "ok", text: "Good use of always-stay-loaded: cheap insurance for one small model. Don't persist big models — they squat on VRAM other loads may need." });
  return { summary: parts.join(" "), notes };
}
async function updateServeGroupExplainer() {
  const box = $("#sg-plain-summary"), hints = $("#sg-hints"), fit = $("#sg-fit-line");
  if (!box) return;
  const members = readServeGroupMembers();
  const flags = { swap: $("#sg-swap").checked, exclusive: $("#sg-exclusive").checked, persistent: $("#sg-persistent").checked };
  let deps = [], machines = [], sgroups = [];
  try {
    [deps, machines, sgroups] = await Promise.all([
      sgCachedDeps(),
      api("GET", "/api/machines").catch(() => []),
      api("GET", "/api/swap-groups").catch(() => []),
    ]);
  } catch (_) { /* form still usable without the explainer */ }
  const depByLabel = Object.fromEntries(deps.map(d => [d.label, d]));
  const editing = ($("#sgroup-original-name") || {}).value || "";
  const otherGroups = sgroups.filter(g => g.name !== editing);
  const grouped = new Set(otherGroups.flatMap(g => g.members || []));
  const ungrouped = deps.map(d => d.label).filter(l => !grouped.has(l) && !members.includes(l));
  const { summary, notes } = describeServeGroup(members, flags, {
    depByLabel, otherGroupNames: otherGroups.map(g => g.name), ungroupedLabels: ungrouped,
  });
  box.innerHTML = summary;
  // --- shared-peer analysis: same box + same GPUs = reuse, else restart ---
  const machinesByName = Object.fromEntries(machines.map(m => [m.name, m]));
  const peerOwners = {};   // machine -> [{label, res}]
  const consider = [];
  for (const l of members) { const d = depByLabel[l]; if (d) consider.push({ label: l, group: null, dep: d }); }
  for (const g of otherGroups) for (const l of (g.members || [])) { const d = depByLabel[l]; if (d) consider.push({ label: l, group: g.name, dep: d }); }
  for (const c of consider) {
    const per = sgPeerResources(c.dep, machinesByName);
    if (!per) continue;
    for (const [mach, res] of Object.entries(per)) (peerOwners[mach] = peerOwners[mach] || []).push({ ...c, res: [...res].sort().join(",") });
  }
  for (const [mach, owners] of Object.entries(peerOwners)) {
    if (owners.length < 2) continue;
    const uniq = [...new Set(owners.map(o => o.res))];
    const who = owners.map(o => o.group ? `${o.label} (${o.group})` : o.label).map(escHtml).join(" vs ");
    if (uniq.length === 1) notes.push({ level: "ok", text: `${who} share ${escHtml(mach)} with the same GPUs (${escHtml(uniq[0])}) — the second load reuses the running server, no interruption.` });
    else notes.push({ level: "warn", text: `${who} want different GPUs on ${escHtml(mach)} (${uniq.map(escHtml).join(" vs ")}) — loading one restarts the other's backend (brief interruption). Give them the same GPUs to share cleanly.` });
  }
  hints.innerHTML = notes.map(n => `<div class="sg-note ${n.level}">${escHtml(n.text)}</div>`).join("");
  // --- VRAM fit: summed estimates vs cluster free ---
  if (fit) {
    fit.textContent = "";
    fit.className = "hint";
    const baseIds = [...new Set(members.map(l => (depByLabel[l] || {}).model_id).filter(Boolean))];
    if (flags.swap) {
      if (baseIds.length) fit.textContent = "Take turns: only one member resident at a time, so VRAM only needs to fit the biggest.";
    } else if (baseIds.length) {
      const ests = await Promise.all(baseIds.map(sgEstimateGb));
      const good = ests.filter(Boolean);
      if (good.length) {
        const need = good.reduce((a, e) => a + e.gb, 0);
        const stats = await sgCachedStats();
        const vram = stats && stats.totals && stats.totals.vram;
        if (vram && vram.total) {
          const free = Math.max(0, vram.total - vram.used);
          fit.textContent = `Together ≈ ${need.toFixed(1)} GB VRAM (weights + context) — cluster has ${free.toFixed(1)} of ${vram.total.toFixed(0)} GB free.`;
          if (need > free) { fit.textContent += " Too big to co-serve — use take turns."; fit.className = "hint status-bad"; }
        } else fit.textContent = `Together ≈ ${need.toFixed(1)} GB VRAM (weights + context).`;
      }
    }
  }
}
function sgApplyPreset(patch) {
  $("#sg-swap").checked = !!patch.swap;
  $("#sg-exclusive").checked = !!patch.exclusive;
  $("#sg-persistent").checked = !!patch.persistent;
  updateServeGroupExplainer();
}
function sgApplyPresetCard(card) {
  document.querySelectorAll("#sg-modal .sg-preset-card").forEach(c =>
    c.classList.toggle("selected", c === card));
  const flags = new Set((card.dataset.flags || "").split(",").filter(Boolean));
  $("#sg-swap").checked = flags.has("swap");
  $("#sg-exclusive").checked = flags.has("exclusive");
  $("#sg-persistent").checked = flags.has("persistent");
  updateServeGroupExplainer();
}
document.querySelectorAll("#sg-modal .sg-preset-card").forEach(card => {
  card.addEventListener("click", () => sgApplyPresetCard(card));
  card.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); sgApplyPresetCard(card); }
  });
});
on("#sg-swap", "change", updateServeGroupExplainer);
on("#sg-exclusive", "change", updateServeGroupExplainer);
on("#sg-persistent", "change", updateServeGroupExplainer);

function serveGroupFormReset() {
  $("#sg-modal formless-reset") /* noop */;
  document.querySelectorAll("#sg-modal input[type=checkbox]").forEach(cb => cb.checked = false);
  $("#sg-name").value = "";
  $("#sg-description").value = "";
  $("#sgroup-original-name").value = "";
  $("#sg-modal-title").innerHTML = '<i class="fa-solid fa-layer-group"></i> New serve group';
  document.querySelectorAll("#sg-modal .sg-preset-card").forEach(c => c.classList.remove("selected"));
  renderServeGroupMembers([]);
  sgModalShow(1);
}
function sgModalShow(n){
  const modal = $("#sg-modal");
  if (!modal) return;
  modal._sgstep = Math.min(3, Math.max(1, n));
  for (const st of [1, 2, 3]) {
    const el = modal.querySelector(".sgwiz-step-" + st);
    if (el) el.hidden = st !== modal._sgstep;
  }
  modal.querySelectorAll(".sgwiz-indicator li").forEach(li => {
    const st = parseInt(li.dataset.s, 10);
    li.classList.toggle("active", st === modal._sgstep);
    li.classList.toggle("done", st < modal._sgstep);
  });
  const back = modal.querySelector(".sgwiz-back");
  const next = modal.querySelector(".sgwiz-next");
  const save = $("#sg-modal-save");
  if (back) back.disabled = modal._sgstep === 1;
  if (next) next.hidden = modal._sgstep === 3;
  if (save) save.hidden = modal._sgstep !== 3;
  if (modal._sgstep === 3) updateServeGroupExplainer();
}
on("#sg-add", "click", () => {
  $("#sgroup-original-name").value = "";
  $("#sg-modal-title").innerHTML = '<i class="fa-solid fa-layer-group"></i> New serve group';
  document.querySelectorAll("#sg-modal input[type=checkbox]").forEach(cb => cb.checked = false);
  $("#sg-name").value = ""; $("#sg-description").value = "";
  document.querySelectorAll("#sg-modal .sg-preset-card").forEach(c => c.classList.remove("selected"));
  renderServeGroupMembers([]).then(() => sgModalShow(1));
  const modal = $("#sg-modal");
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
});
on("#sg-modal-close", "click", () => { $("#sg-modal").classList.add("hidden"); $("#sg-modal").setAttribute("aria-hidden", "true"); });
on("#sg-modal-backdrop", "click", () => { $("#sg-modal").classList.add("hidden"); $("#sg-modal").setAttribute("aria-hidden", "true"); });
on("#sg-modal .sgwiz-back", "click", () => sgModalShow(($("#sg-modal")._sgstep || 1) - 1));
on("#sg-modal .sgwiz-next", "click", async () => {
  const modal = $("#sg-modal");
  const step = modal._sgstep || 1;
  if (step === 1 && !document.querySelector("#sg-modal .sg-preset-card.selected")) {
    toast("Pick a behavior card to continue", "bad");
    return;
  }
  if (step === 2) {
    if (!$("#sg-name").value.trim()) { toast("Name the group first", "bad"); $("#sg-name").focus(); return; }
    if (!readServeGroupMembers().length) { toast("Pick at least one member deployment", "bad"); return; }
    const next = modal.querySelector(".sgwiz-next");
    const orig = next.textContent;
    next.disabled = true; next.textContent = "loading…";
    try {
      await renderServeGroupMembers(readServeGroupMembers(), $("#sgroup-original-name").value);
    } finally {
      next.disabled = false; next.textContent = orig;
    }
  }
  sgModalShow(step + 1);
});
function readServeGroupFlags(){
  return [$("#sg-swap").checked, $("#sg-exclusive").checked, $("#sg-persistent").checked];
}

async function loadServeGroups() {
  const body = $("#sgroups-list");
  if (body && !body.querySelector(".sg-group-card"))
    body.innerHTML = loadingSpinner("Loading serve groups…");
  let sgroups = [];
  try { sgroups = await api("GET", "/api/swap-groups"); }
  catch (e) { body.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`; return; }
  body.innerHTML = "";
  if (!sgroups.length) body.innerHTML = '<p class="hint">No serve groups yet — everything runs one-at-a-time (llama-swap default). Use <b>New group</b> to let models run together.</p>';
  const depByLabel = Object.fromEntries((await sgCachedDeps()).map(x => [x.label, x]));
  for (const g of sgroups) {
    const card = document.createElement("div");
    card.classList.add("sg-group-card");
    const icon = g.swap ? (g.exclusive ? "fa-person-rays" : "fa-rotate") : (g.persistent ? "fa-anchor" : "fa-objects-column");
    const mode = g.swap ? (g.exclusive ? "Solo run" : "Take turns") : (g.persistent ? "Always-on" : "Run together");
    const chips = (g.members || []).map(m => {
      const dep = depByLabel[m];
      const srv = (dep && dep.servers || []).join(", ");
      return `<span class="sg-member-chip" title="${escHtml(srv ? "serves on: " + srv : "")}"><i class="fa-solid fa-cube" style="color:var(--accent)"></i>${escHtml((dep && dep.title) || m)}</span>`;
    }).join("") || '<span class="hint">no members</span>';
    const flags = [
      g.swap ? '<span class="on"><i class="fa-solid fa-rotate"></i> take turns</span>' : '',
      g.exclusive ? '<span class="on"><i class="fa-solid fa-broom"></i> clears stage</span>' : '',
      g.persistent ? '<span class="on"><i class="fa-solid fa-anchor"></i> stays loaded</span>' : '',
    ].filter(Boolean).join(" ") || '<span>plain co-serving</span>';
    card.innerHTML = `
      <div class="sg-group-head">
        <div class="sg-group-title">
          <i class="fa-solid ${icon}" style="color:var(--accent)"></i>
          ${escHtml(g.name)}
          <span class="badge" style="background:var(--accent-dim);color:var(--accent)">${escHtml(mode)}</span>
        </div>
        <div class="scan-model-actions">
          <button class="small" data-act="edit">Edit</button>
          <button class="small danger" data-act="delete">Delete</button>
        </div>
      </div>
      ${g.description ? `<div class="sg-group-desc">${escHtml(g.description)}</div>` : ""}
      <div class="sg-group-members">${chips}</div>
      <div class="sg-group-behavior">${flags}</div>`;
    card.querySelector('[data-act="edit"]').onclick = () => {
      $("#sgroup-original-name").value = g.name;
      $("#sg-name").value = g.name;
      $("#sg-description").value = g.description || "";
      $("#sg-swap").checked = !!g.swap;
      $("#sg-exclusive").checked = !!g.exclusive;
      $("#sg-persistent").checked = !!g.persistent;
      $("#sg-modal-title").innerHTML = `<i class="fa-solid fa-layer-group"></i> Edit ${escHtml(g.name)}`;
      // highlight the card whose flags match exactly (falls back to none for custom combos)
      const gflags = [g.swap && "swap", g.exclusive && "exclusive", g.persistent && "persistent"].filter(Boolean).sort().join(",");
      document.querySelectorAll("#sg-modal .sg-preset-card").forEach(c => {
        const cflags = (c.dataset.flags || "").split(",").filter(Boolean).sort().join(",");
        c.classList.toggle("selected", cflags === gflags);
      });
      renderServeGroupMembers(g.members, g.name).then(() => {
        const modal = $("#sg-modal");
        modal.classList.remove("hidden");
        modal.setAttribute("aria-hidden", "false");
        sgModalShow(1);
      });
    };
    card.querySelector('[data-act="delete"]').onclick = async (e) => {
      const btn = e.currentTarget;
      if (!armed(btn, "Delete")) return;
      btn.disabled = true;
      try {
        await api("DELETE", `/api/swap-groups/${encodeURIComponent(g.name)}`);
        toast(`Deleted serve group ${g.name}`, "ok");
        pushSwapRoutes(`serve group ${g.name} deleted`);
        loadServeGroups();
      } catch (err) { toast(`Delete failed: ${err.message}`, "bad"); btn.disabled = false; btn.textContent = "Delete"; }
    };
    body.appendChild(card);
  }
}

// push regenerated routes (no restart — daemon watches the config)
async function pushSwapRoutes(why) {
  try {
    const r = await api("POST", "/api/swap/push");
    const n = (r.models || []).length;
    toast(`${why || "Saved"} — swap routes pushed (${n} models)`, "ok");
  } catch (err) { toast(`Saved, but route push failed: ${err.message} — Launch All to apply`, "bad", 9000); }
}

on("#sg-modal-save", "click", async () => {
  const data = {
    name: $("#sg-name").value.trim(),
    description: $("#sg-description").value.trim(),
    members: readServeGroupMembers(),
    swap: $("#sg-swap").checked,
    exclusive: $("#sg-exclusive").checked,
    persistent: $("#sg-persistent").checked,
  };
  if (!data.name) { toast("name the group", "bad"); return; }
  if (!data.members.length) { toast("pick at least one member deployment", "bad"); return; }
  const original = $("#sgroup-original-name").value;
  try {
    if (original) await api("PUT", `/api/swap-groups/${encodeURIComponent(original)}`, data);
    else await api("POST", "/api/swap-groups", data);
    $("#sg-modal").classList.add("hidden");
    $("#sg-modal").setAttribute("aria-hidden", "true");
    loadServeGroups();
    pushSwapRoutes(`Serve group ${data.name} saved`);
  } catch (err) { toast(err.message, "bad"); }
});

/* ---------------- models ---------------- */
let flagsMode = "rows";   // "rows" | "text"

/* flags<->text conversion: one --flag [value] per line */
function flagsToText(flags) {
  return Object.entries(flags || {}).map(([k, v]) => {
    if (v === true) return `--${k}`;
    return `--${k} ${v}`;
  }).join("\n");
}

function textToFlags(text) {
  const flags = {};
  for (const line of (text || "").split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const parts = t.replace(/^--/, "").split(/\s+/);
    const key = parts[0];
    if (!key) continue;
    const raw = parts.slice(1).join(" ");
    if (raw === "") flags[key] = true;
    else if (raw === "true" || raw === "false") flags[key] = raw === "true";
    else if (!isNaN(raw) && raw !== "") flags[key] = Number(raw);
    else flags[key] = raw;
  }
  return flags;
}

function setFlagsMode(mode) {
  flagsMode = mode;
  const isRows = mode === "rows";
  if (!isRows) {
    // rows -> text: serialize current chip state
    $("#flags-text").value = flagsToText(readFlags());
  } else {
    // text -> rows: parse text into chips
    const flags = textToFlags($("#flags-text").value);
    $("#flags-container").innerHTML = "";
    for (const [k, v] of Object.entries(flags)) addFlagRow(k, v);
  }
  $("#flags-container").hidden = !isRows;
  $("#flags-row-actions").hidden = !isRows;
  $("#flags-text").hidden = isRows;
  $("#flags-mode-rows").style.borderColor = isRows ? "var(--accent)" : "";
  $("#flags-mode-text").style.borderColor = !isRows ? "var(--accent)" : "";
  $("#flags-mode-hint").textContent = isRows
    ? "one --flag value per pair" : "one --flag [value] per line · # comments";
}

on("#flags-mode-rows", "click", () => setFlagsMode("rows"));
on("#flags-mode-text", "click", () => setFlagsMode("text"));

function addFlagRow(key = "", value = "") {
  const row = document.createElement("div");
  row.className = "flag-row";
  row.innerHTML = `<input class="flag-key" placeholder="ctx-size" value="${key}"><input class="flag-value" placeholder="200000" value="${value}"><button type="button" class="small danger">x</button>`;
  row.querySelector("button").onclick = () => row.remove();
  $("#flags-container").appendChild(row);
}
on("#add-flag", "click", () => addFlagRow());

function applyTensorModeUI() {
  const mode = $("#mo-tensor-mode")?.value || "auto";
  const elManual = $("#mo-tensor-analysis-wrap");   // analysis/preview row (fit only)
  const elMargin = $("#mo-fit-margin-wrap");
  const elRam = $("#mo-fit-ram-wrap");
  const elSplit = $("#mo-tensor-split-wrap");
  if (elManual) elManual.hidden = mode !== "fit";
  if (elMargin) elMargin.hidden = mode !== "fit";
  if (elRam) elRam.hidden = mode !== "fit";
  // when we collapsed the two fit rows into one, elRam may be null — treat margin wrap as proxy for both
  if (!elRam && elMargin) elMargin.hidden = mode !== "fit";
  if (elSplit) elSplit.hidden = mode !== "split";
}
on("#mo-tensor-mode", "change", applyTensorModeUI);

function readFlags() {
  if (flagsMode === "text") return textToFlags($("#flags-text").value);
  const flags = {};
  document.querySelectorAll("#flags-container .flag-row").forEach((row) => {
    const key = row.querySelector(".flag-key").value.trim().replace(/^--/, "");
    const raw = row.querySelector(".flag-value").value.trim();
    if (!key) return;
    if (raw === "true" || raw === "false") flags[key] = raw === "true";
    else if (raw !== "" && !isNaN(raw)) flags[key] = Number(raw);
    else flags[key] = raw;
  });
  return flags;
}

async function refreshMoGroups(selected) {
  const sel = $("#mo-group");
  if (!sel) return;
  let groups = [];
  try { groups = await api("GET", "/api/groups"); } catch (_) {}
  const cur = selected || sel.value || "";
  sel.innerHTML = '<option value="">(custom / no group)</option>' + groups.map(g => `<option value="${escHtml(g.name)}">${escHtml(g.name)} — ${escHtml((g.machines||[]).join(", "))}</option>`).join("");
  sel.value = cur;
  if (selected && !groups.find(g=>g.name===selected)) sel.value = "";
}

function modelFormLoad(m, isEdit) {
  // isEdit=true only for a SAVED model via table Edit/row click. Scan fills
  // and fresh forms are new machines even with a suggested id — conflating
  // the two would hide the wizard (same bug class as the machines wizard).
  const editing = isEdit ?? Boolean(m.model_id);
  $("#model-original-id").value = editing ? (m.model_id || "") : "";
  $("#mo-id").value = m.model_id || "";
  $("#mo-name").value = m.name || "";
  $("#mo-path").value = m.base_path || "";
  $("#mo-runtime").value = m.runtime || "llama";
  // runtime: custom (external server) — kind/bat/port round-trip so a
  // custom model is a fully editable saved config, not a hand-only file
  const mc = m.custom || {};
  $("#mo-custom-kind").value = mc.kind || "winbox-bat";
  $("#mo-custom-bat").value = mc.bat || "";
  $("#mo-custom-port").value = mc.port ?? "";
  $("#mo-custom-proxy").value = mc.proxy || "";
  $("#mo-custom-ep").value = mc.checkEndpoint || "";
  $("#mo-custom-cmd").value = mc.cmd || "";
  $("#mo-custom-cmdstop").value = mc.cmdStop || "";
  $("#mo-custom-env").value = Object.entries(mc.env || {}).map(([k, v]) => `${k}=${v}`).join("\n");
  $("#mo-custom-script").value = mc.script || "";
  $("#mo-custom-vision").checked = mc.vision === true;
  syncCustomRow();
  const mode = (m.tensor_split_mode === "manual") ? "fit" : (m.tensor_split_mode || "fit");
  $("#mo-tensor-mode").value = mode;
  $("#mo-tensor-split").value = m.tensor_split || "";
  $("#mo-fit-margin").value = m.fit_margin || 1024;
  $("#mo-fit-ram-allow").value = m.fit_ram_allow ?? 1024;
  $("#mo-order-policy").value = m.device_order_policy || "vram-first";
  applyTensorModeUI();
  $("#mo-llama-bin").value = m.llama_bin || "";
  if (!m.llama_bin) getAppSettings().then(s => {
    if (!$("#mo-llama-bin").value) {
      $("#mo-llama-bin").value = s.llama_bin
        ? s.llama_bin.replace(/\/?$/, "") + "/llama-server"
        : "~/llama.cpp/build/bin/llama-server";
    }
  });
  refreshMoGroups(m.group || "");
  $("#mo-pinned").checked = !!m.pinned;
  $("#mo-ttl").value = m.ttl || "";
  // common flags toggles
  const f = m.flags || {};
  $("#flag-ctx-size").value = f["ctx-size"] || f["c"] || "";
  $("#flag-flash-attn").checked = f["flash-attn"] === "on" || f["flash-attn"] === true;
  $("#flag-jinja").checked = !!f["jinja"];
  $("#flag-cache-k").value = f["cache-type-k"] || "";
  $("#flag-cache-v").value = f["cache-type-v"] || "";
  if (!m.model_id && !Object.keys(f).length) {
    // blank new-model form: pre-apply the known-good defaults
    // (q8 KV, flash, jinja) — Tune refines from here, ctx stays user's
    $("#flag-flash-attn").checked = true;
    $("#flag-jinja").checked = true;
    $("#flag-cache-k").value = "q8_0";
    $("#flag-cache-v").value = "q8_0";
  }
  renderModelServers(m);
  $("#flags-container").innerHTML = "";
  for (const [k, v] of Object.entries(m.flags || {})) addFlagRow(k, v);
  $("#flags-text").value = flagsToText(m.flags || {});
  // per-model vllm section (schema-valid keys only). A NEW model under
  // the vllm engine pre-fills the live-validated farm defaults (fp32,
  // eager, tool-calling, 32k len, pp layout) — mirrors the backend's
  // save-time fill so the editor and the emitted script can't drift,
  // and every value stays individually editable.
  const mv = (m.vllm && Object.keys(m.vllm).length)
    ? m.vllm
    : (!m.model_id && (appSettings?.engine === "vllm")
       ? (appSettings.ui_manifest?.vllm_new_model_defaults || {}) : {});
  $("#mo-vllm-max-model-len").value = mv.max_model_len ?? "";
  $("#mo-vllm-max-num-seqs").value = mv.max_num_seqs ?? "";
  $("#mo-vllm-tp").value = mv.tensor_parallel_size ?? "";
  $("#mo-vllm-pp").value = mv.pipeline_parallel_size ?? "";
  // A2: per-model gmu override; empty = settings default (shown as
  // placeholder so the effective value is visible without being stored)
  const gmInp = $("#mo-vllm-gmu");
  if (gmInp) {
    gmInp.value = mv.gpu_memory_utilization ?? "";
    const dflt = (appSettings && typeof appSettings.vllm?.default_gpu_memory_utilization === "number")
      ? appSettings.vllm.default_gpu_memory_utilization : 0.9;
    gmInp.placeholder = `default ${dflt}`;
  }
  $("#mo-vllm-kv").value = mv.kv_cache_dtype || "";
  $("#mo-vllm-cpu-offload").value = mv.cpu_offload_gb ?? "";
  $("#mo-vllm-quant").value = mv.quantization || "";
  $("#mo-vllm-dtype").value = mv.dtype || "";
  // eager/tools are hw-chip toggles (llama.cpp flash-attn pattern):
  // default ON unless explicitly stored false — matches the launch
  // default, so editor and emitted script never drift
  const eg = $("#mo-vllm-eager");
  if (eg) { eg.checked = mv.enforce_eager !== false; syncToggleChip("chip-vllm-eager", "mo-vllm-eager"); }
  const tcol = $("#mo-vllm-tools");
  if (tcol) { tcol.checked = mv.enable_auto_tool_choice !== false; syncToggleChip("chip-vllm-tools", "mo-vllm-tools"); }
  const tpr = $("#mo-vllm-tool-parser");
  if (tpr) tpr.value = mv.tool_call_parser || "";
  const lay = $("#mo-vllm-layout");
  if (lay) { lay.value = mv.parallel_layout || "pp"; syncVllmLayoutChips(); }
  $("#mo-vllm-tokenizer").value = mv.tokenizer || "";
  $("#mo-vllm-extra-args").value = (mv.extra_args || []).join(", ");
  const reEnv = $("#mo-vllm-ray-env");
  if (reEnv) {
    const pairs = Object.entries(mv.ray_env || {})
      .map(([k, v]) => `${k}=${v}`);
    reEnv.value = pairs.join("\n");
  }
  $("#model-cancel").hidden = !editing;
  // Advanced is always open by default (wizard decision) — no collapsing logic
  const adv = $("#mo-advanced");
  if (adv) adv.open = true;
  try{ syncTensorChips(); syncOrderChips(); syncVllmLayoutChips(); syncToggleChip("chip-pinned","mo-pinned"); syncToggleChip("chip-flash-attn","flag-flash-attn"); syncToggleChip("chip-jinja","flag-jinja"); syncToggleChip("chip-vllm-eager","mo-vllm-eager"); syncToggleChip("chip-vllm-tools","mo-vllm-tools"); }catch(_){}
  updateAdvancedBadge();
  // variants
  renderVariants(m.variants || []);
  // show the additional-configs card whenever a model is being configured
  const vcard = $("#variants-card");
  if (vcard) vcard.style.display = "block";
  // add mode = 3-step wizard (Source → Hardware → Details); edit mode
  // follows the same wizard but starts at step 2, prefilled
  mowizShow(editing ? 2 : 1);
}

/* ---- configure-model wizard: Source → Hardware → Details ---- */
let mowizStep = 1;
function mowizEditing() { return Boolean(($("#model-original-id") || {}).value); }
function mowizShow(n) {
  mowizStep = Math.min(3, Math.max(1, n));
  for (const s of [1, 2, 3]) {
    const el = $("#mowiz-step-" + s);
    if (el) el.hidden = s !== mowizStep;
  }
  document.querySelectorAll("#mowiz-indicator li").forEach(li => {
    const s = parseInt(li.dataset.s, 10);
    li.classList.toggle("active", s === mowizStep);
    li.classList.toggle("done", s < mowizStep);
  });
  $("#mowiz-indicator").style.display = "";
  $("#mowiz-nav").style.display = "";
  const bar = $("#model-actions");
  if (bar) bar.style.display = mowizStep !== 3 ? "none" : "";
  $("#mowiz-back").disabled = mowizStep === 1;
  // Continue on the last step acts as Save
  const mowizNextBtn = $("#mowiz-next");
  if (mowizNextBtn) mowizNextBtn.textContent = mowizStep === 3 ? "Save ✓" : "Continue →";
  if (mowizEditing()) $("#model-form-title").textContent = `Edit ${$("#model-original-id").value}`;
  else $("#model-form-title").textContent = `Configure model — step ${mowizStep} of 3`;
  // landing back on Source re-syncs the list (download may have landed)
  if (mowizStep === 1 && $("#tab-models")?.classList.contains("active")) runScan();
}
function mowizHardwareCount() {
  const cl = $("#mo-servers-chip-list");
  return cl ? cl.querySelectorAll("input[type=checkbox]:checked").length : 0;
}
function mowizNext() {
  if (mowizStep === 1) {
    if (!$("#mo-path").value.trim()) { toast("Pick a scanned model or paste a model path", "bad"); return; }
    // refresh hardware chips for the picked path's group, then advance
    mowizShow(2);
    return;
  }
  if (mowizStep === 2) {
    if (!mowizHardwareCount()) { toast("Select at least one GPU/RAM chip", "bad"); return; }
    mowizShow(3);
    return;
  }
  if (mowizStep === 3) { saveModelForm($("#mowiz-next")); }
}
on("#mowiz-next", "click", mowizNext);
on("#mowiz-back", "click", () => { if (mowizStep > 1) mowizShow(mowizStep - 1); });
on("#mowiz-cancel", "click", () => { modelFormLoad({}, false); const vc = $("#variants-card"); if (vc) vc.style.display = "none"; });

/* ---- variants (additional configs per model) ---- */
function updateVariantsHint(){
  const hint = $("#mo-variants-hint");
  if (!hint) return;
  const n = document.querySelectorAll("#mo-variants-list .variant-card").length;
  hint.textContent = n ? `(${n})` : "";
}
async function renderVariants(variants){
  const list = $("#mo-variants-list");
  if (!list) return;
  list.innerHTML = "";
  const arr = variants || [];
  for (let i=0;i<arr.length;i++){
    await addVariantCard(arr[i]);
  }
  updateVariantsHint();
}
async function addVariantCard(data){
  const list = $("#mo-variants-list");
  if (!list) return;
  const card = document.createElement("div");
  card.className = "variant-card";
  card._data = data || {};
  card.innerHTML = `
    <div class="variant-summary">
      <div class="variant-summary-main">
        <div class="variant-summary-title">${escHtml(card._data.label || "config")}</div>
        <small class="variant-summary-sub"></small>
      </div>
      <div class="scan-model-actions">
        <button type="button" class="small variant-edit">Edit</button>
        <button type="button" class="small danger variant-remove">Remove</button>
      </div>
    </div>`;
  list.appendChild(card);
  card.querySelector(".variant-edit").onclick = () => openVariantModal(card);
  card.querySelector(".variant-remove").onclick = () => { card.remove(); updateVariantsHint(); };
  refreshVariantSummary(card);
  updateVariantsHint();
  return card;
}
function refreshVariantSummary(card){
  const d = card._data || {};
  const sub = card.querySelector(".variant-summary-sub");
  if (!sub) return;
  const parts = [];
  if (d.servers && d.servers.length) parts.push(`servers: ${d.servers.join(", ")}`);
  if (d.group) parts.push(`group: ${d.group}`);
  if (d.tensor_split_mode) parts.push(`split: ${d.tensor_split_mode}${d.tensor_split ? ` (${d.tensor_split})` : ""}`);
  const f = d.flags || {};
  // DEC-10 (T4-G1): summary follows the active engine's vocabulary —
  // under vllm the llama.cpp flags never render; the engine's own
  // `vllm:` overrides do
  if (manifestHiddenPanels().includes("tensor_split")) {
    const v = d.vllm || {};
    if (v.tensor_parallel_size) parts.push(`TP ${v.tensor_parallel_size}`);
    if (v.pipeline_parallel_size) parts.push(`PP ${v.pipeline_parallel_size}`);
    if (v.max_model_len) parts.push(`max-len ${v.max_model_len}`);
    if (v.kv_cache_dtype) parts.push(`kv ${v.kv_cache_dtype}`);
    if (v.gpu_memory_utilization) parts.push(`gmu ${v.gpu_memory_utilization}`);
    if (v.cpu_offload_gb) parts.push(`offload ${v.cpu_offload_gb}G`);
    if (v.quantization) parts.push(`quant ${v.quantization}`);
    if ((v.extra_args || []).length) parts.push(`${v.extra_args.length} extra`);
  } else {
    if (f["ctx-size"] || f["c"]) parts.push(`ctx: ${f["ctx-size"] || f["c"]}`);
    if (f["flash-attn"]) parts.push("flash");
    if (f["jinja"]) parts.push("jinja");
  }
  if (d.pinned) parts.push("pinned");
  sub.textContent = parts.length ? parts.join(" · ") : "no overrides — inherits primary";
}
/* ---- variant modal wizard (separate from the primary model wizard) ---- */
let variantModalCard = null;
/* DEC-10 (T4-G1): gate the modal per the active engine's manifest. When
   the engine hides the tensor_split panel (vllm), the llama.cpp blocks
   (device order / tensor placement / llama bin / flags override) hide
   and the engine's own v-vllm override card shows. Hidden-not-
   translated: nothing is converted, the fields simply are not that
   engine's vocabulary. */
function applyVariantModalEngineGate(modal){
  const lcppHidden = manifestHiddenPanels().includes("tensor_split");
  modal.querySelectorAll("[data-variant-lcpp]").forEach(el => { el.hidden = lcppHidden; });
  const vllmCard = modal.querySelector("[data-variant-vllm]");
  if (vllmCard) vllmCard.hidden = !lcppHidden;
  return lcppHidden;
}
/* read the modal's v-vllm override inputs into a `vllm:` section dict
   (empty-string fields are omitted → inherit the primary's values).
   Mirrors the primary's #mo-vllm-* save path field-for-field. */
function readVariantVllm(modal){
  const sec = {};
  const numOr = (cls) => {
    const t = (modal.querySelector("." + cls)?.value || "").trim();
    if (t === "") return null;
    const n = parseInt(t, 10);
    return Number.isFinite(n) ? n : null;
  };
  for (const [key, cls] of [
    ["max_model_len", "v-vllm-max-model-len"],
    ["max_num_seqs", "v-vllm-max-num-seqs"],
    ["tensor_parallel_size", "v-vllm-tp"],
    ["pipeline_parallel_size", "v-vllm-pp"],
    ["cpu_offload_gb", "v-vllm-cpu-offload"],
  ]) {
    const n = numOr(cls);
    if (n !== null) sec[key] = n;
  }
  // gmu is a float override (0–1); out-of-range input is dropped (same
  // rule as the primary's #mo-vllm-gmu)
  const gRaw = (modal.querySelector(".v-vllm-gmu")?.value || "").trim();
  if (gRaw !== "") {
    const g = parseFloat(gRaw);
    if (Number.isFinite(g) && g > 0 && g <= 1) sec.gpu_memory_utilization = g;
  }
  const kv = (modal.querySelector(".v-vllm-kv")?.value || "").trim();
  if (kv) sec.kv_cache_dtype = kv;
  const quant = (modal.querySelector(".v-vllm-quant")?.value || "").trim();
  if (quant) sec.quantization = quant;
  const lay = (modal.querySelector(".v-vllm-layout")?.value || "").trim();
  if (lay) sec.parallel_layout = lay;
  const dt = (modal.querySelector(".v-vllm-dtype")?.value || "").trim();
  if (dt) sec.dtype = dt;
  const eg = modal.querySelector(".v-vllm-eager");
  if (eg && eg.checked) sec.enforce_eager = true;
  const tcol = modal.querySelector(".v-vllm-tools");
  if (tcol && !tcol.checked) sec.enable_auto_tool_choice = false;
  const tpr = (modal.querySelector(".v-vllm-tool-parser")?.value || "").trim();
  if (tpr) sec.tool_call_parser = tpr;
  const tokz = (modal.querySelector(".v-vllm-tokenizer")?.value || "").trim();
  if (tokz) sec.tokenizer = tokz;
  // ray_env: same KEY=VALUE parsing as the primary's field
  const reTxt = (modal.querySelector(".v-vllm-ray-env")?.value || "").trim();
  if (reTxt) {
    const env = {};
    reTxt.split(/[\n,]/).map(x => x.trim()).filter(Boolean).forEach(pair => {
      const i = pair.indexOf("=");
      if (i < 1) return;
      const k = pair.slice(0, i).trim();
      const raw = pair.slice(i + 1).trim();
      if (!k) return;
      if (raw === "false") env[k] = false;
      else if (raw === "true") env[k] = true;
      else env[k] = raw;
    });
    if (Object.keys(env).length) sec.ray_env = env;
  }
  const extra = (modal.querySelector(".v-vllm-extra-args")?.value || "")
    .split(/[\n,]/).map(x => x.trim()).filter(Boolean);
  if (extra.length) sec.extra_args = extra;
  return Object.keys(sec).length ? sec : null;
}
function openVariantModal(card){
  variantModalCard = card;
  const modal = $("#variant-modal");
  if (!modal) return;
  const d = card._data || {};
  const f = d.flags || {};
  // DEC-10 (T4-G1): gate the modal to the active engine's vocabulary,
  // then load the engine's own override fields (below)
  const lcppHidden = applyVariantModalEngineGate(modal);
  // vLLM per-variant overrides: prefill from the variant's existing
  // `vllm:` section (blank = inherit the primary's values)
  const vSec = d.vllm || {};
  const vSet = (cls, val) => { const el = modal.querySelector("." + cls); if (el) el.value = val == null ? "" : String(val); };
  vSet("v-vllm-max-model-len", vSec.max_model_len);
  vSet("v-vllm-max-num-seqs", vSec.max_num_seqs);
  vSet("v-vllm-tp", vSec.tensor_parallel_size);
  vSet("v-vllm-pp", vSec.pipeline_parallel_size);
  vSet("v-vllm-gmu", vSec.gpu_memory_utilization);
  vSet("v-vllm-kv", vSec.kv_cache_dtype);
  vSet("v-vllm-quant", vSec.quantization);
  vSet("v-vllm-tokenizer", vSec.tokenizer);
  const vLay = modal.querySelector(".v-vllm-layout");
  if (vLay) vLay.value = vSec.parallel_layout || "";
  const vDt = modal.querySelector(".v-vllm-dtype");
  if (vDt) vDt.value = vSec.dtype || "";
  const vEg = modal.querySelector(".v-vllm-eager");
  if (vEg) vEg.checked = vSec.enforce_eager === true;
  const vTcol = modal.querySelector(".v-vllm-tools");
  if (vTcol) vTcol.checked = vSec.enable_auto_tool_choice !== false;
  const vTpr = modal.querySelector(".v-vllm-tool-parser");
  if (vTpr) vTpr.value = vSec.tool_call_parser || "";
  const vRe = modal.querySelector(".v-vllm-ray-env");
  if (vRe) {
    vRe.value = Object.entries(vSec.ray_env || {})
      .map(([k, val]) => `${k}=${val}`).join("\n");
  }
  const vExtraEl = modal.querySelector(".v-vllm-extra-args");
  if (vExtraEl) vExtraEl.value = (vSec.extra_args || []).join(", ");
  // label
  $("#variant-modal .v-label").value = d.label || "";
  // group
  const gsel = $("#variant-modal .v-group");
  gsel.innerHTML = '<option value="">(inherit)</option>';
  api("GET", "/api/groups").then(groups => {
    for (const g of groups) {
      const opt = document.createElement("option");
      opt.value = g.name; opt.textContent = g.name;
      if (d.group === g.name) opt.selected = true;
      gsel.appendChild(opt);
    }
  }).catch(() => {});
  // order / tensor — blank = inherit primary; primary value shown as
  // placeholder/title so the inherited advanced config is visible + editable
  // (mirrors the primary Advanced editor: auto/fit/split + weights field).
  let _pMode = "", _pSplit = "", _pFitM = "", _pFitR = "", _pOrder = "", _pBin = "";
  try {
    _pMode = $("#mo-tensor-mode")?.value || "";
    _pSplit = $("#mo-tensor-split")?.value || "";
    _pFitM = $("#mo-fit-margin")?.value || "";
    _pFitR = $("#mo-fit-ram-allow")?.value || "";
    _pOrder = $("#mo-order-policy")?.value || "";
    _pBin = $("#mo-llama-bin")?.value || "";
  } catch (_) {}
  // legacy "manual" value normalizes to "split" (same as primary loader)
  let _vMode = d.tensor_split_mode || "";
  if (_vMode === "manual") _vMode = "split";
  $("#variant-modal .v-order-policy").value = d.device_order_policy || "";
  $("#variant-modal .v-tensor-mode").value = _vMode;
  $("#variant-modal .v-fit-margin").value = d.fit_margin != null ? d.fit_margin : "";
  $("#variant-modal .v-fit-ram").value = d.fit_ram_allow != null ? d.fit_ram_allow : "";
  const _vSplitEl = $("#variant-modal .v-tensor-split");
  if (_vSplitEl) _vSplitEl.value = d.tensor_split || "";
  $("#variant-modal .v-llama-bin").value = d.llama_bin || "";
  // inherit hints from primary
  const _setPh = (sel, val) => { const el = modal.querySelector(sel); if (el) el.placeholder = val ? `inherit (${val})` : "inherit"; };
  _setPh(".v-fit-margin", _pFitM);
  _setPh(".v-fit-ram", _pFitR);
  if (_vSplitEl) _vSplitEl.placeholder = _pSplit ? `inherit (${_pSplit})` : "e.g. 8,10,10,6,10";
  const _binEl = modal.querySelector(".v-llama-bin");
  if (_binEl) _binEl.placeholder = _pBin ? `inherit (${_pBin})` : "inherit";
  const _ordSel = modal.querySelector(".v-order-policy");
  if (_ordSel) _ordSel.title = _pOrder ? `inherit: ${_pOrder}` : "inherit";
  const _tenSel = modal.querySelector(".v-tensor-mode");
  if (_tenSel) _tenSel.title = _pMode ? `inherit: ${_pMode}` : "inherit";
  applyVariantTensorVisibility();
  $("#variant-modal .v-pinned").checked = !!d.pinned;
  $("#variant-modal .v-ttl").value = (d.ttl && d.ttl != 0) ? d.ttl : "";
  // flags — prefill with effective (primary + override) so a slight tweak
  // doesn't require retyping everything. Save diffs back to primary to
  // preserve inherit where values are unchanged.
  let primaryFlags = {};
  try { primaryFlags = (typeof readFlags === "function") ? (readFlags() || {}) : {}; } catch (_) { primaryFlags = {}; }
  const eff = Object.assign({}, primaryFlags, f);
  $("#variant-modal .v-ctx").value = f["ctx-size"] || f["c"] || "";
  $("#variant-modal .v-ctx").placeholder = eff["ctx-size"] || eff["c"] ? `inherit (${eff["ctx-size"] || eff["c"]})` : "inherit";
  $("#variant-modal .v-flash").checked = f["flash-attn"] === "on" || f["flash-attn"] === true;
  $("#variant-modal .v-jinja").checked = !!f["jinja"];
  $("#variant-modal .v-cache-k").value = f["cache-type-k"] || "";
  $("#variant-modal .v-cache-v").value = f["cache-type-v"] || "";
  if (!$("#variant-modal .v-cache-k").querySelector(`option[value="${primaryFlags["cache-type-k"] || ""}"]`)) {}
  $("#variant-modal .v-cache-k").title = primaryFlags["cache-type-k"] ? `inherit: ${primaryFlags["cache-type-k"]}` : "inherit";
  $("#variant-modal .v-cache-v").title = primaryFlags["cache-type-v"] ? `inherit: ${primaryFlags["cache-type-v"]}` : "inherit";
  const commonKeys = new Set(["ctx-size","c","flash-attn","jinja","cache-type-k","cache-type-v"]);
  const variantExtra = Object.entries(f).filter(([k])=>!commonKeys.has(k)).map(([k,v])=> `--${k} ${v===true?"" : v}`.trim()).join("\n");
  const primaryExtra = Object.entries(primaryFlags).filter(([k])=>!commonKeys.has(k)).map(([k,v])=> `--${k} ${v===true?"" : v}`.trim()).join("\n");
  // If this config has no own extra flags, start from the primary's current
  // flags so the user can tweak rather than retype. If it already has
  // overrides, show those (existing behavior) — primary is still visible via title.
  const flagsTa = $("#variant-modal .v-flags-text");
  flagsTa.value = variantExtra || primaryExtra || "";
  flagsTa.title = variantExtra ? (primaryExtra ? `primary flags:\n${primaryExtra}` : "") : "showing primary flags — edit to create an override";
  // stash primary snapshot for diff-on-save
  try { modal._primaryFlags = JSON.parse(JSON.stringify(primaryFlags)); } catch (_) { modal._primaryFlags = primaryFlags; }
  // render hardware
  const hwContainer = $("#variant-modal-hw");
  renderVariantServers(d, hwContainer).then(() => {
    syncVariantModalChips();
    wireVariantModalCtxSync();
  });
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  variantModalShow(1);
}
function closeVariantModal(){
  const modal = $("#variant-modal");
  if (modal) { modal.classList.add("hidden"); modal.setAttribute("aria-hidden", "true"); }
  variantModalCard = null;
}
function variantModalShow(n){
  const modal = $("#variant-modal");
  if (!modal) return;
  modal._vstep = Math.min(3, Math.max(1, n));
  for (const s of [1, 2, 3]) {
    const el = modal.querySelector(".vwiz-step-" + s);
    if (el) el.hidden = s !== modal._vstep;
  }
  modal.querySelectorAll(".vwiz-indicator li").forEach(li => {
    const s = parseInt(li.dataset.s, 10);
    li.classList.toggle("active", s === modal._vstep);
    li.classList.toggle("done", s < modal._vstep);
  });
  const back = modal.querySelector(".vwiz-back");
  const next = modal.querySelector(".vwiz-next");
  const save = $("#variant-modal-save");
  if (back) back.disabled = modal._vstep === 1;
  if (next) next.hidden = modal._vstep === 3;
  if (save) save.hidden = modal._vstep !== 3;
  if (modal._vstep === 1) {
    const src = modal.querySelector(".v-source-display");
    if (src) {
      const p = ($("#mo-path") || {}).value || "";
      src.innerHTML = p
        ? `<i class="fa-solid fa-cube"></i> <code>${escHtml(p)}</code>`
        : `<span class="status-warn">No model file picked yet — set it in the primary Source step above.</span>`;
    }
  }
}
function variantModalNext(){
  const modal = $("#variant-modal");
  if (!modal) return;
  const step = modal._vstep || 1;
  if (step >= 3) return;
  if (step === 2) {
    const n = modal.querySelectorAll("#variant-modal-hw input[type=checkbox]:checked").length;
    if (!n) { toast("Select at least one GPU/RAM chip for this config", "bad"); return; }
  }
  variantModalShow(step + 1);
}
function variantModalSave(){
  const modal = $("#variant-modal");
  if (!modal || !variantModalCard) return;
  const label = modal.querySelector(".v-label").value.trim();
  if (!label) { toast("Label is required", "bad"); return; }
  const hwContainer = $("#variant-modal-hw");
  const selectedHardware = hwContainer?._selectedHardware || new Set();
  const deviceOrder = Array.from(selectedHardware).join(",");
  const exposeConfig = {};
  const machineNames = new Set();
  selectedHardware.forEach(hw => {
    const { machine: machineName, resource } = parseChip(hw);
    machineNames.add(machineName);
    if (!exposeConfig[machineName]) exposeConfig[machineName] = { gpus: [], ram: false };
    if (resource.startsWith("CUDA") || resource.startsWith("ROCm")) exposeConfig[machineName].gpus.push(resource);
    if (resource === "RAM") exposeConfig[machineName].ram = true;
  });
  const exposeUpdates = {};
  for (const [mn, hw] of Object.entries(exposeConfig)) {
    if (hw.gpus.length && hw.ram) exposeUpdates[mn] = { mode: "both", gpus: hw.gpus };
    else if (hw.gpus.length) exposeUpdates[mn] = { mode: "vram", gpus: hw.gpus };
    else if (hw.ram) exposeUpdates[mn] = { mode: "ram", gpus: [] };
  }
  const vGroup = modal.querySelector(".v-group").value || "";
  const vOrder = modal.querySelector(".v-order-policy").value || "";
  let vTensorMode = modal.querySelector(".v-tensor-mode").value || "";
  if (vTensorMode === "manual") vTensorMode = "split"; // legacy alias
  const vFitMargin = modal.querySelector(".v-fit-margin").value.trim() || "";
  const vFitRam = modal.querySelector(".v-fit-ram").value.trim() || "";
  const vTensorSplit = (modal.querySelector(".v-tensor-split")?.value || "").trim() || "";
  const vLlamaBin = modal.querySelector(".v-llama-bin").value.trim() || "";
  const vPinned = modal.querySelector(".v-pinned").checked || false;
  const vTtl = modal.querySelector(".v-ttl").value.trim() || "";
  const vCtx = modal.querySelector(".v-ctx").value.trim() || "";
  const vFlash = modal.querySelector(".v-flash").checked || false;
  const vJinja = modal.querySelector(".v-jinja").checked || false;
  const vCacheK = modal.querySelector(".v-cache-k").value || "";
  const vCacheV = modal.querySelector(".v-cache-v").value || "";
  const vExtraText = modal.querySelector(".v-flags-text").value || "";
  // DEC-10 (T4-G1): vLLM per-variant overrides — read the engine's own
  // card. When the llama.cpp surface is hidden (vllm), the llama.cpp
  // vocab keys are stripped from the payload instead of saved from
  // hidden inputs (device-order policy / tensor placement / llama bin /
  // flags are not this engine's fields — hidden-not-translated).
  const vllmOverride = readVariantVllm(modal);
  let vFlags = null;
  const hasFlagOverride = vCtx || vFlash || vJinja || vCacheK || vCacheV || vExtraText.trim();
  if (hasFlagOverride) {
    vFlags = {};
    if (vCtx) vFlags["ctx-size"] = isNaN(vCtx) ? vCtx : Number(vCtx);
    if (vFlash) vFlags["flash-attn"] = "on";
    if (vJinja) vFlags["jinja"] = true;
    if (vCacheK) vFlags["cache-type-k"] = vCacheK;
    if (vCacheV) vFlags["cache-type-v"] = vCacheV;
    if (vExtraText.trim()) Object.assign(vFlags, textToFlags(vExtraText));
    // diff against primary so untouched prefilled values stay inherited
    // (keeps "slight tweak" saves minimal instead of duplicating all flags)
    try {
      const base = (modal && modal._primaryFlags) || {};
      const sameVal = (a, b) => {
        if (a === b) return true;
        // flash-attn "on" vs true equivalence
        if ((a === "on" || a === true) && (b === "on" || b === true)) return true;
        return String(a) === String(b);
      };
      for (const k of Object.keys(vFlags)) {
        let bv = base[k];
        // ctx-size / c alias
        if ((k === "ctx-size" || k === "c") && bv === undefined) bv = base[k === "ctx-size" ? "c" : "ctx-size"];
        if (bv !== undefined && sameVal(vFlags[k], bv)) delete vFlags[k];
      }
      if (!Object.keys(vFlags).length) vFlags = null;
    } catch (_) {}
  }
  const lcppHidden = manifestHiddenPanels().includes("tensor_split");
  variantModalCard._data = {
    label,
    servers: Array.from(machineNames),
    device_order: (lcppHidden ? null : (deviceOrder || null)),
    device_order_policy: lcppHidden ? null : (vOrder || null),
    expose: Object.keys(exposeUpdates).length ? exposeUpdates : null,
    group: vGroup || null,
    tensor_split_mode: lcppHidden ? null : (vTensorMode || null),
    tensor_split: lcppHidden ? null : (vTensorSplit || null),
    fit_margin: lcppHidden ? null : (vFitMargin ? parseInt(vFitMargin,10) : null),
    fit_ram_allow: lcppHidden ? null : (vFitRam ? parseInt(vFitRam,10) : null),
    llama_bin: lcppHidden ? null : (vLlamaBin || null),
    flags: lcppHidden ? null : vFlags,
    // DEC-10 (T4-G1/A2): per-variant vllm overrides — saved from the
    // engine's own card (readVariantVllm); backend shallow-merges over
    // the primary's `vllm:` section (same partial-override semantics as
    // the flags merge in swap_manager).
    vllm: lcppHidden ? vllmOverride : ((variantModalCard._data || {}).vllm || null),
    pinned: vPinned ? true : null,
    ttl: vTtl ? parseInt(vTtl,10) : null
  };
  refreshVariantSummary(variantModalCard);
  closeVariantModal();
  toast(`Saved config ${label}`, "ok");
}
/* mirrors the primary applyTensorModeUI: fit shows margins, split shows
   weights. Blank (inherit) falls back to the primary's live mode so the
   visible fields always match what will actually apply. */
function applyVariantTensorVisibility(){
  const modal = $("#variant-modal");
  if (!modal) return;
  let mode = modal.querySelector(".v-tensor-mode")?.value || "";
  if (!mode) {
    try { mode = $("#mo-tensor-mode")?.value || "fit"; } catch (_) { mode = "fit"; }
  }
  if (mode === "manual") mode = "split";
  const fitWrap = modal.querySelector(".v-fit-wrap");
  const splitWrap = modal.querySelector(".v-split-wrap");
  if (fitWrap) fitWrap.hidden = mode !== "fit";
  if (splitWrap) splitWrap.hidden = mode !== "split";
}
function syncVariantModalChips(){
  const modal = $("#variant-modal");
  if (!modal) return;
  let tVal = modal.querySelector(".v-tensor-mode").value || "";
  if (tVal === "manual") { tVal = "split"; modal.querySelector(".v-tensor-mode").value = "split"; }
  modal.querySelectorAll(".v-chips-tensor-mode .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.value===tVal));
  const oVal = modal.querySelector(".v-order-policy").value || "";
  modal.querySelectorAll(".v-chips-order .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.value===oVal));
  [[".v-chip-flash",".v-flash"],[".v-chip-jinja",".v-jinja"],[".v-chip-pinned",".v-pinned"],[".v-chip-vllm-eager",".v-vllm-eager"],[".v-chip-vllm-tools",".v-vllm-tools"]].forEach(([chipSel,inpSel])=>{
    const chip = modal.querySelector(chipSel), inp = modal.querySelector(inpSel);
    if (chip && inp) { chip.classList.toggle("active", inp.checked); chip.dataset.checked = inp.checked ? "true":"false"; }
  });
  const layVal = modal.querySelector(".v-vllm-layout")?.value || "";
  modal.querySelectorAll(".v-chips-vllm-layout .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.value===layVal));
  applyVariantTensorVisibility();
}
function wireVariantModalCtxSync(){
  const modal = $("#variant-modal");
  if (!modal || modal._ctxWired) return;
  modal._ctxWired = true;
  const box = modal.querySelector(".v-ctx");
  const ta = modal.querySelector(".v-flags-text");
  if (!box || !ta) return;
  let syncing = false;
  // generic upsert of `--key value` inside the variant's raw-text view
  const upsertLine = (key, value) => {
    const v = (value ?? "").toString().trim();
    const re = new RegExp(`^\\s*--${key}(\\s|$|=)`);
    const out = [];
    let done = false;
    for (const line of ta.value.split("\n")) {
      if (re.test(line)) {
        if (!done && v) { out.push(`--${key} ${v}`); done = true; }
      } else out.push(line);
    }
    if (!done && v) out.push(`--${key} ${v}`);
    ta.value = out.join("\n");
  };
  box.addEventListener("input", () => {
    if (syncing) return; syncing = true;
    try {
      const v = box.value.trim();
      upsertLine("ctx-size", v);
      // keep ctx-size/c alias from disagreeing: a --c line mirrors ctx
      const hasC = /^\s*--c(\s|$|=)/m.test(ta.value);
      if (hasC) upsertLine("c", v);
    } finally { syncing = false; }
  });
  // KV cache dropdowns <-> raw text (same live binding as ctx above)
  for (const key of ["cache-type-k", "cache-type-v"]) {
    const sel = modal.querySelector(key === "cache-type-k" ? ".v-cache-k" : ".v-cache-v");
    if (!sel) continue;
    for (const evt of ["input", "change"]) sel.addEventListener(evt, () => {
      if (syncing) return; syncing = true;
      try { upsertLine(key, sel.value); } finally { syncing = false; }
    });
  }
  ta.addEventListener("input", () => {
    if (syncing) return;
    const f = textToFlags(ta.value);
    const ctx = f["ctx-size"] ?? f["c"] ?? null;
    const k = f["cache-type-k"], v = f["cache-type-v"];
    if (ctx === null || ctx === true) { /* leave ctx alone */ }
    else {
      syncing = true;
      try { if (box.value !== String(ctx)) box.value = String(ctx); } finally { syncing = false; }
    }
    for (const [key, val] of [["cache-type-k", k], ["cache-type-v", v]]) {
      if (val === undefined || val === true) continue; // absent or mid-typing bare flag
      const sel = modal.querySelector(key === "cache-type-k" ? ".v-cache-k" : ".v-cache-v");
      syncing = true;
      try { if (sel && sel.value !== String(val)) sel.value = String(val); } finally { syncing = false; }
    }
  });
}
on("#variant-modal-backdrop", "click", closeVariantModal);
on("#variant-modal-close", "click", closeVariantModal);
on("#variant-modal-cancel", "click", closeVariantModal);
on("#variant-modal .vwiz-back", "click", () => {
  const modal = $("#variant-modal");
  if (modal && (modal._vstep || 1) > 1) variantModalShow((modal._vstep || 1) - 1);
});
on("#variant-modal .vwiz-next", "click", variantModalNext);
on("#variant-modal-save", "click", variantModalSave);
document.querySelector("#variant-modal .v-tensor-mode")?.addEventListener("change", syncVariantModalChips);
document.querySelectorAll("#variant-modal .v-chips-tensor-mode .hw-chip").forEach(b => {
  b.addEventListener("click", () => {
    const modal = $("#variant-modal");
    if (!modal) return;
    modal.querySelector(".v-tensor-mode").value = b.dataset.value;
    syncVariantModalChips();
  });
});
document.querySelectorAll("#variant-modal .v-chips-order .hw-chip").forEach(b => {
  b.addEventListener("click", () => {
    const modal = $("#variant-modal");
    if (!modal) return;
    modal.querySelector(".v-order-policy").value = b.dataset.value;
    syncVariantModalChips();
  });
});
[[".v-chip-flash",".v-flash"],[".v-chip-jinja",".v-jinja"],[".v-chip-pinned",".v-pinned"],[".v-chip-vllm-eager",".v-vllm-eager"],[".v-chip-vllm-tools",".v-vllm-tools"]].forEach(([chipSel,inpSel])=>{
  document.querySelectorAll(`#variant-modal ${chipSel}`).forEach(chip => {
    chip.addEventListener("click", () => {
      const modal = $("#variant-modal");
      if (!modal) return;
      const inp = modal.querySelector(inpSel);
      inp.checked = !inp.checked;
      syncVariantModalChips();
    });
  });
});
document.querySelectorAll("#variant-modal .v-chips-vllm-layout .hw-chip").forEach(b => {
  b.addEventListener("click", () => {
    const modal = $("#variant-modal");
    if (!modal) return;
    modal.querySelector(".v-vllm-layout").value = b.dataset.value;
    syncVariantModalChips();
  });
});
async function renderVariantServers(vData, container){
  let machines = [];
  try { machines = await api("GET", "/api/machines"); } catch (_) { container.innerHTML = '<span class="status-bad">machines unavailable</span>'; return; }
  const modelServers = new Set(vData.servers || []);
  let selectedHardware = new Set();
  if (vData.device_order) {
    vData.device_order.split(",").forEach(d => { const t=d.trim(); if(t) selectedHardware.add(t); });
  } else if (modelServers.size) {
    for (const mac of machines) {
      if (modelServers.has(mac.name)) {
        const d = mac.detected || {};
        (d.gpu_names || []).forEach((_, i) => selectedHardware.add(`${mac.name}-${gpuPrefix(mac.gpu_vendor)}${i}`));
        if (d.cores) selectedHardware.add(`${mac.name}-RAM`);
      }
    }
  }
  container.innerHTML = machines.map(mac => {
    const d = mac.detected || {};
    const gpus = d.gpu_names || [];
    const pref = gpuPrefix(mac.gpu_vendor);
    const hasRAM = d.ram_total_gb && d.ram_total_gb > 0;
    const ramGB = hasRAM ? Math.round(d.ram_total_gb * 0.8) : 0;
    let gpuCheckboxes = gpus.map((gpu, i) => {
      const id = `${mac.name}-${pref}${i}`;
      const checked = selectedHardware.has(id) ? "checked" : "";
      return '<label class="hw-chip"><input type="checkbox" value="' + escHtml(id) + '" ' + checked + ' data-machine="' + escHtml(mac.name) + '"><i class="fa-solid fa-microchip chip-icon"></i><span class="chip-name">' + escHtml(gpu) + '</span><span class="chip-sub">' + pref + i + '</span></label>';
    }).join("");
    let ramCheckbox = "";
    if (hasRAM) {
      const id = `${mac.name}-RAM`;
      const checked = selectedHardware.has(id) ? "checked" : "";
      ramCheckbox = '<label class="hw-chip ram-chip"><input type="checkbox" value="' + escHtml(id) + '" ' + checked + ' data-machine="' + escHtml(mac.name) + '"><i class="fa-solid fa-memory chip-icon"></i><span class="chip-name">RAM (' + ramGB + ' GB)</span><span class="chip-sub">~80% of ' + escHtml(String(d.ram_total_gb)) + ' GB</span></label>';
    }
    return '<div class="machine-hw-group"><div class="machine-hw-header"><i class="fa-solid fa-server hw-server-icon"></i><strong>' + escHtml(mac.name) + '</strong> <small>' + escHtml(mac.ip) + '</small></div><div class="machine-hw-chips">' + gpuCheckboxes + ramCheckbox + '</div></div>';
  }).join("");
  container._selectedHardware = selectedHardware;
  container._machinesData = machines;
  try { renderDeviceOrderPreview(container, machines); }
  catch (e) { console.warn("[ui] device-order preview failed", e); }
  container.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.onchange = () => {
      if (cb.checked) container._selectedHardware.add(cb.value);
      else container._selectedHardware.delete(cb.value);
      const chip = cb.closest('.hw-chip, .ram-chip');
      if (chip) chip.classList.toggle('active', cb.checked);
      try { renderDeviceOrderPreview(container, container._machinesData || machines); }
      catch (e) { console.warn("[ui] device-order preview failed", e); }
    };
  });
  syncAllHwChips();
}
function collectVariants(){
  const list = $("#mo-variants-list");
  if (!list) return [];
  const out = [];
  list.querySelectorAll(".variant-card").forEach(card => {
    const d = card._data || {};
    const label = (d.label || "").trim();
    if (!label) return;
    out.push({
      label,
      servers: d.servers || [],
      device_order: d.device_order || null,
      device_order_policy: d.device_order_policy || null,
      expose: d.expose || null,
      group: d.group || null,
      tensor_split_mode: d.tensor_split_mode || null,
      tensor_split: d.tensor_split || null,
      fit_margin: d.fit_margin || null,
      fit_ram_allow: d.fit_ram_allow || null,
      llama_bin: d.llama_bin || null,
      flags: d.flags || null,
      // A2: per-variant vllm overrides round-trip untouched (backend
      // sanitizes against the same schema as the primary's section)
      vllm: d.vllm || null,
      pinned: d.pinned ? true : null,
      ttl: d.ttl || null
    });
  });
  return out;
}

function updateAdvancedBadge(){
  const badge = $("#mo-advanced-badge");
  if (!badge) return;
  let n = 0;
  const isVllm = (appSettings?.engine === "vllm");
  if (isVllm) {
    // Advanced IS the vLLM card in this engine — count its non-empty
    // overrides (the llama.cpp counters below tally hidden fields).
    const ids = ["mo-vllm-max-model-len","mo-vllm-max-num-seqs","mo-vllm-tp",
                 "mo-vllm-pp","mo-vllm-gmu","mo-vllm-cpu-offload",
                 "mo-vllm-quant","mo-vllm-tokenizer","mo-vllm-extra-args",
                 "mo-vllm-ray-env"];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el && el.value.trim()) n++;
    }
    if (!$("#mo-vllm-eager")?.checked) n++;   // eager off = override
    if (!$("#mo-vllm-tools")?.checked) n++;         // tools off = override
    if ($("#mo-vllm-layout")?.value !== "pp") n++; // non-default layout
    if ($("#mo-vllm-dtype")?.value) n++;
    if ($("#mo-vllm-kv")?.value) n++;
    if ($("#mo-vllm-tool-parser")?.value) n++;
  } else {
    const mode = $("#mo-tensor-mode")?.value;
    if (mode && mode!=="auto") n++;
    if ($("#mo-fit-margin")?.value && $("#mo-fit-margin").value!="1024") n++;
    if ($("#mo-fit-ram-allow")?.value && $("#mo-fit-ram-allow").value!="4096") n++;
    // DEC-10 (T4-G2): device-order policy is llama.cpp tensor-placement
    // vocabulary — under a manifest that hides the tensor_split panel it
    // is NOT collected and must not badge-count either
    if (!manifestHiddenPanels().includes("tensor_split")
        && $("#mo-order-policy")?.value && $("#mo-order-policy").value!=="as-selected") n++;
    if ($("#mo-llama-bin")?.value.trim()) n++;
    if ($("#mo-pinned")?.checked) n++;
    if ($("#mo-ttl")?.value.trim()) n++;
    const flagCount = document.querySelectorAll("#flags-container .flag-row").length;
    if (flagCount) n+= flagCount;
  }
  badge.textContent = n ? `(${n})` : "";
}
function syncCommonFlagsToRows(){
  // push common toggles into flag rows so they save
  const map = {
    "ctx-size": $("#flag-ctx-size")?.value.trim(),
    "flash-attn": $("#flag-flash-attn")?.checked ? "on" : "",
    "jinja": $("#flag-jinja")?.checked ? true : "",
    "cache-type-k": $("#flag-cache-k")?.value,
    "cache-type-v": $("#flag-cache-v")?.value,
  };
  for (const [k,v] of Object.entries(map)){
    // remove existing row for this key
    document.querySelectorAll("#flags-container .flag-row").forEach(row=>{
      const key = row.querySelector(".flag-key")?.value.trim().replace(/^--/,"");
      if (key===k) row.remove();
    });
    if (v!=="" && v!==undefined && v!==null && v!==false) addFlagRow(k, v);
  }
  upsertFlagLine($("#flag-ctx-size")?.value.trim() || "");
  updateAdvancedBadge();
}
// wire common flags -> badge
["flag-ctx-size","flag-cache-k","flag-cache-v"].forEach(id=>{ const el=document.getElementById(id); if(el) el.addEventListener("input", ()=>{ updateAdvancedBadge(); }); });
["flag-flash-attn","flag-jinja"].forEach(id=>{ const el=document.getElementById(id); if(el) el.addEventListener("change", ()=>{ updateAdvancedBadge(); }); });

/* ---- two-way Context binding: common input <-> flag rows <-> raw text ----
   Editing any one view updates the other two live (previously they only
   converged on mode-switch, so views silently disagreed mid-edit). */
let syncingCtx = false;
function upsertFlagRow(key, value) {
  let row = null;
  document.querySelectorAll("#flags-container .flag-row").forEach(r => {
    const k = r.querySelector(".flag-key")?.value.trim().replace(/^--/, "");
    if (k === key || (key === "ctx-size" && k === "c")) { if (!row) row = r; else r.remove(); }
  });
  const v = (value ?? "").toString().trim();
  if (!v) { if (row) row.remove(); return; }
  if (row) { row.querySelector(".flag-key").value = key; row.querySelector(".flag-value").value = v; }
  else addFlagRow(key, v);
}
function upsertFlagLine(value) {
  upsertFlagTextLine("ctx-size", value);
}
function setCtxViews(value, except) {
  if (syncingCtx) return;
  syncingCtx = true;
  try {
    const v = (value ?? "").toString();
    if (except !== "common") { const el = $("#flag-ctx-size"); if (el && el.value !== v) el.value = v; }
    if (except !== "rows") upsertFlagRow("ctx-size", v);
    if (except !== "text") upsertFlagLine(v);
    updateAdvancedBadge();
  } finally { syncingCtx = false; }
}
/* ---- two-way KV-cache binding: common dropdowns <-> flag rows <-> raw text ----
   Mirrors the Context binding: editing the cache-type-k/v dropdowns, a
   --cache-type-k/--cache-type-v chip row, or the raw text view updates all
   three live (previously the dropdowns only converged on save/mode-switch,
   so the views silently disagreed mid-edit). */
function setKvCacheViews(key, value, except) {
  if (syncingCtx) return;
  syncingCtx = true;
  try {
    const v = (value ?? "").toString().trim();
    const sel = key === "cache-type-k" ? $("#flag-cache-k") : $("#flag-cache-v");
    if (except !== "common" && sel && sel.value !== v) sel.value = v;
    if (except !== "rows") upsertFlagRow(key, v);
    if (except !== "text") upsertFlagTextLine(key, v);
    updateAdvancedBadge();
  } finally { syncingCtx = false; }
}
function upsertFlagTextLine(key, value) {
  const ta = $("#flags-text");
  if (!ta) return;
  const v = (value ?? "").toString().trim();
  const re = new RegExp(`^\\s*--${key}(\\s|$|=)`);
  const out = [];
  let done = false;
  for (const line of ta.value.split("\n")) {
    if (re.test(line)) {
      if (!done && v) { out.push(`--${key} ${v}`); done = true; }
      // else drop: duplicate line, or cleared value
    } else out.push(line);
  }
  if (!done && v) out.push(`--${key} ${v}`);
  const next = out.join("\n");
  if (ta.value !== next) ta.value = next;
}
(function wireKvCacheSync() {
  for (const key of ["cache-type-k", "cache-type-v"]) {
    const sel = document.getElementById(key === "cache-type-k" ? "flag-cache-k" : "flag-cache-v");
    if (sel) for (const evt of ["input", "change"]) sel.addEventListener(evt, () => setKvCacheViews(key, sel.value, "common"));
    const rows = document.getElementById("flags-container");
    if (rows) rows.addEventListener("input", (e) => {
      const row = e.target.closest(".flag-row");
      if (!row) return;
      const k = row.querySelector(".flag-key")?.value.trim().replace(/^--/, "");
      if (k === key) setKvCacheViews(key, row.querySelector(".flag-value").value.trim(), "rows");
    });
    const text = document.getElementById("flags-text");
    if (text) text.addEventListener("input", () => {
      const f = textToFlags(text.value);
      const v = f[key];
      if (v === undefined || v === true) return; // absent or mid-typing a bare flag
      setKvCacheViews(key, String(v), "text");
    });
  }
})();
(function wireCtxSync() {
  const common = document.getElementById("flag-ctx-size");
  if (common) common.addEventListener("input", () => setCtxViews(common.value.trim(), "common"));
  const rows = document.getElementById("flags-container");
  if (rows) rows.addEventListener("input", (e) => {
    const row = e.target.closest(".flag-row");
    if (!row) return;
    const k = row.querySelector(".flag-key")?.value.trim().replace(/^--/, "");
    if (k === "ctx-size" || k === "c") setCtxViews(row.querySelector(".flag-value").value.trim(), "rows");
  });
  const text = document.getElementById("flags-text");
  if (text) text.addEventListener("input", () => {
    const f = textToFlags(text.value);
    const v = f["ctx-size"] ?? f["c"] ?? null;
    if (v === null || v === true) return; // absent or mid-typing a bare flag — leave others alone
    setCtxViews(String(v), "text");
  });
})();
["mo-tensor-mode","mo-tensor-split","mo-fit-margin","mo-fit-ram-allow","mo-order-policy","mo-llama-bin","mo-pinned","mo-ttl"].forEach(id=>{ const el=document.getElementById(id); if(el) el.addEventListener("input", updateAdvancedBadge); if(el) el.addEventListener("change", updateAdvancedBadge); });

// card system for advanced — sync chips <-> hidden inputs (no stock radio/select)
function syncTensorChips(){ const v=$("#mo-tensor-mode")?.value; document.querySelectorAll("#chips-tensor-mode .hw-chip").forEach(b=>b.classList.toggle("active", b.dataset.value===v)); }
function syncOrderChips(){ const v=$("#mo-order-policy")?.value; document.querySelectorAll("#chips-order-policy .hw-chip").forEach(b=>b.classList.toggle("active", b.dataset.value===v)); }
function syncVllmLayoutChips(){ const v=$("#mo-vllm-layout")?.value || "pp"; document.querySelectorAll("#chips-vllm-layout .hw-chip").forEach(b=>b.classList.toggle("active", b.dataset.value===v)); }
function syncToggleChip(chipId, inputId){
  const chip=document.getElementById(chipId), inp=document.getElementById(inputId);
  if(!chip||!inp) return;
  const chk = inp.type==="checkbox" ? inp.checked : !!inp.value;
  chip.classList.toggle("active", !!chk);
  chip.dataset.checked = chk ? "true":"false";
}
document.querySelectorAll("#chips-tensor-mode .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=>{ $("#mo-tensor-mode").value=b.dataset.value; syncTensorChips(); applyTensorModeUI(); updateAdvancedBadge(); });
});
document.querySelectorAll("#chips-order-policy .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=>{ $("#mo-order-policy").value=b.dataset.value; syncOrderChips(); updateAdvancedBadge();
    const cl = $("#mo-servers-chip-list"); if (cl) renderDeviceOrderPreview(cl, cl._machinesData || []); });
});
document.querySelectorAll("#chips-vllm-layout .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=>{ $("#mo-vllm-layout").value=b.dataset.value; syncVllmLayoutChips(); updateAdvancedBadge(); });
});
for(const [chipId,inputId] of [["chip-pinned","mo-pinned"],["chip-flash-attn","flag-flash-attn"],["chip-jinja","flag-jinja"],["chip-vllm-eager","mo-vllm-eager"],["chip-vllm-tools","mo-vllm-tools"]]){
  const chip=document.getElementById(chipId), inp=document.getElementById(inputId);
  if(!chip||!inp) continue;
  chip.addEventListener("click", ()=>{
    if(inp.type==="checkbox") inp.checked=!inp.checked;
    syncToggleChip(chipId,inputId); updateAdvancedBadge();
    // keep hidden checkbox in sync for save
    inp.dispatchEvent(new Event("change",{bubbles:true}));
  });
  // keep chip in sync when input changes programmatically
  inp.addEventListener("change", ()=>syncToggleChip(chipId,inputId));
}
// global card system: any .hw-chip/.server-chip with a checkbox must show .active (fix for :has not firing / stale)
function syncAllHwChips(){
  document.querySelectorAll('.hw-chip, .server-chip, .ram-chip').forEach(chip=>{
    const inp = chip.querySelector('input[type="checkbox"]');
    if(inp) chip.classList.toggle('active', inp.checked);
  });
  // also sync the new card chips for tensor/device/pinned
  try{ syncTensorChips(); syncOrderChips(); }catch(_){}
  try{ syncToggleChip("chip-pinned","mo-pinned"); syncToggleChip("chip-flash-attn","flag-flash-attn"); syncToggleChip("chip-jinja","flag-jinja"); syncToggleChip("chip-vllm-eager","mo-vllm-eager"); syncToggleChip("chip-vllm-tools","mo-vllm-tools"); }catch(_){}
}
document.addEventListener('change', (e)=>{
  if(e.target.matches('.hw-chip input, .server-chip input, .ram-chip input')){
    const chip = e.target.closest('.hw-chip, .server-chip, .ram-chip');
    if(chip) chip.classList.toggle('active', e.target.checked);
  }
});
// also handle label click that toggles via bubbling (some browsers don't fire change immediately)
document.addEventListener('click', (e)=>{
  const chip = e.target.closest('.hw-chip, .server-chip, .ram-chip');
  if(chip && chip.querySelector('input[type="checkbox"]')){
    // delay to let checkbox state settle
    setTimeout(()=>chip.classList.toggle('active', chip.querySelector('input').checked), 0);
  }
});
window.addEventListener('DOMContentLoaded', ()=>setTimeout(syncAllHwChips, 600));
setTimeout(syncAllHwChips, 1000);

// presets
async function applyPreset(kind){
  if (kind==="single"){
    $("#mo-group").value="";
    $("#mo-tensor-mode").value="fit"; applyTensorModeUI();
    $("#mo-order-policy").value="as-selected";
    // pick first machine's hardware
    try{
      const machines = await api("GET","/api/machines");
      const first = machines.find(m=>m.role==="master")||machines[0];
      if(first) renderModelServers({servers:[first.name], device_order:""});
    }catch(_){}
  } else if (kind==="full"){
    let groups=[]; try{ groups=await api("GET","/api/groups"); }catch(_){}
    const full = groups.find(g=>g.machines.length>2) || groups[0];
    if(full){ $("#mo-group").value=full.name; $("#mo-group").dispatchEvent(new Event("change")); }
    $("#mo-tensor-mode").value="fit"; applyTensorModeUI(); $("#mo-order-policy").value="vram-first";
  } else if (kind==="moe"){
    $("#mo-order-policy").value="vram-first"; $("#mo-tensor-mode").value="fit"; applyTensorModeUI();
    $("#flag-ctx-size").value="160000"; $("#flag-flash-attn").checked=true; $("#flag-jinja").checked=true;
    $("#flag-cache-k").value="q8_0"; $("#flag-cache-v").value="q8_0";
    syncCommonFlagsToRows();
  }
  try{ syncTensorChips(); syncOrderChips(); syncVllmLayoutChips(); syncToggleChip("chip-pinned","mo-pinned"); syncToggleChip("chip-flash-attn","flag-flash-attn"); syncToggleChip("chip-jinja","flag-jinja"); syncToggleChip("chip-vllm-eager","mo-vllm-eager"); syncToggleChip("chip-vllm-tools","mo-vllm-tools"); }catch(_){}
  updateAdvancedBadge();
  toast(`Preset ${kind} applied`, "ok");
}
// presets removed per user request — Quick presets row deleted from template

async function saveModelForm(btn, skipArm = false) {
  // scratch ids (tuner leftovers) always resolve to their proper model:
  // tuning in the edit space saves a real config, never a tmp one
  const rawId = $("#mo-id").value.trim();
  const mid = rawId.replace(/(__tune_tmp.*|__ab_single.*)$/, "");
  if (mid !== rawId) {
    $("#mo-id").value = mid;
    toast(`Saved as proper model ${mid} (scratch suffix removed)`, "ok");
  }
  // are you sure — blank overwrite destroyed Huihui-35B before, so guard saves
  const isEdit = !!$("#model-original-id").value.trim();
  if (isEdit && !skipArm && !armed(btn, "Save")) return null;
  btn.disabled = true; const origLabel = btn.textContent; btn.textContent = "saving…";
  syncCommonFlagsToRows();
  const mode = $("#mo-tensor-mode").value;
  const chipList = $("#mo-servers-chip-list");
  
  const selectedHardware = new Set();
  chipList.querySelectorAll("input[type=checkbox]:checked").forEach(cb => {
    selectedHardware.add(cb.value);
  });

  const deviceOrder = Array.from(selectedHardware).join(",");

  const exposeConfig = {};
  const machineNames = new Set();
  selectedHardware.forEach(hw => {
    const { machine: machineName, resource } = parseChip(hw);
    machineNames.add(machineName);
    if (!exposeConfig[machineName]) exposeConfig[machineName] = { gpus: [], ram: false };
    if (resource.startsWith("CUDA") || resource.startsWith("ROCm")) exposeConfig[machineName].gpus.push(resource);
    if (resource === "RAM") exposeConfig[machineName].ram = true;
  });

  const exposeUpdates = {};
  for (const [machineName, hw] of Object.entries(exposeConfig)) {
    if (hw.gpus.length && hw.ram) exposeUpdates[machineName] = { mode: "both", gpus: hw.gpus };
    else if (hw.gpus.length) exposeUpdates[machineName] = { mode: "vram", gpus: hw.gpus };
    else if (hw.ram) exposeUpdates[machineName] = { mode: "ram", gpus: [] };
  }

  // mid = sanitized proper id from the top of this function
  if (!mid) { toast("Model ID is required — cannot save blank model", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
  if (mid.includes("/") || mid.includes("\\") || mid === "." || mid === "..") { toast("Model ID cannot contain / or \\", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
  const rt = $("#mo-runtime").value || "llama";
  // runtime: custom serves no GGUF — the bat path is the required
  // artifact, not the model path
  if (rt === "custom") {
    if (!$("#mo-custom-bat").value.trim()) { toast("Bat path on box is required for custom models", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
  } else if (!$("#mo-path").value.trim()) { toast("Model path is required", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
  const grp = $("#mo-group")?.value?.trim() || "";
  const variants = collectVariants();
  // validate variant labels client-side
  const seen = new Set();
  for (const v of variants) {
    if (seen.has(v.label)) { toast(`duplicate variant label '${v.label}'`, "bad"); btn.disabled=false; btn.textContent=origLabel; return null; }
    seen.add(v.label);
  }
  const data = {
    model_id: mid,
    original_model_id: $("#model-original-id").value.trim(),
    name: $("#mo-name").value.trim(),
    base_path: $("#mo-path").value.trim(),
    runtime: $("#mo-runtime").value || "llama",
    tensor_split_mode: $("#mo-tensor-mode").value,
    tensor_split: ($("#mo-tensor-mode").value === "split")
      ? $("#mo-tensor-split").value.trim() : "",
    fit_margin: parseInt($("#mo-fit-margin").value, 10) || 1024,
    fit_ram_allow: parseInt($("#mo-fit-ram-allow").value, 10) || 0,
    llama_bin: $("#mo-llama-bin").value.trim(),
    servers: Array.from(machineNames),
    group: grp,
    pinned: $("#mo-pinned")?.checked || false,
    ttl: parseInt($("#mo-ttl")?.value, 10) || 0,
    device_order: deviceOrder,
    device_order_policy: $("#mo-order-policy").value,
    expose: exposeUpdates,
    flags: readFlags(),
    variants,
    reset: !!($("#model-reset")?.checked),
  };
  // runtime: custom — carry the kind/bat/port section so the saved config
  // keeps it (the backend validates it with the same strict schema as
  // plan time); any other runtime drops a stale section instead of
  // filing a model that is both GGUF and custom
  if (rt === "custom") {
    const kind = $("#mo-custom-kind").value || "winbox-bat";
    if (kind === "raw") {
      // user-supplied swap entry: strict client-side shape check (the
      // backend re-validates with the same schema — 400 on violations)
      const cmd = $("#mo-custom-cmd").value.trim();
      if (!cmd) { toast("Cmd is required for raw custom models", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
      const sec = { kind, cmd };
      const proxy = $("#mo-custom-proxy").value.trim();
      if (proxy) {
        if (!/^https?:\/\/[^/\s]+(:\d+)?(\/.*)?$/.test(proxy)) { toast("Proxy must be an http(s) URL", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
        sec.proxy = proxy;
      }
      const ep = $("#mo-custom-ep").value.trim();
      if (ep) {
        if (!ep.startsWith("/")) { toast("Check endpoint must start with /", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
        if (!proxy) { toast("Check endpoint needs Proxy set", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
        sec.checkEndpoint = ep;
      }
      const cmdStop = $("#mo-custom-cmdstop").value.trim();
      if (cmdStop) sec.cmdStop = cmdStop;
      const env = {};
      for (const line of $("#mo-custom-env").value.split("\n").map(x => x.trim()).filter(Boolean)) {
        const i = line.indexOf("=");
        if (i < 1) { toast(`Bad env line (need KEY=VALUE): ${line}`, "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
        env[line.slice(0, i).trim()] = line.slice(i + 1).trim();
      }
      if (Object.keys(env).length) sec.env = env;
      const script = $("#mo-custom-script").value;
      if (script.trim()) sec.script = script;
      if (cmd.includes("{SCRIPT}") && !script.trim()) { toast("Cmd references {SCRIPT} but no Script provided", "bad"); btn.disabled = false; btn.textContent = origLabel; return null; }
      data.custom = sec;
    } else {
      const pRaw = ($("#mo-custom-port").value || "").trim();
      const pNum = parseInt(pRaw, 10);
      data.custom = {
        kind,
        bat: $("#mo-custom-bat").value.trim(),
        port: Number.isFinite(pNum) ? pNum : 8080,
      };
    }
    // vision flag applies to both custom kinds (listed for vision tests)
    if ($("#mo-custom-vision")?.checked) data.custom.vision = true;
  } else {
    data.custom = null;
  }
  // DEC-10 (T4-G2): tensor-placement vocabulary is llama.cpp-only. When
  // the active engine's manifest hides the tensor_split panel, strip the
  // keys from the payload so a vllm model can never silently file
  // device-order policy the swap config ignores (hidden-not-translated —
  // the values are simply not this engine's fields).
  if (manifestHiddenPanels().includes("tensor_split")) {
    data.tensor_split_mode = "auto";
    data.tensor_split = "";
    data.fit_margin = 1024;
    data.fit_ram_allow = 1024;
    data.device_order = "";
    data.device_order_policy = "as-selected";
    data.expose = {};
  }
  // per-model vllm section — only when the active engine is vllm; empty
  // fields are omitted so no-op keys never reach the config file
  if ((appSettings && appSettings.engine) === "vllm") {
    const numOr = (id) => {
      const t = ($(id)?.value || "").trim();
      if (t === "") return null;
      const n = parseInt(t, 10);
      return Number.isFinite(n) ? n : null;
    };
    const sec = {};
    for (const [key, id] of [
      ["max_model_len", "#mo-vllm-max-model-len"],
      ["max_num_seqs", "#mo-vllm-max-num-seqs"],
      ["tensor_parallel_size", "#mo-vllm-tp"],
      ["pipeline_parallel_size", "#mo-vllm-pp"],
      ["cpu_offload_gb", "#mo-vllm-cpu-offload"],
    ]) {
      const n = numOr(id);
      if (n !== null) sec[key] = n;
    }
    // A2: gmu is a float override (0–1); empty = inherit the settings
    // default. Out-of-range input is dropped here — the backend 400s it.
    const gmRaw = ($("#mo-vllm-gmu")?.value || "").trim();
    if (gmRaw !== "") {
      const g = parseFloat(gmRaw);
      if (Number.isFinite(g) && g > 0 && g <= 1) sec.gpu_memory_utilization = g;
    }
    const kv = ($("#mo-vllm-kv")?.value || "").trim();
    if (kv) sec.kv_cache_dtype = kv;
    const quant = ($("#mo-vllm-quant")?.value || "").trim();
    if (quant) sec.quantization = quant;
    const dt = ($("#mo-vllm-dtype")?.value || "").trim();
    if (dt && dt !== "auto") sec.dtype = dt;
    // eager/tools are hw-chip toggles now (default ON): the stored
    // section carries the explicit bool, matching launch behavior
    const eg = $("#mo-vllm-eager");
    if (eg) sec.enforce_eager = eg.checked;
    // tools: explicit opt-out only... stored as bool either way so the
    // editor round-trips what it shows
    const tcol = $("#mo-vllm-tools");
    if (tcol) sec.enable_auto_tool_choice = tcol.checked;
    const tpr = $("#mo-vllm-tool-parser");
    if (tpr && tpr.value) sec.tool_call_parser = tpr.value;
    // multi-host layout: pp (layer split) is the default — only store an
    // explicit choice; absent keeps the build-time pp default
    const lay = $("#mo-vllm-layout");
    if (lay && lay.value && lay.value !== "pp") sec.parallel_layout = lay.value;
    const tokz = ($("#mo-vllm-tokenizer")?.value || "").trim();
    if (tokz) sec.tokenizer = tokz;
    const extra = ($("#mo-vllm-extra-args")?.value || "")
      .split(/[\n,]/).map(x => x.trim()).filter(Boolean);
    if (extra.length) sec.extra_args = extra;
    // ray_env: KEY=VALUE per line; bare KEY=false (literal) drops the
    // built-in default. Parsed leniently — the backend sanitizes and
    // rejects malformed names/shapes with a 400.
    const reTxt = ($("#mo-vllm-ray-env")?.value || "").trim();
    if (reTxt) {
      const env = {};
      reTxt.split(/[\n,]/).map(x => x.trim()).filter(Boolean).forEach(pair => {
        const i = pair.indexOf("=");
        if (i < 1) return;
        const k = pair.slice(0, i).trim();
        const raw = pair.slice(i + 1).trim();
        if (!k) return;
        if (raw === "false") env[k] = false;
        else if (raw === "true") env[k] = true;
        else env[k] = raw;
      });
      if (Object.keys(env).length) sec.ray_env = env;
    }
    if (Object.keys(sec).length) data.vllm = sec;
  }
  let ok = false;
  try {
    const saved = await api("POST", "/api/models", data);
    modelFormLoad({}, false);
    loadModels();
    // the save hook pushes the regenerated swap config (variants included);
    // report what became routable instead of a bare "saved"
    const push = (saved && saved._swap_push) || {};
    const rst = (saved && saved._reset) || null;
    const routes = push.models || [];
    const mine = routes.filter(r => r === mid || r.startsWith(mid + "--"));
    if (rst) {
      const nStopped = (rst.stopped || []).length;
      const errs = (rst.errors || []).length ? ` · reset errors: ${rst.errors.join("; ")}` : "";
      toast(`Cluster reset: stopped ${nStopped} target(s)${errs}`, errs ? "warn" : "ok", 8000);
    }
    if (push.pushed) {
      const warn = (push.warnings || []).length ? ` · warnings: ${push.warnings.join("; ")}` : "";
      toast(`Saved ${mid} — swap routes live: ${mine.length ? mine.join(", ") : routes.length + " total"}${warn}`, push.warnings && push.warnings.length ? "warn" : "ok", 9000);
      if (saved && saved._swap_reload) pollSwapReload(mid);
    } else {
      toast(`Saved ${mid} (${push.reason || push.error || "swap push skipped"} — Launch All to apply)`, "warn", 9000);
    }
    ok = true;
  } catch (err) { toast(err.message, "bad"); }
  btn.disabled = false; btn.textContent = origLabel;
  // clear armed state after save
  delete btn.dataset.armed;
  return ok ? mid : null;
}
on("#model-save", "click", async (e) => { await saveModelForm(e.currentTarget); });
// runtime: custom shows the kind/bat/port row (GGUF path + tensor panels
// don't apply — the server runs externally); anything else hides it
function syncCustomRow() {
  const isCustom = (($("#mo-runtime") || {}).value || "llama") === "custom";
  const row = $("#mo-custom-row");
  if (row) row.hidden = !isCustom;
  // per-kind inputs: winbox-bat takes bat/port, raw takes the entry
  const kind = ($("#mo-custom-kind") || {}).value || "winbox-bat";
  const bat = $("#mo-custom-bat-fields");
  if (bat) bat.style.display = (isCustom && kind === "winbox-bat") ? "contents" : "none";
  const raw = $("#mo-custom-raw");
  if (raw) raw.hidden = !(isCustom && kind === "raw");
}
on("#mo-runtime", "change", syncCustomRow);
on("#mo-custom-kind", "change", syncCustomRow);
// save-triggered daemon reload: poll until the backend reports completion
async function pollSwapReload(mid) {
  toast(`Reloading swap so ${mid} is usable…`, "info", 4000);
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    await new Promise((res) => setTimeout(res, 2000));
    let st = null;
    try { st = await api("GET", "/api/swap/reload-status"); } catch (_) { continue; }
    if (st && !st.active && (st.done || st.error)) {
      if (st.error) { toast(`Swap reload failed: ${st.error}`, "bad", 9000); return; }
      const ok = st.done && st.done.running;
      toast(ok ? `Swap reloaded — ${mid} ready to load` : `Swap reload finished but daemon not healthy`, ok ? "ok" : "bad", 9000);
      return;
    }
  }
  toast("Swap reload still running (timed out waiting) — check Launch tab", "warn", 9000);
}
on("#model-cancel", "click", () => { modelFormLoad({}, false); const vc = $("#variants-card"); if (vc) vc.style.display = "none"; });
on("#mo-add-variant", "click", async () => {
  const list = $("#mo-variants-list");
  const n = list ? list.children.length : 0;
  const label = `config-${n+1}`;
  const card = await addVariantCard({label, servers:[], device_order:"", expose:{}});
  if (card) openVariantModal(card);
});
on("#mo-group", "change", async () => {
  const grp = $("#mo-group").value.trim();
  if (!grp) return;
  try {
    const g = await api("GET", `/api/groups`);
    const found = g.find(x => x.name === grp);
    if (!found || !(found.machines||[]).length) return;
    // build a temp model that pins to group's machines so renderModelServers selects them
    const cur = {
      model_id: $("#mo-id").value.trim(),
      name: $("#mo-name").value.trim(),
      base_path: $("#mo-path").value.trim(),
      servers: found.machines,
      device_order: "", // let renderModelServers auto-pick all GPUs for those machines
      group: grp,
    };
    // keep existing manual selections? For affinity we reset to group's full hardware
    renderModelServers(cur);
    toast(`Pinned to group ${grp}: ${found.machines.join(", ")}`, "ok");
  } catch (e) { toast(e.message, "bad"); }
});
let selectedModelId = null;
let selectedScanPath = null;

async function loadModels() {
  const body = $("#models-list");
  // only show the spinner on first load; refreshes update in place (no flicker)
  if (body && !body.querySelector(".arch-section") && !body.querySelector(".scan-model"))
    body.innerHTML = loadingSpinner("Loading models…");
  const models = await api("GET", "/api/models");
  // filename -> model_id map so scan rows reuse configured models' icons
  scanFileIconKey = {};
  for (const m of models) {
    const f = (m.base_path || "").split("/").pop();
    if (f && m.model_id) scanFileIconKey[f] = m.model_id;
  }
  // blank-id guard: legacy .yaml file would cause 405 on delete; auto-cleanup via backend
  const blanks = models.filter(m => !(m.model_id || "").trim());
  if (blanks.length) {
    const box = $("#model-test-results");
    if (box) box.innerHTML = `<p class="status-bad">Found ${blanks.length} blank model config(s) — removing…</p>`;
    try { await api("POST", "/api/models/cleanup", {}); } catch (e) { /* ignore */ }
    // reload without blanks
    const filtered = models.filter(m => (m.model_id || "").trim());
    if (filtered.length !== models.length) {
      // re-fetch clean list and recurse once
      return loadModels();
    }
  }
  body.innerHTML = "";
  const valid = models.filter(m => (m.model_id || "").trim());
  // group by architecture, then sort each group alphabetically by model_id
  const groups = new Map();
  for (const m of valid) {
    const arch = (m.architecture || "unknown").toLowerCase();
    if (!groups.has(arch)) groups.set(arch, []);
    groups.get(arch).push(m);
  }
  const sortedArchs = Array.from(groups.keys()).sort();
  for (const arch of sortedArchs) {
    const archModels = groups.get(arch).sort((a, b) =>
      (a.model_id || "").toLowerCase().localeCompare((b.model_id || "").toLowerCase()));
    const { sec, archBody } = makeArchSection(body, "models:" + arch, arch, archModels.length);
    for (const m of archModels) {
      const tr = document.createElement("div");
      tr.classList.add("scan-model");
      tr.dataset.modelId = m.model_id;
      if (selectedModelId === m.model_id) tr.classList.add("row-selected");
      const vllmMode = manifestHasSection("vllm_section");
      // DEC-10 (T4-G4): per-engine row rendering. Under vllm the
      // llama.cpp columns (servers / tensor-split / llama.cpp flags)
      // and the Tune button (409s — _refuse_tuner_in_vllm) hide; the
      // row subtitle shows the engine's own fields instead, same idiom
      // as modelSubVllm on the launch cards. Under llama.cpp nothing
      // changes (verbatim columns).
      const sub1 = vllmMode
        ? `${escHtml((m.base_path || "").split("/").pop())}${m.group ? ` · group: ${escHtml(m.group)}` : ""} · ${escHtml(modelSubVllm(m))}`
        : `${escHtml((m.base_path || "").split("/").pop())}${m.group ? ` · group: ${escHtml(m.group)}` : ""} · servers: ${escHtml((m.servers || []).join(", ")) || "-"}${m.pinned ? ' · <span class="badge" style="background:var(--ok-dim);color:var(--ok)">pinned</span>' : ""}`;
      const sub2 = vllmMode
        ? `${(m.variants||[]).length ? ` · configs: ${escHtml((m.variants||[]).map(v=>v.label).join(" | "))}` : ""}${m.pinned ? ' · <span class="badge" style="background:var(--ok-dim);color:var(--ok)">pinned</span>' : ""}`
        : `${escHtml(((m.tensor_split_mode === "manual" ? "fit" : m.tensor_split_mode || "auto")) + " " + Object.entries(m.flags || {}).map(([k, v]) => `${k}=${v}`).join(" ").trim() || "-")}${(m.variants||[]).length ? ` · variants: ${escHtml((m.variants||[]).map(v=>`${v.label}(${(v.servers||[]).join(",")})`).join(" | "))}` : ""}`;
      tr.innerHTML = `
      <div class="scan-main"><div class="scan-title">${modelIconHTML(m.model_id)}${escHtml(m.model_id)} ${(m.variants||[]).length ? `<span class="badge" style="background:var(--accent-dim);color:var(--accent)" title="${escHtml((m.variants||[]).map(v=>v.label).join(", "))}">+${(m.variants||[]).length} configs</span>` : ""}${m.decision_type ? ` <span class="badge" style="background:var(--ok-dim);color:var(--ok)" title="TypeSafe decision model — serves /v1/systemone only (no chat generation)">decision</span>` : ""}</div>
        <small>${sub1}</small>
        <small>${sub2}</small></div>
      <div class="scan-model-actions">
        <button class="small" data-act="edit">Edit</button>
        <button class="small" data-act="check">Check</button>
        ${vllmMode ? "" : `<button class="small" data-act="tune" title="Auto-tune: ctx, KV, MTP, ubatch — saves fastest >=20 tok/s">Tune</button>`}
        <button class="small" data-act="pin">${m.pinned ? "Unpin" : "Pin"}</button>
        <button class="small danger" data-act="delfiles">Delete files</button>
        <button class="small danger" data-act="delete">Delete</button>
      </div>`;
      tr.querySelector('[data-act="edit"]').onclick = () => {
        selectedModelId = m.model_id;
        body.querySelectorAll(".scan-model").forEach(r =>
          r.classList.toggle("row-selected", r.dataset.modelId === m.model_id));
        modelFormLoad(m, true);
      };
      tr.querySelector('[data-act="check"]').onclick = (e) => testModel(m.model_id, e.currentTarget);
      const tuneBtn = tr.querySelector('[data-act="tune"]');
      if (tuneBtn) tuneBtn.onclick = (e) => tuneStart(m.model_id, e.currentTarget);
      tr.querySelector('[data-act="delfiles"]').onclick = async (e) => {
        const btn = e.currentTarget;
        if (!armed(btn, "Delete files")) return;
        btn.disabled = true; btn.textContent = "deleting…";
        try {
          const r = await api("DELETE", `/api/models/${encodeURIComponent(m.model_id)}/file?delete_file=true`, {});
          const del = (r.deleted || []).map(p => p.split("/").pop()).join(", ");
          toast(r.errors && r.errors.length
            ? `Delete files failed: ${r.errors.join(" | ")}`
            : `Deleted files for ${m.model_id}: ${del || "none found"}${r.config_removed ? " + config removed" : ""}`,
            r.errors && r.errors.length ? "bad" : "ok");
        } catch (err) { toast(`Delete files failed: ${err.message}`, "bad"); }
        btn.disabled = false; btn.textContent = "Delete files";
        await loadModels();
        runScan();  // files changed → re-sync the Source list
      };
      tr.querySelector('[data-act="pin"]').onclick = async (e) => {
        const btn = e.currentTarget;
        btn.disabled = true;
        try {
          const next = !m.pinned;
          await api("POST", `/api/models/${encodeURIComponent(m.model_id)}/pin`, { pinned: next });
          toast(`${m.model_id} ${next ? "pinned" : "unpinned"}`, "ok");
          await loadModels();
        } catch (err) { toast(`Pin failed: ${err.message}`, "bad"); btn.disabled = false; }
      };
      tr.querySelector('[data-act="delete"]').onclick = async (e) => {
        const btn = e.currentTarget;
        if (!armed(btn, "Delete")) return;
        btn.disabled = true; btn.textContent = "deleting…";
        try {
          await api("DELETE", `/api/models/${encodeURIComponent(m.model_id)}`);
          toast(`Deleted model ${m.model_id}`, "ok");
          if (selectedModelId === m.model_id) selectedModelId = null;
          await loadModels();
        } catch (err) {
          toast(`Delete failed: ${err.message}`, "bad");
          btn.disabled = false; btn.textContent = "Delete";
        }
      };
      /* click empty space of a row -> select/highlight it so the user
         sees which model they're configuring or deleting */
      tr.addEventListener("click", (ev) => {
        if (ev.target.closest("button")) return;
        selectedModelId = (selectedModelId === m.model_id) ? null : m.model_id;
        body.querySelectorAll(".scan-model").forEach(r =>
          r.classList.toggle("row-selected", r.dataset.modelId === selectedModelId));
        if (selectedModelId) modelFormLoad(m, true);
      });
      archBody.appendChild(tr);
      resolveModelIcon(m.model_id);
    }
    body.appendChild(sec);
  }
}

/* keep-warm active */
async function loadActiveModels() {
  const box = $("#active-models");
  if (!box) return;
  try {
    const resp = await api("GET", "/api/swap/loaded");
    const loaded = resp.loaded || [];
    if (!resp.running) { box.innerHTML = `<p class="hint">Swap not running</p>`; return; }
    if (!loaded.length) { box.innerHTML = `<p class="hint">No models loaded (they stay loaded until you evict — never via timeout)</p>`; return; }
    box.innerHTML = loaded.map(id=>`<div class="row" style="justify-content:space-between;align-items:center;padding:6px 8px;border-bottom:1px solid var(--border-soft)"><span><span class="badge" style="background:var(--ok-dim);color:var(--ok)">loaded</span> ${escHtml(id)}</span><button class="small danger" data-evict="${escHtml(id)}">Evict</button></div>`).join("");
    box.querySelectorAll("[data-evict]").forEach(btn=>{
      btn.onclick = async () => {
        const id = btn.dataset.evict;
        if (!armed(btn, "Evict")) return;
        btn.disabled = true; btn.textContent = "evicting…";
        try { await api("POST", `/api/swap/evict/${encodeURIComponent(id)}`, {}); toast(`Evicted ${id}`, "ok"); loadActiveModels(); } catch(e){ toast(e.message,"bad"); btn.disabled=false; btn.textContent="Evict"; }
      };
    });
  } catch (e) { box.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`; }
}
on("#evict-all", "click", async (e)=>{
  const btn = e.currentTarget;
  if (!armed(btn, "Evict all")) return;
  btn.disabled = true; btn.textContent = "evicting…";
  try { await api("POST", "/api/swap/evict-all", {}); toast("Evicted all", "ok"); loadActiveModels(); } catch(err){ toast(err.message,"bad"); }
  btn.disabled = false; btn.textContent = "Evict all";
});
// refresh active models when entering Launch tab
const _origLaunch = typeof loadLaunch === "function" ? loadLaunch : null;
const _origTabHandler = document.querySelectorAll("nav button");
// hook via tab click already calls loadLaunch; also poll active on that tab
on('nav button[data-tab="launch"]', "click", () => setTimeout(loadActiveModels, 300));

const escHtml = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function testModel(modelId, btn) {
  console.log("[ui] check click", modelId);
  const box = $("#model-test-results");
  if (!box) { toast("UI is stale — hard-refresh with Ctrl+Shift+R and retry.", "bad"); return; }
  box.innerHTML = `<p>Checking ${escHtml(modelId)}…</p>`;
  // fetch directly (not api()): the report body matters on failure too
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/test`, {
      method: "POST", headers: { "Content-Type": "application/json" },
    });
    const r = await res.json();
    const items = (r.issues || []).map(i =>
      `<li class="${i.level === "error" ? "status-bad" : "status-warn"}">[${i.level}] ${escHtml(i.msg)}</li>`).join("");
    box.innerHTML = `<p class="${r.ok ? "status-ok" : "status-bad"}">Check ${escHtml(modelId)}: ${r.ok ? "OK" : "FAILED"}</p>`
      + `<p><small>--device: ${escHtml(r.devices || "-")}</small></p>`
      + (r.fit_targets ? `<p><small>--fit-target: ${escHtml(r.fit_targets)}</small></p>` : "")
      + (items ? `<ul>${items}</ul>` : "<p>no issues found</p>");
  } catch (err) {
    box.innerHTML = `<p class="status-bad">check ${escHtml(modelId)} failed: request error</p><pre></pre>`;
    box.querySelector("pre").textContent = err.message;
  }
}

/* ---- auto-tune: ONE button. Saves current form inputs as the config,
   applies safe defaults, math-estimates max ctx, live-verifies down to
   >=20 tok/s (MTP on iff faster), saves winner.
   Hardware MUST be selected in the form (servers + GPU chips). */
function formHardware() {
  const chipList = $("#mo-servers-chip-list");
  const hw = new Set();
  if (chipList) chipList.querySelectorAll("input[type=checkbox]:checked").forEach(cb => hw.add(cb.value));
  const machines = new Set();
  hw.forEach(c => { const { machine } = parseChip(c); if (machine) machines.add(machine); });
  const flags = readFlags();
  return {
    base_path: $("#mo-path").value.trim(),
    servers: Array.from(machines),
    device_order: Array.from(hw).join(","),
    mmproj: flags["mmproj"] || "",
  };
}

let _tuneTimer = null;
async function tuneStart(modelId, btn) {
  const box = $("#model-test-results");
  if (!box) { toast("UI stale — hard-refresh", "bad"); return; }
  // single click starts (no two-click arm: Tune IS the confirm)
  box.innerHTML = `<p>Tune ${escHtml(modelId)} starting… (restarts RPC peers!)</p>`;
  try {
    // no target_tps: server picks topology default (25 local, 20 RPC)
    await api("POST", `/api/models/${encodeURIComponent(modelId)}/tune/start`, { try_mtp_off: true });
  } catch (err) { box.innerHTML = `<p class="status-bad">Tune failed to start: ${escHtml(err.message)}</p>`; return; }
  if (_tuneTimer) clearInterval(_tuneTimer);
  _tuneTimer = setInterval(async () => {
    try {
      const s = await api("GET", `/api/models/${encodeURIComponent(modelId)}/tune/status`);
      const log = (s.log || []).slice(-15).map(l => escHtml(l)).join("\n");
      const w = s.winner ? `ctx=${s.winner["ctx-size"]} kv=${s.winner.kv} mtp=${s.winner.mtp}${s.winner.mtp_n ? "/n" + s.winner.mtp_n : ""} split=${s.winner.split || "fit"} ub=${s.winner.ubatch} fa=${s.winner.flash} ${s.winner.tps} tok/s headroom=${s.winner.min_free_gb ?? "?"}GB ${s.winner.safe ? "SAFE" : "UNSAFE"}` : "";
      box.innerHTML = `<p>Tune ${escHtml(modelId)}: <b>${escHtml(s.state)}</b>`
        + (s.winner ? ` — WINNER ${escHtml(w)} (saved)` : "")
        + (s.detail ? ` — ${escHtml(s.detail)}` : "")
        + `</p><pre>${log}</pre>`
        + (s.state === "running" ? "" : `<div class="row"><button class="small" id="tune-reload">Reload models</button></div>`);
      if (s.state !== "running") {
        clearInterval(_tuneTimer); _tuneTimer = null;
        const rb = $("#tune-reload");
        if (rb) rb.onclick = () => loadModels();
        if (s.ok && s.winner) toast(`Tuned ${modelId}: ctx ${s.winner["ctx-size"]} split ${s.winner.split || "fit"} ub ${s.winner.ubatch} ${s.winner.tps} tok/s`, "ok");
      }
    } catch (err) { /* keep polling */ }
  }, 5000);
}

on("#model-tune", "click", async (e) => {
  // NEW-model flow: save current inputs as the config first, then tune it.
  // (skip the Save two-click arm — Tune is explicit enough)
  const mid = await saveModelForm(e.currentTarget, true);
  if (!mid) return;  // validation failed, toast already shown
  tuneStart(mid, null);
});

/* ---- device-order preview: show the resolved --device token chain as
   chips are clicked, mirroring the backend _map_devices/_vram_first_key
   logic (master GPUs keep native tokens, each peer GPU/RAM becomes its
   own RPC<N> numbered in --rpc order; vram-first re-sorts output). ---- */
function chipIdx(r){ const d=(r||"").replace(/\D/g,""); return d ? parseInt(d,10) : 0; }

function buildDeviceOrderTokens(chips, machines, policy){
  const byName = {};
  machines.forEach(m => byName[m.name] = m);
  const masterName = (machines.find(m => m.role === "master") || machines[0] || {}).name || "";
  // peers in --rpc order = first appearance of a peer chip
  const peers = [];
  chips.forEach(c => {
    const {machine: mn} = parseChip(c);
    if (mn && mn !== masterName && !peers.includes(mn)) peers.push(mn);
  });
  // RPC numbering per peer: selected GPUs by index, then RAM last
  const rpcOf = {}; let n = 0;
  peers.forEach(pn => {
    const mac = byName[pn] || {};
    const pref = gpuPrefix(mac.gpu_vendor);
    const gpus = chips
      .map(c => parseChip(c))
      .filter(p => p.machine === pn && p.resource !== "RAM")
      .map(p => p.resource)
      .sort((a, b) => chipIdx(a) - chipIdx(b));
    gpus.forEach(r => { rpcOf[pn + "|" + r] = "RPC" + (n++); });
    if (chips.some(c => c === pn + "-RAM")) rpcOf[pn + "|RAM"] = "RPC" + (n++);
  });
  // tokens in chip order
  const out = [];
  const seen = new Set();
  chips.forEach(c => {
    const {machine: mn, resource: res} = parseChip(c);
    if (!mn) { if (!seen.has(c)) { seen.add(c); out.push({token: c, machine: "", resource: res}); } return; }
    if (mn === masterName) {
      if (res === "RAM") return;               // host RAM is implicit
      const key = "M|" + res;
      if (!seen.has(key)) { seen.add(key); out.push({token: res, machine: mn, resource: res, host: true}); }
      return;
    }
    const key = "P|" + mn + "|" + res;
    if (!seen.has(key)) {
      seen.add(key);
      out.push({token: rpcOf[mn + "|" + res] || "?", machine: mn, resource: res});
    }
  });
  if (policy === "vram-first") {
    // mirrors _vram_first_key: VRAM tier before RAM; host leads; then
    // machine memory desc (VRAM for GPUs, RAM for RAM); then index
    const vram = mn => ((byName[mn] || {}).detected || {}).vram_total_gb || 0;
    const ram  = mn => ((byName[mn] || {}).detected || {}).ram_total_gb || 0;
    const tier = t => (t.resource === "RAM") ? 1 : 0;
    out.sort((a, b) => {
      const ta = tier(a), tb = tier(b);
      if (ta !== tb) return ta - tb;
      const ha = a.host ? 0 : 1, hb = b.host ? 0 : 1;
      if (ha !== hb) return ha - hb;
      const va = ta ? ram(a.machine) : vram(a.machine);
      const vb = tb ? ram(b.machine) : vram(b.machine);
      if (vb !== va) return vb - va;
      const ia = chipIdx(a.token), ib = chipIdx(b.token);
      if (ia !== ib) return ia - ib;
      return a.token < b.token ? -1 : (a.token > b.token ? 1 : 0);
    });
  }
  return out;
}

function renderDeviceOrderPreview(container, machines){
  if (!container) return;
  let host = container.querySelector(".device-order-preview");
  if (!host) {
    host = document.createElement("div");
    host.className = "device-order-preview";
    container.appendChild(host);
  }
  const policySel = $("#mo-order-policy");
  const policy = policySel ? policySel.value : "as-selected";
  const chips = Array.from(container._selectedHardware || []);
  const tokens = buildDeviceOrderTokens(chips, machines, policy);
  const label = policy === "vram-first"
    ? "--device order (vram-first: master GPUs, then peers by VRAM, RAM last)"
    : "--device order (as-clicked)";
  const pills = tokens.map(t =>
    `<span class="dev-pill${t.host ? " dev-host" : ""}${t.resource === "RAM" ? " dev-ram" : ""}">`
    + `<strong>${escHtml(t.token)}</strong><small>${escHtml(t.machine || "local")}</small></span>`);
  const body = tokens.length
    ? pills.join('<i class="fa-solid fa-arrow-right dev-arrow"></i>')
    : '<span class="hint">select hardware chips above — the order shows here live</span>';
  host.innerHTML = `<div class="dev-order-label">${escHtml(label)}</div><div class="dev-pills">${body}</div>`;
}

async function renderModelServers(m) {
  const chipList = $("#mo-servers-chip-list");
  const modelServers = new Set(m.servers || []);
  let machines = [];
  try { machines = await api("GET", "/api/machines"); }
  catch (_) { chipList.innerHTML = '<span class="status-bad">machines unavailable</span>'; return; }

  let selectedHardware = new Set();
  if (m.device_order) {
    m.device_order.split(",").forEach(d => selectedHardware.add(d.trim()));
  } else if (modelServers.size) {
    for (const mac of machines) {
      if (modelServers.has(mac.name)) {
        const d = mac.detected || {};
        (d.gpu_names || []).forEach((_, i) => selectedHardware.add(`${mac.name}-${gpuPrefix(mac.gpu_vendor)}${i}`));
        if (d.cores) selectedHardware.add(`${mac.name}-RAM`);
      }
    }
  }

  chipList.innerHTML = machines.map(mac => {
    const d = mac.detected || {};
    const gpus = d.gpu_names || [];
    const pref = gpuPrefix(mac.gpu_vendor);
    const hasRAM = d.ram_total_gb && d.ram_total_gb > 0;
    const ramGB = hasRAM ? Math.round(d.ram_total_gb * 0.8) : 0;

    let gpuCheckboxes = gpus.map((gpu, i) => {
      const id = `${mac.name}-${pref}${i}`;
      const checked = selectedHardware.has(id) ? "checked" : "";
      return '<label class="hw-chip"><input type="checkbox" value="' + id + '" ' + checked + ' data-machine="' + mac.name + '"><span class="chip-name">' + escHtml(gpu) + '</span><span class="chip-sub">' + pref + i + '</span></label>';
    }).join("");

    let ramCheckbox = "";
    if (hasRAM) {
      const id = `${mac.name}-RAM`;
      const checked = selectedHardware.has(id) ? "checked" : "";
      ramCheckbox = '<label class="hw-chip ram-chip"><input type="checkbox" value="' + id + '" ' + checked + ' data-machine="' + mac.name + '"><span class="chip-name">RAM (' + ramGB + ' GB)</span><span class="chip-sub">~80% of ' + d.ram_total_gb + ' GB</span></label>';
    }

    return '<div class="machine-hw-group"><div class="machine-hw-header"><strong>' + mac.name + '</strong> <small>' + mac.ip + '</small></div><div class="machine-hw-chips">' + gpuCheckboxes + ramCheckbox + '</div></div>';
  }).join("");

  chipList._selectedHardware = selectedHardware;
  chipList._machinesData = machines;
  chipList.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.onchange = () => {
      if (cb.checked) selectedHardware.add(cb.value);
      else selectedHardware.delete(cb.value);
      // vLLM is multi-host now: checking a chip while other machines are
      // selected just announces the forming cluster. llama.cpp
      // multi-select is untouched.
      if (cb.checked && appSettings && appSettings.engine === "vllm") {
        const keep = parseChip(cb.value).machine;
        const machinesSet = new Set();
        chipList.querySelectorAll("input[type=checkbox]").forEach(other => {
          if (other.checked) machinesSet.add(parseChip(other.value).machine);
        });
        if (machinesSet.size > 1) {
          toast(`multi-host vLLM — ${machinesSet.size} machines will form the tensor-parallel cluster (master: ${keep})`, "info", 8000);
        }
      }
      // sync card visual
      const chip = cb.closest('.hw-chip, .ram-chip');
      if (chip) chip.classList.toggle('active', cb.checked);
      updateServersFromHardware(selectedHardware, machines, chipList);
      try { renderDeviceOrderPreview(chipList, machines); }
      catch (e) { console.warn("[ui] device-order preview failed", e); }
    };
  });
  // initial sync for pre-checked (selectedHardware)
  syncAllHwChips();

  updateServersFromHardware(selectedHardware, machines, chipList);
  try { renderDeviceOrderPreview(chipList, machines); }
  catch (e) { console.warn("[ui] device-order preview failed", e); }
}

/* Hardware chips are "<machine>-<resource>" (e.g. "aibox1-CUDA0",
   "box2-RAM"). Machine names may themselves contain hyphens, so always
   split on the LAST hyphen -- mirroring _parse_chip() on the backend.
   One grammar, parsed the same way on both sides. */
function parseChip(hw) {
  const i = (hw || "").lastIndexOf("-");
  if (i < 0) return { machine: "", resource: hw || "" };
  return { machine: hw.slice(0, i), resource: hw.slice(i + 1) };
}

function updateServersFromHardware(selectedHardware, machines, chipList) {
  const machineNames = new Set();
  selectedHardware.forEach(hw => {
    machineNames.add(parseChip(hw).machine);
  });
  
  let hiddenInput = $("#mo-servers-hidden");
  if (!hiddenInput) {
    hiddenInput = document.createElement("input");
    hiddenInput.type = "hidden";
    hiddenInput.id = "mo-servers-hidden";
    hiddenInput.name = "servers";
    chipList.parentNode.insertBefore(hiddenInput, chipList);
  }
  hiddenInput.value = Array.from(machineNames).join(",");
}


/* ---------------- engine manifest (settings engine + ui_manifest) -------
   GET /api/settings returns {engine, ui_manifest{labels, hidden_panels,
   hidden_tabs, model_editor{sections}, flag_presets, launch, downloads},
   ui_hints{...same view, deprecated shim}}. The engine adapter OWNS the
   vocabulary — no engine flag lists are hardcoded here (AUDIT-T6 §1.5).
   applyEngineHints is now a manifest renderer: sections/panels/tabs are
   gated from the manifest, not per-element hardcoded (Cards 10/11 emit
   from model_editor.sections; until they land, the template's gated
   nodes consult manifest.hidden_panels / hidden_tabs). */

/* DEC-10 (T4-G10): manifest accessors — every gated surface asks the
   ACTIVE engine's ui_manifest, never a hardcoded engine-name list. The
   manifest is the complete field inventory; a surface listed there is
   native vocabulary (show), absent = foreign (hidden-not-translated). */
function engineManifest() {
  const s = appSettings || {};
  return (s.ui_manifest && Object.keys(s.ui_manifest).length)
    ? s.ui_manifest : (s.ui_hints || {});
}
/* does the active engine's model_editor surface include `secId`? */
function manifestHasSection(secId) {
  const secs = ((engineManifest().model_editor || {}).sections) || [];
  if (secs.some(sec => sec && sec.id === secId)) return true;
  // engine-name fallback ONLY when no manifest is cached (older backend
  // / stale cache — same rule as applyEngineHints' ownsVllm)
  const man = (appSettings || {}).ui_manifest;
  if (!man || !Object.keys(man).length) {
    if (secId === "vllm_section") return (appSettings || {}).engine === "vllm";
  }
  return false;
}
/* llama.cpp-only panels the active engine hides (vllm lists them) */
function manifestHiddenPanels() {
  return engineManifest().hidden_panels || [];
}
function applyEngineHints(s) {
  if (!s) return;
  const engine = s.engine || "llama_cpp";
  const manifest = s.ui_manifest || {};
  // legacy shim view: only read when the manifest itself is missing
  // (older backend / stale cache) so the payload's ui_hints stays unused
  const hints = Object.keys(manifest).length ? manifest : (s.ui_hints || {});
  // engine-owned vocab signals from the manifest (launch.mode + the
  // editor sections list) — engine-name fallback only when the manifest
  // itself is missing (stale cache / older backend)
  const sections = ((hints.model_editor || {}).sections) || [];
  const ownsVllm = sections.some(sec => sec && sec.id === "vllm_section")
    || (!Object.keys(manifest).length && engine === "vllm");
  const launchMode = ((hints.launch || {}).mode)
    || (engine === "vllm" ? "swap_only" : "rpc_cluster");
  const hidden = hints.hidden_panels || [];
  // DEC-12: the Launch tab renders engine-native views from the
  // manifest's launch.mode (the adapter's launch semantics, DEC-06) —
  // no engine-name comparisons. rpc_cluster keeps the llama.cpp copy
  // (RPC peers + per-model grid); swap_only is vLLM's own view (daemon
  // launch + HTTP /health prose; the rpc_launch grid hides via
  // hidden_panels and this view replaces it).
  document.querySelectorAll("[data-launch-view]").forEach(el => {
    el.hidden = el.dataset.launchView !== launchMode;
  });
  // llama.cpp-only panels hide when the active engine lists them
  document.querySelectorAll("[data-panel]").forEach(el => {
    el.hidden = hidden.includes(el.dataset.panel);
  });
  // engine-exclusive test tabs hide the same way (nav button + section)
  // — e.g. the Audio tab is llama.cpp-only (G-8). The Access button is
  // EXCLUDED from manifest control: its visibility is trust-gated
  // (accessGateCheck / loadAccess), not engine-gated — without this
  // exclusion a settings load re-unhid it for restricted users.
  const hiddenTabs = hints.hidden_tabs || [];
  document.querySelectorAll("nav button[data-tab]").forEach(el => {
    if (el.dataset.tab === "access") return;  // trust-gated, not engine-gated
    el.hidden = hiddenTabs.includes(el.dataset.tab);
  });
  document.querySelectorAll("section.tab[id^='tab-']").forEach(el => {
    const name = el.id.slice(4);
    if (name === "access") {
      // trust-gated (accessGateCheck), never engine-affected — an
      // engine manifest must not resurrect a hidden Access tab
      return;
    }
    if (hiddenTabs.includes(name)) {
      el.hidden = true;
      el.classList.remove("active");
      document.querySelectorAll(`nav button[data-tab="${name}"]`).forEach(b => b.classList.remove("active"));
    } else {
      el.hidden = false;
    }
  });
  // engine-exclusive panels show only for their engine
  document.querySelectorAll("[data-engine]").forEach(el => {
    el.hidden = el.dataset.engine !== engine;
  });
  // DEC-10 (T4-G3): the model editor's surfaces render FROM the
  // manifest's model_editor.sections — a surface the active engine
  // doesn't list hides (vllm lists only vllm_section, so the llama.cpp
  // flags fieldset + common-flags card disappear; hidden-not-
  // translated). Panels above already gate via hidden_panels.
  const activeSecs = sections.map(sec => sec && sec.id).filter(Boolean);
  if (activeSecs.length) {
    document.querySelectorAll("[data-editor-section]").forEach(el => {
      el.hidden = !activeSecs.includes(el.dataset.editorSection);
    });
  }
  // DEC-11 (T4-G5): the Downloads tab renders ONLY the active engine's
  // pane — the manifest's downloads.formats drives it, no engine-name
  // comparisons. Single-format engines hide the format toggle entirely
  // (nothing to choose — a hidden format never reaches POST and can't
  // 409); engines listing both keep the chips. hfFormat is pinned to
  // the engine's format so the first search asks for the right one.
  const dlFormats = ((hints.downloads || {}).formats) || [];
  if (dlFormats.length) {
    if (dlFormats.length === 1) {
      const only = dlFormats[0] === "hf" ? "safetensors" : dlFormats[0];
      const chips = $("#hf-fmt-chips");
      if (chips) chips.parentElement.style.display = "none";
      if (hfFormat !== only) hfSetFormat(only);
      document.querySelectorAll("#hf-fmt-chips .hw-chip").forEach(b => {
        b.hidden = b.dataset.fmt !== only;
        b.classList.toggle("active", b.dataset.fmt === only);
      });
    } else {
      const chips = $("#hf-fmt-chips");
      if (chips) chips.parentElement.style.display = "";
      document.querySelectorAll("#hf-fmt-chips .hw-chip").forEach(b => {
        b.hidden = !dlFormats.includes(
          b.dataset.fmt === "safetensors" ? "hf" : (b.dataset.fmt || "gguf"));
      });
    }
    // custom-link pane speaks the engine's vocabulary: llama.cpp points
    // at single .gguf files; vLLM downloads whole HF repos. The copy
    // (hint + placeholder) and the repo flag follow the manifest.
    const vllmDl = dlFormats.includes("hf") && !dlFormats.includes("gguf");
    const dlHint = $("#dl-pane-custom p.hint");
    if (dlHint) dlHint.textContent = vllmDl
      ? "Paste an hf:// repo link (e.g. hf://Qwen/Qwen3-0.6B — the whole model repo downloads together: config.json + *.safetensors)."
      : "Paste an hf:// link (e.g. hf://orcarouter/Repo/file.gguf — split models download all shards together). The file streams straight to the store location on disk; the HF cache is bypassed entirely.";
    const dlInput = $("#dl-link");
    if (dlInput) dlInput.placeholder = vllmDl
      ? "hf://Qwen/Qwen3-0.6B"
      : "hf://orcarouter/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-F16-00001-of-00002.gguf";
  }
  // relabels (server / peers / probe) come from the adapter's labels
  document.querySelectorAll("[data-engine-label]").forEach(el => {
    const lbl = (hints.labels || {})[el.dataset.engineLabel];
    if (lbl) el.textContent = lbl;
  });
  // FIX-A7 (t_cc420d42): Start All is NOT hidden under vLLM — its backend
  // (/api/start-all) is engine-aware: it skips the RPC phase and still
  // launches llama-swap. The old blanket hide of llama.cpp-only controls
  // is retired; relabel instead so the action reads honestly in each
  // mode — wording from launch.mode (swap_only = daemon-only start).
  const startAll = $("#start-all");
  if (startAll) {
    if (launchMode === "swap_only") {
      startAll.textContent = "Start llama-swap";
      startAll.title = "Launch the llama-swap daemon (RPC phase skipped — vLLM serves through llama-swap)";
    } else {
      startAll.textContent = "Start All";
      startAll.title = "Start RPC servers on every machine, then llama-swap";
    }
  }
  const eng = $("#set-engine");
  if (eng) eng.value = engine;
  // the settings vllm card shows when the manifest lists the engine's
  // vllm_section editor surface (engine-name fallback for stale caches)
  const vcard = $("#set-vllm");
  if (vcard) vcard.hidden = !ownsVllm;
  const v = s.vllm || {};
  const del = $("#set-vllm-delivery");
  if (del) del.value = v.delivery || "venv";
  const vb = $("#set-vllm-venv-bin");
  if (vb) vb.value = v.venv_bin || "";
  const di = $("#set-vllm-docker-image");
  if (di) di.value = v.docker_image || "";
  const gmu = (typeof v.default_gpu_memory_utilization === "number")
    ? v.default_gpu_memory_utilization : 0.9;
  const slider = $("#set-vllm-gpu-mem");
  if (slider) {
    slider.value = gmu;
    const out = $("#set-vllm-gpu-mem-v");
    if (out) out.textContent = Number(gmu).toFixed(2);
  }
  syncVllmDeliveryRows();
  renderFlagPresets();
}

function syncVllmDeliveryRows() {
  const d = ($("#set-vllm-delivery")?.value) || "venv";
  document.querySelectorAll("[data-vllm-delivery]").forEach(el => {
    el.hidden = el.dataset.vllmDelivery !== d;
  });
}

/* Preset chips in the flags editor, built from ui_manifest.flag_presets.
   llama.cpp: click adds a generic flag row. vLLM: schema fields fill the
   per-model vllm card directly; anything else appends to extra args
   (vLLM's launch command reads extra_args, not the generic flags dict). */
function renderFlagPresets() {
  const box = $("#flag-presets");
  if (!box) return;
  // DEC-02: presets come from the engine's ui_manifest (the legacy
  // ui_hints payload key is a deprecated shim — Card 15 retires it)
  const src = (appSettings && (appSettings.ui_manifest || appSettings.ui_hints)) || {};
  const presets = src.flag_presets || {};
  const names = Object.keys(presets);
  if (!names.length) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = '<small class="hint" style="flex-basis:100%">presets from the active engine — click to add</small>'
    + names.map(n => {
      const p = presets[n] || {};
      const note = p.note ? ` title="${escHtml(p.note)}"` : "";
      return `<button type="button" class="small flag-preset" data-preset="${escHtml(n)}"${note}>${escHtml(p.flag || n)}</button>`;
    }).join("");
  box.querySelectorAll(".flag-preset").forEach(btn => {
    btn.onclick = () => {
      const p = presets[btn.dataset.preset] || {};
      const flag = String(p.flag || "").replace(/^--/, "");
      if (!flag) return;
      // route by the manifest's editor surface: an engine whose
      // model_editor lists vllm_section fills the vllm card; anything
      // else appends a generic flag row. Engine-name fallback only when
      // no manifest is cached (older backend — same rule as
      // applyEngineHints' ownsVllm).
      const man = (appSettings && appSettings.ui_manifest) || {};
      const hasManifest = Object.keys(man).length > 0;
      const secs = ((hasManifest ? man
        : ((appSettings || {}).ui_hints || {})).model_editor || {}).sections || [];
      if (secs.some(sec => sec && sec.id === "vllm_section")
        || (!hasManifest && (appSettings && appSettings.engine) === "vllm"))
        applyVllmPreset(flag, p);
      else addFlagRow(flag, "");
    };
  });
}

function applyVllmPreset(flag, p) {
  const fieldInputs = {
    "max-model-len": "#mo-vllm-max-model-len",
    "max-num-seqs": "#mo-vllm-max-num-seqs",
    "tensor-parallel-size": "#mo-vllm-tp",
    "pipeline-parallel-size": "#mo-vllm-pp",
    "cpu-offload-gb": "#mo-vllm-cpu-offload",
    "quantization": "#mo-vllm-quant",
  };
  const example = String(p.placeholder || "").replace(/^e\.g\.\s*/, "").split(/[,;]/)[0].trim();
  if (fieldInputs[flag]) {
    const el = $(fieldInputs[flag]);
    if (el) { el.value = example; el.focus(); }
    return;
  }
  if (flag === "kv-cache-dtype") {
    const sel = $("#mo-vllm-kv");
    if (sel) sel.value = (p.choices || []).find(c => c !== "auto") || (p.choices || [""])[0] || "";
    return;
  }
  const inp = $("#mo-vllm-extra-args");
  if (!inp) return;
  const token = p.flag || flag;
  const have = inp.value.split(/[\n,]/).map(x => x.trim()).filter(Boolean);
  if (have.includes(token)) return;
  inp.value = have.length ? have.join(", ") + ", " + token : token;
}

/* launch-card subtitle under vllm — reads the per-model vllm section.
   Mirrors the editor's Advanced card field-for-field so the launch page
   shows the actual configured launch options, not a stale subset. */
function modelSubVllm(m) {
  const v = (m && m.vllm) || {};
  const bits = ["vLLM"];
  if (v.parallel_layout) bits.push(v.parallel_layout === "pp" ? "PP layer-split" : "TP tensor-split");
  if (v.max_model_len) bits.push(`max-len ${v.max_model_len}`);
  if (v.tensor_parallel_size) bits.push(`TP ${v.tensor_parallel_size}`);
  if (v.pipeline_parallel_size) bits.push(`PP ${v.pipeline_parallel_size}`);
  if (v.dtype && v.dtype !== "auto") bits.push(v.dtype);
  if (v.enforce_eager) bits.push("eager");
  if (v.enable_auto_tool_choice !== false) bits.push("tools");
  if (v.tool_call_parser && v.tool_call_parser !== "hermes" && v.tool_call_parser !== "auto") bits.push(v.tool_call_parser);
  if (v.kv_cache_dtype) bits.push(`kv ${v.kv_cache_dtype}`);
  if (v.gpu_memory_utilization) bits.push(`gmu ${v.gpu_memory_utilization}`);
  const chips = (m.device_order || "").split(",").map(s => s.trim()).filter(Boolean);
  if (chips.length) bits.push(chips.join(" "));
  else if ((m.servers || []).length) bits.push(`servers: ${m.servers.join(",")}`);
  return bits.join(" · ");
}

on("#set-engine", "change", () => {
  const card = $("#set-vllm");
  if (card) card.hidden = ($("#set-engine")?.value) !== "vllm";
  // B8: engine switch must re-gate the machine chips (Windows greys out
  // in vllm mode) — re-render the group wizard list if it's on screen
  if (document.querySelector("#g-machines-chip-list")) renderGroupMachines(readGroupMachines());
});
on("#set-vllm-delivery", "change", syncVllmDeliveryRows);
on("#set-vllm-gpu-mem", "input", () => {
  const out = $("#set-vllm-gpu-mem-v");
  if (out) out.textContent = Number($("#set-vllm-gpu-mem")?.value).toFixed(2);
});

/* ---------------- on page load ---------------- */
window.addEventListener("DOMContentLoaded", async () => {
  // load persisted accordion state in the background (don't block first paint)
  loadArchPersisted();
  const dashBtn = document.querySelector('nav button[data-tab="dashboard"]');
  if (dashBtn) dashBtn.click();
  getAppSettings().then(s => { applyEngineHints(s); refreshVllmInstallButton(); }).catch(() => {});
  // mode-save round-trip: an Access-tab save reloaded us — re-open the
  // tab onto fresh data so the operator lands where they were working
  try {
    if (sessionStorage.getItem("acc-mode-saved") === "1") {
      sessionStorage.removeItem("acc-mode-saved");
      switchToTab("access");
    }
  } catch (_) {}
  // key-by-URL adoption: /?key=sk-… stores the key once and cleans the
  // address bar — lets the operator hand themselves a link when the tab
  // is locked out (no tab UI needed to adopt)
  try {
    const qs = new URLSearchParams(window.location.search);
    const k = (qs.get("key") || "").trim();
    if (k) {
      setAccessKey(k);
      qs.delete("key");
      const clean = qs.toString();
      history.replaceState(null, "", window.location.pathname +
        (clean ? "?" + clean : ""));
      setTimeout(() => window.location.reload(), 150);
    }
  } catch (_) {}
  // web-login identity: cookie resolves to an access key — no manual
  // key management needed for humans. No identity + logins exist →
  // login modal (session cookie keeps it across reloads/restarts).
  try {
    const me = await api("GET", "/api/auth/me").catch(() => null);
    // login-first UX: a login modal appears unless the identity comes
    // from a cookie SESSION (user) or the loopback console. A bare
    // localStorage key still works (api() attaches it) but doesn't skip
    // the login prompt — humans sign in; keys are for API clients.
    if (me && me.has_logins && !me.user && me.trust_via !== "loopback") {
      showLogin(true);
    }
    window.__me = me || null;
    renderLoginChip(me);
  } catch (_) {}
  // Access-tab gating at boot: whoami fires NOW (not only on tab click)
  // so a view-scoped/foreign client never rests on a restored Access tab
  // — including one resurrected by the browser's back-forward cache on
  // refresh. Runs again on every bfcache restore to keep it honest.
  accessGateCheck();
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) accessGateCheck();
  });
});/* scan */


on("#login-go", "click", doLogin);
wireKeyEditorChips();
/* ---------------- web login (vault-backed, cookie session) --------- */
function showLogin(force) {
  const m = $("#login-modal");
  if (!m) return;
  m.classList.remove("hidden");
  // MODALITY: while sign-in is up the user must not interact with
  // anything else — page inert for pointer AND keyboard
  document.body.classList.add("login-locked");
  const u = $("#login-user");
  if (u) setTimeout(() => u.focus(), 50);
  // focus trap: Tab cycles inside the dialog only
  m._trap = (e) => {
    if (e.key !== "Tab") return;
    const f = [...m.querySelectorAll(
      "button, input, select, a[href]")].filter(x => !x.hidden);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      last.focus(); e.preventDefault();
    } else if (!e.shiftKey && document.activeElement === last) {
      first.focus(); e.preventDefault();
    }
  };
  m.addEventListener("keydown", m._trap);
}
function _hideLoginModal() {
  const m = $("#login-modal");
  if (m) {
    m.classList.add("hidden");
    if (m._trap) m.removeEventListener("keydown", m._trap);
  }
  document.body.classList.remove("login-locked");
}

function renderLoginChip(me) {
  const chip = $("#login-chip");
  if (!chip) return;
  if (me && me.name) {
    // identity display only — Log out moved to the User tab (System)
    chip.innerHTML = `<i class="fa-solid fa-user-shield"></i> ${escHtml(me.name)}${me.admin ? " · admin" : ""}
      <span class="hint">log out under the User tab</span>`;
    chip.style.display = "flex";
  } else {
    chip.style.display = "none";
  }
}

async function doLogin() {
  const u = $("#login-user")?.value.trim();
  const p = $("#login-pass")?.value || "";
  const err = $("#login-err");
  const btn = $("#login-go");
  try {
    if (btn) btn.disabled = true;
    await api("POST", "/api/auth/login", { username: u, password: p });
    // verify the cookie actually landed BEFORE reloading — Firefox
    // HTTPS-only upgrades / cookie denials would otherwise loop
    const me = await api("GET", "/api/auth/me");
    if (me.logged_in || me.trust_via === "loopback") {
      $("#login-modal")?.classList.add("hidden");
      window.location.reload();
    } else {
      throw new Error(
        `sign-in did not stick (cookie not accepted — ` +
        `check HTTPS-Only mode / cookie settings for this host)`);
    }
  } catch (e) {
    if (err) { err.textContent = e.message; err.hidden = false; }
    if (btn) btn.disabled = false;
  }
}

async function doBootstrap() {
  const u = $("#login-user")?.value.trim();
  const p = $("#login-pass")?.value || "";
  const err = $("#login-err");
  try {
    await api("POST", "/api/auth/bootstrap", { username: u, password: p, key_name: u });
    $("#login-modal")?.classList.add("hidden");
    window.location.reload();
  } catch (e) {
    if (err) { err.textContent = e.message; err.hidden = false; }
  }
}

/* Boot-time Access-tab gate: same decision loadAccess makes
   (whoami.trusted), applied without loading the tab — hides the nav
   button, deactivates the section (clicking Dashboard is the current
   tab so nothing else needed), and re-arms the tab for trusted users. */
async function accessGateCheck() {
  try {
    const w = await api("POST", "/api/views/whoami", {});
    const btn = $("#tab-btn-access");
    // admins keep the Access tab even when strict mode scopes their
    // model visibility (visibility ≠ control of access itself)
    const tabOk = !!(w.trusted || w.admin);
    if (btn) btn.hidden = !tabOk;
    if (!tabOk) {
      const sec = $("#tab-access");
      if (sec) {
        sec.classList.remove("active");
        sec.hidden = true;
      }
      document.querySelectorAll('nav button[data-tab="access"]')
        .forEach(b => b.classList.remove("active"));
    }
  } catch (_) { /* whoami 404s for foreign peers — tab stays hidden */ }
}
let appSettings = null;
async function getAppSettings() {
  if (!appSettings) {
    try { appSettings = await api("GET", "/api/settings"); }
    catch (_) { appSettings = { model_dir: "store://llm-models", llama_bin: "" }; }
  }
  return appSettings;
}
// hardware-aware safe defaults — only on Configure, not Scan
async function applySafeDefaultsForScan(path){
  try{
    const grp = $("#mo-group")?.value?.trim() || "";
          const qs = new URLSearchParams({path});
          if(grp) qs.set("group", grp);
          const resp = await api("GET", `/api/safe-defaults?${qs}`);
    const f = resp.flags || {};
    if(f["ctx-size"]) $("#flag-ctx-size").value = f["ctx-size"];
    if(f["cache-type-k"]) $("#flag-cache-k").value = f["cache-type-k"];
    if(f["cache-type-v"]) $("#flag-cache-v").value = f["cache-type-v"];
    if(f["flash-attn"]) $("#flag-flash-attn").checked = f["flash-attn"]==="on";
    if(f["jinja"]!==undefined) $("#flag-jinja").checked = !!f["jinja"];
    syncCommonFlagsToRows();
    for(const k of ["threads","batch-size","ubatch-size","parallel","split-mode","n-gpu-layers","embedding","rope-scaling","rope-freq-scale","pooling"]){
      if(f[k]!==undefined){
        document.querySelectorAll("#flags-container .flag-row").forEach(row=>{
          const key=row.querySelector(".flag-key")?.value.trim().replace(/^--/,"");
          if(key===k) row.remove();
        });
        if(String(f[k]).trim()!=="") addFlagRow(k, f[k]);
      }
    }
    try{ syncTensorChips(); syncOrderChips(); syncToggleChip("chip-flash-attn","flag-flash-attn"); syncToggleChip("chip-jinja","flag-jinja"); updateAdvancedBadge(); }catch(_){}
  }catch(e){ console.warn("safe defaults failed", e); }
}

async function runScan() {
  if (runScan._running) return;  // interval overlap guard
  runScan._running = true;
  try {
  if (!$("#scan-path").value.trim()) {
    const s = await getAppSettings();
    $("#scan-path").value = s.model_dir || "store://llm-models";
  }
  const path = $("#scan-path").value.trim();
  const recursive = true;
  const box = $("#scan-results");
  // first load shows a placeholder; refreshes update silently (no flicker)
  if (!box.querySelector(".scan-model") && !box.querySelector(".arch-section"))
    box.innerHTML = loadingSpinner("Scanning for models…");
  try {
    const data = await api("GET", `/api/scan?path=${encodeURIComponent(path)}&recursive=${recursive}`);
    // configured-ness rides the scan payload now: the server derives it
    // from the FULL config set (not this viewer's view-filtered list),
    // so restricted users can't mistake another view's configured model
    // for an unconfigured download — those rows simply don't arrive.
    // Trust but verify for older daemons: fall back to /api/models.
    const configured = m => m.configured !== undefined
      ? !!m.configured
      : (() => { try { return !!configuredFallbackSet?.has(m.base_path); } catch (_) { return false; } })();
    let configuredFallbackSet = null;
    try {
      configuredFallbackSet = new Set((await api("GET", "/api/models"))
        .map(m => (m.base_path || "")));
    } catch (_) {}
    // ensure the fname -> model_id icon-reuse map exists even if the
    // Models list hasn't rendered yet
    if (!Object.keys(scanFileIconKey).length) {
      try {
        for (const mm of await api("GET", "/api/models")) {
          const f = (mm.base_path || "").split("/").pop();
          if (f && mm.model_id) scanFileIconKey[f] = mm.model_id;
        }
      } catch (_) {}
    }
    box.innerHTML = "";
    // group scan results by architecture, then sort each group alphabetically
    const groups = new Map();
    for (const m of data.models) {
      const arch = (m.architecture || "unknown").toLowerCase();
      if (!groups.has(arch)) groups.set(arch, []);
      groups.get(arch).push(m);
    }
    const sortedArchs = Array.from(groups.keys()).sort();
    for (const arch of sortedArchs) {
      const archModels = groups.get(arch).sort((a, b) =>
        (a.name || "").toLowerCase().localeCompare((b.name || "").toLowerCase()));
      const { sec, archBody } = makeArchSection(box, "scan:" + arch, arch, archModels.length);
      for (const m of archModels) {
      const isConfigured = configured(m);
      const el = document.createElement("div");
      el.className = "scan-model";
      el.dataset.scanPath = m.base_path;
      if (selectedScanPath === m.base_path) el.classList.add("row-selected");
      el.innerHTML = `<span>${scanModelIconHTML(m.base_path)}${escHtml(m.name)}${m.decision_type ? ` <span class="badge" style="background:var(--ok-dim);color:var(--ok)" title="TypeSafe decision model — serves /v1/systemone only (no chat generation)">decision</span>` : ""} <small>${m.shard_count} shard(s), ${fmtBytes(m.total_size_bytes)}${isConfigured ? " · configured" : ""}</small></span>`;
      /* click open space -> highlight + load into configure form */
      el.addEventListener("click", (ev) => {
        if (ev.target.closest("button")) return;
        selectedScanPath = (selectedScanPath === m.base_path) ? null : m.base_path;
        box.querySelectorAll(".scan-model").forEach(r =>
          r.classList.toggle("row-selected", r.dataset.scanPath === selectedScanPath));
        if (selectedScanPath) {
          modelFormLoad({}, false);
          $("#mo-id").value = m.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
          $("#mo-name").value = m.name;
          $("#mo-path").value = m.base_path;
        }
      });
      const actions = document.createElement("div");
      actions.className = "scan-model-actions";
      if (!isConfigured) {
        const btn = document.createElement("button");
        btn.className = "small";
        btn.textContent = "Configure";
        btn.onclick = () => {
          selectedScanPath = m.base_path;
          box.querySelectorAll(".scan-model").forEach(r =>
            r.classList.toggle("row-selected", r.dataset.scanPath === m.base_path));
          modelFormLoad({}, false);
          $("#mo-id").value = m.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
          $("#mo-name").value = m.name;
          $("#mo-path").value = m.base_path;
          applySafeDefaultsForScan(m.base_path);
          mowizShow(2);
        };
        actions.appendChild(btn);
      }
      const delBtn = document.createElement("button");
      delBtn.className = "small danger";
      delBtn.textContent = "Delete files";
      delBtn.onclick = async () => {
        if (!armed(delBtn, "Delete files")) return;
        delBtn.disabled = true; delBtn.textContent = "deleting…";
        try {
          const r = await api("DELETE", `/api/models/scan-file?delete_file=true&path=${encodeURIComponent(m.base_path)}`, {});
          const del = (r.deleted || []).map(p2 => p2.split("/").pop()).join(", ");
          toast(r.errors && r.errors.length
            ? `Delete failed: ${r.errors.join(" | ")}`
            : `Deleted: ${del || "none found"}`,
            r.errors && r.errors.length ? "bad" : "ok");
        } catch (err) { toast(`Delete failed: ${err.message}`, "bad"); }
        delBtn.disabled = false; delBtn.textContent = "Delete files";
        runScan();
      };
      actions.appendChild(delBtn);
      el.appendChild(actions);
      archBody.appendChild(el);
      }
      box.appendChild(sec);
    }
    if (data.projectors.length) {
      const p = document.createElement("p");
      p.innerHTML = `<small>Projectors (not models): ${data.projectors.map(x => x.name).join(", ")}</small>`;
      box.appendChild(p);
    }
    // drop a highlight if its file vanished since (deleted/downloaded over)
    if (selectedScanPath && !data.models.some(m => m.base_path === selectedScanPath))
      selectedScanPath = null;
  } catch (err) { box.innerHTML = `<p class="status-bad">${err.message}</p>`; }
  } finally { runScan._running = false; }
}
runScan._running = false;

/* self-refreshing scan list: 10s poll while Source step is visible, plus
   immediate refresh when downloads land or files are deleted */
let scanTimer = null;
let scanDirty = false;
function scanStepVisible() {
  if (document.hidden) return false;
  return $("#tab-models")?.classList.contains("active") && mowizStep === 1;
}
function startScanRefresh() {
  stopScanRefresh();
  scanTimer = setInterval(() => { if (scanStepVisible()) runScan(); }, 10000);
}
function stopScanRefresh() {
  clearInterval(scanTimer);
  scanTimer = null;
}
function noteScanDirty() {
  scanDirty = true;
  if (scanStepVisible()) { scanDirty = false; runScan(); }
}

function swapPayload() {
  return {
    binary: $("#swap-binary").value.trim() || null,
    listen_port: parseInt($("#swap-port").value, 10) || null,
    config_path: $("#swap-config-path").value.trim() || null,
    host_name: $("#swap-host-select").value || null,
  };
}

function fmtSwapSteps(r) {
  return (r.steps || []).map(s => {
    const v = s.error ? "ERR " + s.error
      : s.running !== undefined ? (s.running ? "up" : "DOWN")
      : s.valid !== undefined ? (s.valid ? "valid" : "INVALID")
      : s.written !== undefined ? "written"
      : s.backed_up !== undefined ? (s.backed_up ? "backed up" : "no existing")
      : s.models ? s.models.join(", ") : "?";
    return `${s.step}: ${v}`;
  }).join("\n");
}

// Start buttons mirror the daemon: while llama-swap is up, starting
// again spawns another instance, so both entries stay disabled until
// Stop. Engine wording ("Start All" / "Start llama-swap") is owned by
// applyEngineHints — this only toggles the disabled state.
function syncStartAll(st) {
  const btn = $("#start-all");
  if (!btn || btn.dataset.busy) return;
  btn.disabled = !!(st && st.running);
  if (st && st.running) btn.title = `llama-swap live on :${st.port} — Stop first to relaunch`;
}
async function loadSwap() {
  try {    const [settings, st, machines] = await Promise.all([
      api("GET", "/api/swap/settings"), api("GET", "/api/swap/status"),
      api("GET", "/api/machines"),
    ]);
    $("#swap-host").textContent = `${settings.host} (${settings.host_os}, ${settings.host_ip})`;
    const sel = $("#swap-host-select");
    const cur = sel.value || settings.host_name || settings.host;
    sel.innerHTML = machines.map(m => `<option value="${m.name}"${m.name === (settings.host_name || settings.host) ? " selected" : ""}>${m.name} (${m.os}${m.role === "master" ? ", master" : ""})</option>`).join("");
    if (![...sel.options].some(o => o.value === cur)) sel.value = settings.host;
    else sel.value = cur;
    if (!$("#swap-binary").value) $("#swap-binary").value = settings.binary;
    if (!$("#swap-port").value || $("#swap-port").value === "8080") $("#swap-port").value = settings.listen_port;
    if (!$("#swap-config-path").value) $("#swap-config-path").value = settings.config_path;
    const badge = $("#swap-status");
    badge.textContent = st.running ? `live on :${st.port}` : "idle";
    badge.className = `serving-ind ${st.running ? "status-ok" : "status-warn"}`;
    // While the daemon is up, Start must stay disabled: every launch
    // spawns a new daemon (port-rolled) and instances pile up. Stop
    // re-enables it. Engine label ("Launch All" / "Start llama-swap")
    // is owned by applyEngineHints — only the disabled state here.
    const launchBtn = $("#swap-launch");
    if (launchBtn && !launchBtn.dataset.busy) {
      launchBtn.disabled = !!st.running;
      launchBtn.textContent = st.running ? `Running on :${st.port}` : "Launch All";
    }
    syncStartAll(st);
  } catch (err) {
    $("#swap-status").textContent = err.message;
    $("#swap-status").className = "serving-ind status-bad";
  }
  $("#swap-preview").onclick = async () => {
    const out = $("#swap-output");
    out.hidden = false; out.textContent = "generating…";
    try {
      const cp = $("#swap-config-path").value.trim();
      const hn = $("#swap-host-select").value;
      const qs = new URLSearchParams();
      if (cp) qs.set("config_path", cp);
      if (hn) qs.set("host_name", hn);
      const r = await api("GET", "/api/swap/preview" + (qs.toString() ? `?${qs}` : ""));
      out.textContent = (r.scripts.length ? `# scripts: ${r.scripts.join(", ")}\n\n` : "") + r.yaml;
    } catch (err) { out.textContent = "ERR " + err.message; }
  };
  $("#swap-launch").onclick = async (e) => {
    const btn = e.target;
    const out = $("#swap-output");
    btn.disabled = true; btn.dataset.busy = "1"; btn.textContent = "launching…";
    try {
      const r = await api("POST", "/api/swap/launch", swapPayload());
      out.hidden = false;
      const rolledNote = r.port_rolled ? ` (port ${swapPayload().listen_port} busy — moved to ${r.port})` : "";
      out.textContent = `${r.failed ? "LAUNCH FAILED" : `llama-swap live on :${r.port} (${r.host})${rolledNote}`}\nmodels: ${(r.models || []).join(", ")}\n` + fmtSwapSteps(r);
      try { await api("POST", "/api/swap/settings", swapPayload()); } catch (_) {}
      delete btn.dataset.busy;
      loadSwap(); loadLaunch();
      if (r.failed) { btn.disabled = false; btn.textContent = "Launch All"; }
      // on success loadSwap() leaves the button disabled ("Running on :port")
    } catch (err) { toast(err.message, "bad"); delete btn.dataset.busy; btn.disabled = false; btn.textContent = "Launch All"; }
  };
  $("#swap-stop").onclick = async (e) => {
    const btn = e.target;
    btn.disabled = true; btn.textContent = "stopping…";
    try { await api("POST", "/api/swap/stop", {}); loadSwap(); }
    catch (err) { toast(err.message, "bad"); }
    btn.disabled = false; btn.textContent = "Stop";
  };
  /* UX-3: the swap-install button is retired (bootstrap.py owns installs);
     the hidden button has no handler — nothing to do here anymore. */
}

async function loadLaunch() {
  loadSwap();
  const [models, machines, active] = await Promise.all([
    api("GET", "/api/models"), api("GET", "/api/machines"), api("GET", "/api/launch/active"),
  ]);
  // vLLM mode: per-model launch cards (configured options + Load/Evict)
  // live in #vllm-launch-view — the llama.cpp RPC grid below stays hidden
  // via the rpc_launch panel gate. Same card idiom, vLLM vocabulary.
  if ((appSettings && appSettings.engine) === "vllm") {
    loadVllmLaunchCards(models, machines).catch(err =>
      console.error("[ui] vllm launch cards failed:", err));
  }
  // live reachability gate (non-blocking: cards render immediately as ready,
  // then downgrade when stats arrive ~seconds later). A server counts as down
  // ONLY on positive signal (stats succeeded AND status !== "online", or name
  // unknown). If the stats call itself fails we fail open — a stats blip must
  // never brick launches.
  const statsByName = {};
  let statsLoaded = false;
  const knownNames = new Set(machines.map(m => m.name));
  const isDown = (name) => statsLoaded && (!knownNames.has(name) || statsByName[name] !== "online");
  // per-card readiness: checked servers all up? dims card + disables Launch.
  // NOTE: server chips are display-only (locked to the model config —
  // change servers in the Models tab). Checked state always == config.
  function refreshLaunchCard(card) {
    const checked = [...card.querySelectorAll("input[type=checkbox]:checked")].map(cb => cb.value);
    const down = checked.filter(isDown);
    const btn = card.querySelector('[data-act="launch"]');
    const hint = card.querySelector(".launch-ready-hint");
    const notReady = down.length > 0;
    if (btn) {
      btn.disabled = notReady;
      btn.title = notReady ? `unavailable — down: ${down.join(", ")}` : "Launch";
    }
    card.classList.toggle("not-ready", notReady);
    if (hint) {
      hint.innerHTML = notReady
        ? `<i class="fa-solid fa-triangle-exclamation"></i> needs ${down.map(d => escHtml(d)).join(", ")} — offline (change servers in the Models tab)`
        : "";
    }
  }
  const servingSet = new Set(active.filter(a => a.serving).map(a => a.name));
  const grid = $("#launch-grid");
  // flatten primary + variants into deployments
  const deployments = [];
  for (const m of models) {
    // Launch-page disable: a swap-disabled model's card collapses to a
    // compact "hidden" chip (still visible so the user can find the
    // Enable button); its cards never launch or push to the swap.
    if (m.swap_disabled) continue;
    deployments.push({ model: m, variant: null, label: m.model_id, servers: m.servers||[], title: m.name||m.model_id,
      sub: ((appSettings && appSettings.engine) === "vllm") ? modelSubVllm(m)
        : ((m.flags||{}).ctx_size ? `-c ${(m.flags||{}).ctx_size} · ${(m.tensor_split_mode === "manual" ? "fit" : m.tensor_split_mode||"auto")} split` : `${(m.tensor_split_mode === "manual" ? "fit" : m.tensor_split_mode||"auto")} split`) });
    for (const v of (m.variants||[])) {
      const lab = (v.label||"").trim(); if (!lab) continue;
      deployments.push({ model: m, variant: lab, label: `${m.model_id}--${lab}`, servers: v.servers||[], title: `${m.name||m.model_id} <span class="badge" style="background:var(--accent-dim);color:var(--accent)">${escHtml(lab)}</span>`, sub: `variant ${escHtml(lab)} · ${(v.servers||[]).join(",")||"no servers"}` });
    }
  }
  grid.innerHTML = deployments.map(dep => {
    const m = dep.model;
    const isVar = !!dep.variant;
    const id = `launch-${dep.label}`;
    const cardServers = dep.servers;
    return `
    <div class="card launch-card" data-model="${escHtml(m.model_id)}" data-variant="${escHtml(dep.variant||"")}">
      <h3>${dep.title}${isVar ? ' <small style="color:var(--muted)">(variant)</small>' : ''}</h3>
      <div class="chip-sub">${escHtml(dep.sub)}</div>
      <div class="checkbox-list" id="${escHtml(id)}">
        ${machines.map(mac => {
          const down = isDown(mac.name);
          const inConfig = cardServers.includes(mac.name);
          if (!inConfig) return "";
          return `
          <label class="server-chip locked${down ? " machine-down" : ""}" title="${down ? "machine is down" : "as configured — change servers in the Models tab"}">
            <input type="checkbox" value="${escHtml(mac.name)}" checked disabled>
            <span class="chip-name">${escHtml(mac.name)}${mac.role === "master" ? ' <span class="badge badge-master">master</span>' : ""}</span>
            <span class="chip-sub">${(mac.detected && mac.detected.gpu_names || []).length || "?"} GPU, ${(mac.detected && mac.detected.cores) || "?"} cores</span>
            ${down ? '<span class="badge badge-down">down</span>' : ""}
          </label>`;}).join("") || '<span class="hint">no servers configured — set them in the Models tab</span>'}
      </div>
      <div class="launch-ready-hint"></div>
      <label class="auto-refresh">Port <input type="number" value="8080" class="port-input" data-model="${escHtml(m.model_id)}"></label>
      <label class="auto-refresh">Run on <select class="run-on-select" data-model="${escHtml(m.model_id)}">
        ${cardServers.map(s => `<option value="${escHtml(s)}">${escHtml(s)}${(machines.find(x => x.name===s)||{}).role === "master" ? " (master)" : ""}</option>`).join("")}
      </select></label>
      <label class="auto-refresh">Mode <select class="launch-mode-select" data-model="${escHtml(m.model_id)}">
        <option value="auto" selected>Auto (device heuristic)</option>
        <option value="rpc">RPC cluster</option>
        <option value="local">Local only (this machine, no RPC)</option>
      </select></label>
      <div class="actions">
        <button class="small" data-act="launch">Launch</button>
        <button class="small danger" data-act="stop">Stop</button>
        <button class="small" data-act="swap-disable" title="Hide this model from the swap config until re-enabled (takes effect on next push / Launch All)">Disable</button>
        <span class="serving-ind ${servingSet.size ? "status-ok" : "status-warn"}">${servingSet.size ? `live (${servingSet.size} node)` : "idle"}</span>
      </div>
    </div>`;
  }).join("");

  // swap-disabled models: compact strip (cards collapse to a single row
  // with just the Enable action; groups/models stay discoverable, keep
  // swap config clean)
  const disabledModels = models.filter(m => m.swap_disabled);
  let disabledHtml = "";
  if (disabledModels.length) {
    disabledHtml = `
    <div class="card disabled-strip">
      <h3><i class="fa-solid fa-eye-slash"></i> Disabled for swap (${disabledModels.length})</h3>
      <div class="chip-list">
        ${disabledModels.map(m => `
        <span class="server-chip" data-model="${escHtml(m.model_id)}">
          <span class="chip-name">${escHtml(m.name || m.model_id)}</span>
          <button class="small" data-act="swap-enable" data-model="${escHtml(m.model_id)}">Enable</button>
        </span>`).join("")}
      </div>
    </div>`;
  }
  const disabledHost = $("#disabled-models-strip");
  if (disabledHost) disabledHost.innerHTML = disabledHtml;
  disabledHost?.querySelectorAll('[data-act="swap-enable"]').forEach(btn => {
    btn.onclick = async () => {
      btn.disabled = true;
      try {
        await api("POST", `/api/models/${encodeURIComponent(btn.dataset.model)}/swap-disable`, { disabled: false });
        toast(`Re-enabled ${btn.dataset.model} — push / Launch All to serve it`, "ok");
      } catch (err) { toast(err.message, "bad"); }
      btn.disabled = false;
      loadLaunch();
    };
  });

  grid.querySelectorAll(".launch-card").forEach(refreshLaunchCard);
  // stats arrive late — re-evaluate every card (downgrades only, never blocks render)
  api("GET", "/api/stats").then(stats => {
    if (stats && stats.data) for (const m of stats.data) statsByName[m.name] = m.status;
    statsLoaded = true;
    grid.querySelectorAll(".launch-card").forEach(refreshLaunchCard);
  }).catch(() => { /* fail open: leave cards as rendered */ });

  grid.querySelectorAll('[data-act="launch"]').forEach(btn => {
    btn.onclick = async () => {
      const card = btn.closest(".launch-card");
      const mid = card.dataset.model;
      const variant = card.dataset.variant || null;
      const servers = [...card.querySelectorAll("input[type=checkbox]:checked")].map(cb => cb.value);
      const downNow = servers.filter(isDown);
      if (downNow.length) {
        toast(`Cannot launch ${mid}: ${downNow.join(", ")} is down (change servers in the Models tab)`, "bad");
        loadLaunch();
        return;
      }
      const port = parseInt(card.querySelector(".port-input").value, 10) || 8080;
      const runOn = card.querySelector(".run-on-select");
      const modeSel = card.querySelector(".launch-mode-select");
      const mode = modeSel ? modeSel.value : "auto";
      const masterName = runOn ? runOn.value : null;
      if (mode === "local" && masterName && !servers.includes(masterName)) {
        toast(`Cannot launch ${mid} local: ${masterName} is not among the configured servers`, "bad");
        btn.disabled = false; btn.textContent = "Launch";
        return;
      }
      btn.disabled = true; btn.textContent = "launching...";
      try {
        const body = { model_id: mid, servers, port };
        if (variant) body.variant = variant;
        if (masterName) body.master_name = masterName;
        if (mode !== "rpc") body.mode = mode;
        const r = await api("POST", "/api/launch", body);
        const ok = !r.failed;
        const livePort = r.port || port;
        const steps = r.steps.map(s => `${s.step}: ${s.error ? "ERR " + s.error : (s.running !== undefined ? (s.running ? "up" : "DOWN") : (s.killed !== undefined ? (s.killed ? "cleared" : "STUCK") : "?"))}`).join("\n");
        const rolledNote = r.port_rolled ? ` (port ${port} busy — moved to ${livePort})` : "";
        toast(ok ? `Launched ${mid}${variant?` (${variant})`:``} on :${livePort}${rolledNote}: ${steps}` : `Launch failed: ${steps}`, ok ? "ok" : "bad", 10000);
        const portInput = card.querySelector(".port-input");
        if (r.port_rolled && portInput) portInput.value = livePort;
        loadLaunch();
      } catch (err) {
        toast(err.message, "bad");
        // 409 (vllm mode) and other launch errors land inline on the card
        // — the hint element already exists in the card template
        const hint = card.querySelector(".launch-ready-hint");
        if (hint) hint.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> ' + escHtml(err.message);
      }
      btn.disabled = false; btn.textContent = "Launch";
    };
  });
  grid.querySelectorAll('[data-act="stop"]').forEach(btn => {
    btn.onclick = async () => {
      const card = btn.closest(".launch-card");
      const mid = card.dataset.model;
      const servers = [...card.querySelectorAll("input[type=checkbox]:checked")].map(cb => cb.value);
      const port = parseInt(card.querySelector(".port-input").value, 10) || 8080;
      const runOn = card.querySelector(".run-on-select");
      btn.disabled = true; btn.textContent = "stopping...";
      try {
        const stopBody = { model_id: mid, servers, port };
        if (runOn && runOn.value) stopBody.master_name = runOn.value;
        await api("POST", "/api/launch/stop", stopBody);
        loadLaunch();
      } catch (err) { toast(err.message, "bad"); }
      btn.disabled = false; btn.textContent = "Stop";
    };
  });
  grid.querySelectorAll('[data-act="swap-disable"]').forEach(btn => {
    btn.onclick = async () => {
      const card = btn.closest(".launch-card");
      const mid = card.dataset.model;
      btn.disabled = true;
      try {
        await api("POST", `/api/models/${encodeURIComponent(mid)}/swap-disable`, { disabled: true });
        toast(`${mid} hidden from swap — push / Launch All to apply to the daemon`, "ok");
      } catch (err) { toast(err.message, "bad"); }
      btn.disabled = false;
      loadLaunch();
    };
  });
}

/* vLLM per-model launch cards (rendered into #vllm-launch-view): the
   configured launch options for each model — same card idiom as the
   llama.cpp launch grid, vLLM vocabulary. Server chips are display-only
   (locked to the model config); Load fires a warmup request through the
   daemon (models load on first request); Evict unloads. Status comes
   from /api/swap/loaded. */
async function loadVllmLaunchCards(models, machines) {
  const view = $("#vllm-launch-view");
  if (!view) return;
  let loaded = [];
  try {
    const r = await api("GET", "/api/swap/loaded");
    loaded = r.loaded || [];
  } catch (_) { /* daemon down — all cards show idle */ }
  const loadedSet = new Set(loaded);
  const known = new Set((machines || []).map(m => m.name));
  const cards = (models || []).map(m => {
    const isLoaded = loadedSet.has(m.model_id);
    const chips = ((m.servers || []).length ? m.servers : []).map(s =>
      `<label class="server-chip${known.has(s) ? "" : " machine-down"}" title="as configured — change servers in the Models tab">
        <input type="checkbox" value="${escHtml(s)}" checked disabled>
        <span class="chip-name">${escHtml(s)}</span>
      </label>`).join("") || '<span class="hint">no servers</span>';
    const variants = (m.variants || []).filter(v => (v.label || "").trim()).map(v =>
      `<span class="badge" style="background:var(--accent-dim);color:var(--accent)">${escHtml(v.label.trim())}</span>`).join(" ");
    return `
    <div class="card launch-card" data-model="${escHtml(m.model_id)}">
      <h3>${escHtml(m.name || m.model_id)}</h3>
      <div class="chip-sub">${escHtml(modelSubVllm(m))}${variants ? " · " + variants : ""}</div>
      <div class="checkbox-list">${chips}</div>
      <div class="launch-ready-hint"></div>
      <div class="actions">
        <button class="small" data-act="vllm-load" ${isLoaded ? "disabled" : ""}>${isLoaded ? "Loaded" : "Load"}</button>
        <button class="small danger" data-act="vllm-evict" ${isLoaded ? "" : "disabled"}>Evict</button>
        <span class="serving-ind ${isLoaded ? "status-ok" : "status-warn"}">${isLoaded ? "loaded" : "idle"}</span>
      </div>
    </div>`;
  }).join("");
  // keep the explanatory prose, append the cards grid after it
  let grid = view.querySelector(".vllm-launch-grid");
  if (!grid) {
    grid = document.createElement("div");
    grid.className = "launch-grid vllm-launch-grid";
    view.appendChild(grid);
  }
  grid.innerHTML = cards || `<p class="hint">No models configured — add one on the Models tab.</p>`;
  grid.querySelectorAll('[data-act="vllm-load"]').forEach(btn => {
    btn.onclick = async () => {
      const card = btn.closest(".launch-card");
      const mid = card.dataset.model;
      btn.disabled = true; btn.textContent = "loading… (minutes on first load)";
      try {
        const r = await api("POST", `/api/swap/load/${encodeURIComponent(mid)}`, {});
        toast(`Loaded ${mid} in ${r.elapsed_s || "?"}s`, "ok");
        loadVllmLaunchCards(models, machines);
        if (typeof loadActiveModels === "function") loadActiveModels();
      } catch (err) {
        toast(err.message, "bad");
        const hint = card.querySelector(".launch-ready-hint");
        if (hint) hint.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> ' + escHtml(err.message);
        btn.disabled = false; btn.textContent = "Load";
      }
    };
  });
  grid.querySelectorAll('[data-act="vllm-evict"]').forEach(btn => {
    btn.onclick = async () => {
      const card = btn.closest(".launch-card");
      const mid = card.dataset.model;
      btn.disabled = true; btn.textContent = "evicting…";
      try {
        await api("POST", `/api/swap/evict/${encodeURIComponent(mid)}`, {});
        toast(`Evicted ${mid}`, "ok");
        loadVllmLaunchCards(models, machines);
        if (typeof loadActiveModels === "function") loadActiveModels();
      } catch (err) { toast(err.message, "bad"); btn.disabled = false; btn.textContent = "Evict"; }
    };
  });
}

/* ---------------- embeddings test ---------------- */
async function loadEmbeddingModels(){
  const sel = $("#emb-model");
  const status = $("#emb-status");
  if(!sel) return;
  try{
    const models = await api("GET", "/api/embeddings/models");
    const ids = models.map(m=>m.model_id);
    if(!ids.length){
      sel.innerHTML = '<option value="">(no embedding models — Scan for nomic-embed*.gguf and Configure)</option>';
      if(status){ status.textContent="no embed models"; status.className="serving-ind status-warn"; }
      return;
    }
    sel.innerHTML = ids.map(id=>`<option value="${escHtml(id)}">${escHtml(id)}</option>`).join("");
    if(status){ status.textContent=`${ids.length} embed model(s)`; status.className="serving-ind status-ok"; }
  }catch(e){
    if(status){ status.textContent="error"; status.className="serving-ind status-bad"; }
  }
}
on("#emb-refresh","click", loadEmbeddingModels);
on("#emb-run","click", async ()=>{
  // manual model-id override (G-6): capability detection is a heuristic —
  // a model it misses can be typed here and is used verbatim
  const override = $("#emb-model-override")?.value?.trim();
  const model = override || $("#emb-model")?.value?.trim();
  const a = $("#emb-a")?.value?.trim();
  const b = $("#emb-b")?.value?.trim();
  const out = $("#emb-output");
  if(!model){ toast("Pick an embedding model", "bad"); return; }
  if(!a){ toast("Enter Text A", "bad"); return; }
  const btn = $("#emb-run");
  btn.disabled = true; btn.textContent = "embedding…";
  if(out) out.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading ${escHtml(model)} if needed (may take 30s)…</p>`;
  try{
    const rA = await api("POST", "/api/embeddings", {model, input: a});
    const vecA = rA.data?.[0]?.embedding || rA.embedding || [];
    const dimA = vecA.length;
    let html = `<p class="status-ok">Text A: ${dimA} dims — [${vecA.slice(0,8).map(v=>v.toFixed(4)).join(", ")}${dimA>8?" …":""}]</p>`;
    if(b){
      const rB = await api("POST", "/api/embeddings", {model, input: b});
      const vecB = rB.data?.[0]?.embedding || rB.embedding || [];
      // cosine
      let dot=0, na=0, nb=0;
      const n = Math.min(vecA.length, vecB.length);
      for(let i=0;i<n;i++){ dot+=vecA[i]*vecB[i]; na+=vecA[i]*vecA[i]; nb+=vecB[i]*vecB[i]; }
      const cos = dot / (Math.sqrt(na)*Math.sqrt(nb) || 1);
      html += `<p>Text B: ${vecB.length} dims</p><p class="${Math.abs(cos)>0.7?"status-ok":Math.abs(cos)>0.4?"status-warn":"status-bad"}">Cosine similarity: <strong>${cos.toFixed(4)}</strong> (${cos>0.8?"very similar":cos>0.6?"similar":cos>0.3?"weak":"dissimilar"})</p>`;
    }
    if(out) out.innerHTML = html;
    toast("Embeddings done", "ok");
  }catch(e){
    if(out) out.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    toast(e.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Get embeddings";
});
// auto-load when tab opened
document.querySelectorAll('nav button[data-tab="embeddings"]').forEach(btn=>{
  btn.addEventListener('click', ()=> setTimeout(loadEmbeddingModels, 200));
});

/* ---------------- decision (TypeSafe /v1/systemone) test ---------------- */
async function loadDecisionModels(){
  const sel = $("#dec-model");
  const status = $("#dec-status");
  if(!sel) return;
  try{
    const models = await api("GET", "/api/decision/models");
    const ids = models.map(m=>m.model_id);
    if(!ids.length){
      sel.innerHTML = '<option value="">(no decision models — Scan for a laya/kev GGUF and Configure)</option>';
      if(status){ status.textContent="no decision models"; status.className="serving-ind status-warn"; }
      return;
    }
    sel.innerHTML = ids.map(id=>`<option value="${escHtml(id)}">${escHtml(id)}</option>`).join("");
    if(status){ status.textContent=`${ids.length} decision model(s)`; status.className="serving-ind status-ok"; }
  }catch(e){
    if(status){ status.textContent="error"; status.className="serving-ind status-bad"; }
  }
}
on("#dec-refresh","click", loadDecisionModels);
// choice options: "key | description" lines -> {key: description|null}
function parseDecChoiceOpts(text){
  const out = {};
  for(const line of (text||"").split("\n")){
    const s = line.trim();
    if(!s) continue;
    const i = s.indexOf("|");
    out[i>0 ? s.slice(0,i).trim() : s] = i>0 ? s.slice(i+1).trim() || null : null;
  }
  return out;
}
on("#dec-run","click", async ()=>{
  const model = $("#dec-model")?.value?.trim();
  const state = $("#dec-state")?.value?.trim();
  const out = $("#dec-output");
  if(!model){ toast("Pick a decision model", "bad"); return; }
  if(!state){ toast("Enter a state to evaluate", "bad"); return; }
  const questions = {};
  const cq = $("#dec-choice-q")?.value?.trim();
  const cOpts = parseDecChoiceOpts($("#dec-choice-opts")?.value);
  if(cq && Object.keys(cOpts).length >= 2) questions.route = {type:"choice", instructions:cq, criteria:cOpts};
  if(!Object.keys(questions).length){ toast("Fill in the choice question and at least 2 options", "bad"); return; }
  const btn = $("#dec-run");
  btn.disabled = true; btn.textContent = "asking…";
  if(out) out.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading ${escHtml(model)} if needed (may take 30s)…</p>`;
  try{
    // ONE test path: straight through the llama-swap frontend, which
    // (v262+) routes POST /v1/systemone to the model named in the body —
    // same single-endpoint pattern as the other test tabs' swap routes.
    // through the FARM GATEWAY (:8090): it carries the access key +
    // per-model scoping AND usage tracking — direct llama-swap calls
    // (v262 routes /v1/systemone by body model) bypass key accounting
    // entirely, so decision/embedding/audio/vision usage vanished.
    const st = await api("GET", "/api/swap/status");
    if(!st.running) throw new Error("swap not running — Launch All first");
    // GATEWAY origin (:8090 on the host serving this page): carries the
    // access key (window.__keyHeader) → per-key scoping + usage row;
    // llama-swap :8080 would bypass accounting entirely
    const r = await fetch(`http://${window.location.hostname}:8090/v1/systemone`, {
      method: "POST",
      credentials: "include",   // session cookie → gateway resolves grants
      headers: {"Content-Type": "application/json",
                ...(_accessKey ? {Authorization: `Bearer ${_accessKey}`} : {})},
      body: JSON.stringify({model, state, questions}),
    });
    if(!r.ok){
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json())?.error?.message || msg; } catch(_){}
      throw new Error(msg);
    }
    const resp = await r.json();
    let html = "";
    for(const [qid, ans] of Object.entries(resp.answers || {})){
      if(ans.type === "choice"){
        const rows = Object.entries(ans.probabilities || {})
          .sort((a,b)=>b[1]-a[1])
          .map(([k,p])=>`<tr><td>${escHtml(k)}${k===ans.choice?' <span class="badge" style="background:var(--ok-dim);color:var(--ok)">top</span>':""}</td><td>${(p*100).toFixed(1)}%</td></tr>`).join("");
        const conf = ans.confidence;
        html += `<h4>${escHtml(qid)} → <strong>${escHtml(ans.choice)}</strong> <span class="${conf>0.7?"status-ok":conf>0.4?"status-warn":"status-bad"}">(confidence ${(conf*100).toFixed(1)}%)</span></h4>
          <table class="gpu-table"><thead><tr><th>option</th><th>probability</th></tr></thead><tbody>${rows}</tbody></table>`;
      }
    }
    if(resp.usage) html += `<p class="hint">input tokens: ${resp.usage.input_tokens} · output tokens: ${resp.usage.output_tokens}</p>`;
    if(out) out.innerHTML = html || `<p class="status-warn">no answers in response</p>`;
  }catch(e){
    if(out) out.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    toast(e.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Ask decision model";
});
// auto-load when tab opened
document.querySelectorAll('nav button[data-tab="decision"]').forEach(btn=>{
  btn.addEventListener('click', ()=> setTimeout(loadDecisionModels, 200));
});

/* ---------------- vision test ---------------- */
async function loadVisionModels(){
  const sel = $("#vis-model");
  const status = $("#vis-status");
  if(!sel) return;
  try{
    const models = await api("GET", "/api/vision/models");
    const ids = models.map(m=>m.model_id);
    if(!ids.length){
      sel.innerHTML = '<option value="">(no vision models — set an mmproj flag in Configure)</option>';
      if(status){ status.textContent="no vision models"; status.className="serving-ind status-warn"; }
      return;
    }
    sel.innerHTML = ids.map(id=>`<option value="${escHtml(id)}">${escHtml(id)}</option>`).join("");
    if(status){ status.textContent=`${ids.length} vision model(s)`; status.className="serving-ind status-ok"; }
  }catch(e){
    if(status){ status.textContent="error"; status.className="serving-ind status-bad"; }
  }
}
on("#vis-refresh","click", loadVisionModels);
on("#vis-file","change", ()=>{
  const f = $("#vis-file").files?.[0];
  const prev = $("#vis-preview");
  if(!f){ if(prev){ prev.style.display="none"; prev.src=""; } return; }
  if(!/image\/(png|jpeg)/.test(f.type)){ toast("PNG or JPG only", "bad"); $("#vis-file").value=""; return; }
  if(f.size > 20*1024*1024){ toast("Image too large (max 20 MB)", "bad"); $("#vis-file").value=""; return; }
  const rd = new FileReader();
  rd.onload = ()=>{ if(prev){ prev.src = rd.result; prev.style.display="block"; } };
  rd.readAsDataURL(f);
});
on("#vis-run","click", async ()=>{
  const model = $("#vis-model")?.value?.trim();
  const prompt = $("#vis-prompt")?.value?.trim() || "Describe this image.";
  const file = $("#vis-file").files?.[0];
  const out = $("#vis-output");
  if(!model){ toast("Pick a vision model", "bad"); return; }
  if(!file){ toast("Choose a PNG/JPG image", "bad"); return; }
  const btn = $("#vis-run");
  btn.disabled = true; btn.textContent = "running…";
  if(out) out.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading ${escHtml(model)} if needed (may take 30s+)…</p>`;
  try{
    const image = await new Promise((res, rej)=>{
      const rd = new FileReader();
      rd.onload = ()=>res(rd.result); rd.onerror = ()=>rej(new Error("read failed"));
      rd.readAsDataURL(file);
    });
    const r = await api("POST", "/api/vision", {model, prompt, image});
    const msg = r.choices?.[0]?.message?.content || "(empty response)";
    const ms = r.usage ? `${r.usage.prompt_tokens} prompt / ${r.usage.completion_tokens} completion tokens` : "";
    if(out) out.innerHTML = `<p class="status-ok">${escHtml(ms)}</p><pre style="white-space:pre-wrap;font-family:var(--font)">${escHtml(msg)}</pre>`;
    toast("Vision test done", "ok");
  }catch(e){
    if(out) out.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    toast(e.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Run vision test";
});
// auto-load when tab opened
document.querySelectorAll('nav button[data-tab="vision"]').forEach(btn=>{
  btn.addEventListener('click', ()=> setTimeout(loadVisionModels, 200));
});

function audioBufferToWav(buffer, targetRate){
  // mixdown to mono + linear resample to targetRate + 16-bit PCM WAV.
  // Pure JS (no OfflineAudioContext) — Firefox renders empty buffers for
  // OfflineAudioContext on some configs, which produced silent WAVs.
  const src = buffer.getChannelData(0);
  const srcRate = buffer.sampleRate;
  // mixdown other channels if present
  let mono;
  if(buffer.numberOfChannels > 1){
    mono = new Float32Array(src.length);
    for(let c = 0; c < buffer.numberOfChannels; c++){
      const d = buffer.getChannelData(c);
      for(let i = 0; i < d.length; i++) mono[i] += d[i] / buffer.numberOfChannels;
    }
  } else {
    mono = src;
  }
  const n = Math.max(1, Math.round(mono.length * targetRate / srcRate));
  const res = new Float32Array(n);
  for(let i = 0; i < n; i++){
    const j = Math.floor(i * (mono.length - 1) / (n - 1 || 1));
    res[i] = mono[Math.min(j, mono.length - 1)];
  }
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ws = (o, s)=>{ for(let i=0;i<s.length;i++) v.setUint8(o+i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + n*2, true); ws(8, "WAVE");
  ws(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, targetRate, true); v.setUint32(28, targetRate*2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, "data"); v.setUint32(40, n*2, true);
  let o = 44;
  for(let i=0;i<n;i++, o+=2){ const x = Math.max(-1, Math.min(1, res[i])); v.setInt16(o, x < 0 ? x*0x8000 : x*0x7FFF, true); }
  return Promise.resolve(new Blob([buf], { type: "audio/wav" }));
}

/* ---------------- audio (ASR) test: browser mic -> text ---------------- */
const aud = { rec: null, chunks: [], stream: null, ac: null, an: null, raf: 0, t0: 0, blob: null };
async function loadAudioModels(){
  const sel = $("#aud-model");
  const status = $("#aud-status");
  if(!sel) return;
  try{
    const models = await api("GET", "/api/audio/models");
    const ids = models.map(m=>m.model_id);
    if(!ids.length){
      sel.innerHTML = '<option value="">(no ASR models — scan/add a qwen3-asr gguf)</option>';
      if(status){ status.textContent="no ASR models"; status.className="serving-ind status-warn"; }
      return;
    }
    sel.innerHTML = ids.map(id=>`<option value="${escHtml(id)}">${escHtml(id)}</option>`).join("");
    if(status){ status.textContent=`${ids.length} ASR model(s)`; status.className="serving-ind status-ok"; }
  }catch(e){
    if(status){ status.textContent="error"; status.className="serving-ind status-bad"; }
  }
}
on("#aud-refresh","click", loadAudioModels);
function audTimerStop(){
  cancelAnimationFrame(aud.t0 && aud.raf);
  const t = $("#aud-timer"); if(t) t.textContent = "0.0s";
  const lv = $("#aud-level"); if(lv) lv.textContent = "mic idle";
}
on("#aud-record","click", async ()=>{
  const btn = $("#aud-record");
  if(aud.rec && aud.rec.state === "recording"){ aud.rec.stop(); return; }
  if(!navigator.mediaDevices?.getUserMedia){
    const out = $("#aud-output");
    const isFF = /firefox/i.test(navigator.userAgent);
    const origin = location.origin;
    if(out) out.innerHTML = isFF ? `
      <p class="status-warn"><code>navigator.mediaDevices</code> is hidden — Firefox only exposes it on http for whitelisted origins.
      <b>about:config</b> → set <b>both</b> prefs to <code>true</code>:<br>
      <code>media.devices.insecure.enabled</code><br><code>media.getusermedia.insecure.enabled</code><br>
      then <b>fully quit Firefox (all windows)</b> and reopen <code>${escHtml(origin)}</code> — a reload is not enough.<br>
      Also check <code>privacy.resistFingerprinting</code> is <code>false</code> (it hides mediaDevices too).
      If the prefs reset or still fail, run a local forward: <code>ssh -N -L 8199:10.0.0.104:8199 zazi@10.0.0.104</code> and open <code>http://localhost:8199</code> (localhost is always secure, zero browser hacks).</p>`
      : `<p class="status-warn"><code>navigator.mediaDevices</code> is unavailable. Chrome/Edge/Brave: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> → add <code>${escHtml(origin)}</code> → enable → relaunch. Or use <code>ssh -N -L 8199:10.0.0.104:8199 zazi@10.0.0.104</code> and open <code>http://localhost:8199</code>.</p>`;
    toast("Mic API hidden by browser — see fix details below", "bad");
    const lv0 = $("#aud-level"); if(lv0) lv0.textContent = "mic API blocked";
    return;
  }
  // ask the user for permission first (browser prompt fires here).
  // Chromium on plain-LAN http blocks getUserMedia by default; the catch
  // below explains the per-browser override flag to enable it anyway.
  const lv = $("#aud-level");
  if(lv) lv.textContent = "awaiting mic permission…";
  try{
    aud.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  }catch(e){
    const name = e && e.name || "";
    if(name === "NotAllowedError" && !window.isSecureContext && location.hostname !== "localhost"){
      toast("Browser blocked mic on LAN http — see hint below the button to whitelist this page (no HTTPS needed)", "bad");
      const out = $("#aud-output");
      if(out) out.innerHTML = `<p class="status-warn">Your browser refuses the mic on plain-http LAN addresses by default. One-time fix (keeps HTTP, no HTTPS):<br>
      <b>Chrome/Edge/Brave:</b> open <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> → add <code>${escHtml(location.origin)}</code> → enable → relaunch.<br>
      <b>Firefox:</b> <code>about:config</code> → <code>media.devices.insecure.enabled</code> = <code>true</code> → reload.<br>
      Or open this dashboard via <code>http://${escHtml(location.hostname)}:8199</code> from the host itself (localhost is always allowed).</p>`;
      if(lv) lv.textContent = "mic blocked (insecure origin)";
      return;
    }
    const why = name === "NotAllowedError" ? "denied"
      : name === "NotFoundError" ? "no microphone found"
      : name === "NotReadableError" ? "mic busy (another app holds it)"
      : "unavailable";
    toast(`Microphone ${why} — allow mic access in the browser prompt (padlock icon to re-ask)`, "bad");
    if(lv) lv.textContent = "mic blocked";
    return;
  }
  // explicit user consent was granted by the prompt; remember choice for this page
  try{ localStorage.setItem("aud-mic-ok", "1"); }catch(_){}
  if(lv) lv.textContent = "mic allowed";
  aud.chunks = [];
  const mime = ["audio/webm;codecs=opus","audio/webm","audio/mp4"].find(t=>MediaRecorder.isTypeSupported(t)) || "";
  aud.rec = new MediaRecorder(aud.stream, mime ? { mimeType: mime } : undefined);
  aud.rec.ondataavailable = e => { if(e.data.size) aud.chunks.push(e.data); };
  aud.rec.onstop = ()=>{
    aud.blob = new Blob(aud.chunks, { type: aud.rec.mimeType || "audio/webm" });
    aud.chunks = [];
    if(aud.stream){ aud.stream.getTracks().forEach(t=>t.stop()); aud.stream = null; }
    if(aud.ac){ aud.ac.close().catch(()=>{}); aud.ac = null; }
    cancelAnimationFrame(aud.raf);
    const pb = $("#aud-playback");
    if(pb){ pb.src = URL.createObjectURL(aud.blob); pb.style.display = "block"; }
    const btn2 = $("#aud-record");
    if(btn2){ btn2.disabled = false; btn2.textContent = "● Start recording"; }
    const run = $("#aud-run");
    if(run && aud.blob.size) run.disabled = false;
    const t = $("#aud-timer"); if(t) t.textContent = `${(aud.blob.size/1024).toFixed(1)} KB clip`;
    const lv = $("#aud-level"); if(lv) lv.textContent = "mic idle";
  };
  // live level meter
  try{
    aud.ac = new (window.AudioContext || window.webkitAudioContext)();
    const src = aud.ac.createMediaStreamSource(aud.stream);
    aud.an = aud.ac.createAnalyser();
    aud.an.fftSize = 256;
    src.connect(aud.an);
    const buf = new Uint8Array(aud.an.frequencyBinCount);
    const tick = ()=>{
      aud.an.getByteFrequencyData(buf);
      const avg = buf.reduce((a,b)=>a+b,0) / buf.length;
      const lv = $("#aud-level");
      if(lv) lv.textContent = avg > 5 ? `level: ${"▮".repeat(Math.min(10, Math.round(avg/10)))}` : "level: (quiet)";
      aud.raf = requestAnimationFrame(tick);
    };
    tick();
  }catch(e){ /* meter optional */ }
  aud.rec.start(250);
  aud.t0 = Date.now();
  btn.textContent = "■ Stop recording";
  lv.textContent = "recording…";
  const tickT = ()=>{
    if(!aud.rec || aud.rec.state !== "recording") return;
    const t = $("#aud-timer");
    if(t) t.textContent = `${((Date.now()-aud.t0)/1000).toFixed(1)}s`;
    aud.raf = requestAnimationFrame(tickT);
  };
  tickT();
});
on("#aud-run","click", async ()=>{
  const model = $("#aud-model")?.value?.trim();
  const prompt = $("#aud-prompt")?.value?.trim() || "Transcribe this audio.";
  const out = $("#aud-output");
  if(!model){ toast("Pick an ASR model", "bad"); return; }
  if(!aud.blob || !aud.blob.size){ toast("Record some audio first", "bad"); return; }
  const btn = $("#aud-run");
  btn.disabled = true; btn.textContent = "transcribing…";
  if(out) out.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading ${escHtml(model)} if needed (may take 30s+)…</p>`;
  try{
    // llama.cpp's audio decoder (miniaudio) only accepts WAV/MP3/FLAC —
    // MediaRecorder emits WebM/Opus, which falls through to the video
    // decoder and fails. Decode the clip in the browser and re-encode
    // as 16 kHz mono WAV before sending.
    let payloadBlob = aud.blob;
    // convert anything that isn't already WAV/MP3/FLAC (what llama.cpp's
    // audio decoder accepts). Covers Firefox's audio/ogg, Chrome's webm, etc.
    if(!/^audio\/(wav|x-wav|wave|mpeg|mp3|flac)/i.test(aud.blob.type)){
      try{
        const ab = await aud.blob.arrayBuffer();
        const acx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuf = await acx.decodeAudioData(ab);
        await acx.close();
        payloadBlob = await audioBufferToWav(audioBuf, 16000);
      }catch(e){ toast("Audio decode failed: " + e.message + " (sending raw anyway)", "warn"); }
    }
    const dataUri = await new Promise((res, rej)=>{
      const rd = new FileReader();
      rd.onload = ()=>res(rd.result); rd.onerror = ()=>rej(new Error("read failed"));
      rd.readAsDataURL(payloadBlob);
    });
    const language = $("#aud-language")?.value || "English";
    const r = await api("POST", "/api/audio", { model, audio: dataUri, language });
    const lang = r.choices?.[0]?.message?.language || "";
    const msg = r.choices?.[0]?.message?.content || "(empty transcription)";
    const ms = (lang ? `lang: ${escHtml(lang)} · ` : "") + (r.usage ? `${r.usage.prompt_tokens} prompt / ${r.usage.completion_tokens} completion tokens` : "");
    if(out) out.innerHTML = `<p class="status-ok">${escHtml(ms)}</p><pre style="white-space:pre-wrap;font-family:var(--font)">${escHtml(msg)}</pre>`;
    toast("Transcription done", "ok");
  }catch(e){
    if(out) out.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    toast(e.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Transcribe";
});
// auto-load when tab opened
document.querySelectorAll('nav button[data-tab="audio"]').forEach(btn=>{
  btn.addEventListener('click', ()=> setTimeout(loadAudioModels, 200));
});

/* ---------------- chat test (native, no chainlit) ---------------- */
let chatHistory = [];
let chatAbort = null;
async function loadChat() {
  // called on tab open: refresh the model list (cheap), keep history
  await loadChatModels();
}
async function loadChatModels() {
  const pill = $("#chat-status");
  const sel = $("#chat-model");
  const count = $("#chat-count");
  if (!sel) return;
  try {
    const s = await api("GET", "/api/chat/models");
    if (pill) {
      pill.className = `serving-ind ${s.running ? "status-ok" : "status-warn"}`;
      pill.textContent = s.running ? "ready" : "swap down";
    }
    const prev = sel.value;
    const models = s.models || [];
    sel.innerHTML = models.length
      ? models.map(m => `<option value="${escHtml(m.id)}">${escHtml(m.id)} ${m.loaded ? "● loaded" : "○ will load"}</option>`).join("")
      : '<option value="">(no models — Launch All first)</option>';
    if (prev && models.some(m => m.id === prev)) sel.value = prev;
    if (count) {
      const nLoaded = models.filter(m => m.loaded).length;
      count.textContent = s.running ? `${nLoaded}/${models.length} loaded — picking an unloaded model loads it on first message.` : "";
    }
    if (!s.running) toast("llama-swap is down — no models can load. Go to the Launch tab → Launch All.", "bad");
  } catch (err) {
    if (pill) { pill.className = "serving-ind status-bad"; pill.textContent = "error"; }
  }
}
// minimal markdown: code fences, inline code, bold, paragraphs (no deps)
function chatMd(src) {
  const parts = String(src || "").split("```");
  let out = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      const nl = parts[i].indexOf("\n");
      const code = (nl >= 0 ? parts[i].slice(nl + 1) : parts[i]).replace(/\n$/, "");
      out += `<pre class="chat-code">${escHtml(code)}</pre>`;
    } else {
      out += escHtml(parts[i])
        .replace(/`([^`\n]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .split(/\n{2,}/).map(p => p.trim() ? `<p>${p.replace(/\n/g, "<br>")}</p>` : "").join("");
    }
  }
  return out || "<p></p>";
}
function chatScroll() {
  const log = $("#chat-log");
  if (log) log.scrollTop = log.scrollHeight;
}
function chatAdd(role, text) {
  const log = $("#chat-log");
  if (!log) return null;
  const div = document.createElement("div");
  div.className = `chat-msg chat-${role}`;
  div.innerHTML = role === "user" ? `<div>${escHtml(text)}</div>` : chatMd(text);
  log.appendChild(div);
  chatScroll();
  return div;
}
async function sendChat() {
  const model = $("#chat-model")?.value?.trim();
  const input = $("#chat-input");
  const text = (input?.value || "").trim();
  if (!model) { toast("Pick a model first", "bad"); return; }
  if (!text || chatAbort) return;
  chatHistory.push({ role: "user", content: text });
  chatAdd("user", text);
  input.value = "";
  const sendBtn = $("#chat-send");
  const stopBtn = $("#chat-stop");
  sendBtn.disabled = true;
  stopBtn.disabled = false;
  const t0 = performance.now();
  let ttft = null, nTok = 0, full = "";
  const holder = chatAdd("assistant", `⏳ Loading \`${model}\`… (first load streams when ready — big multi-node loads can take a while)`);
  chatAbort = new AbortController();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: chatHistory,
        system_prompt: $("#chat-system")?.value || "",
        temperature: parseFloat($("#chat-temp")?.value) || 0.7,
        max_tokens: parseInt($("#chat-max")?.value, 10) || 1024,
      }),
      signal: chatAbort.signal,
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(d.detail || res.statusText);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const ln of lines) {
        const t = ln.trim();
        if (!t.startsWith("data:")) continue;
        const payload = t.slice(5).trim();
        if (payload === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(payload); } catch (_) { continue; }
        if (ev.error) throw new Error(ev.error);
        const delta = (ev.choices && ev.choices[0] && ev.choices[0].delta && ev.choices[0].delta.content) || "";
        if (!delta) continue;
        if (ttft === null) {
          ttft = (performance.now() - t0) / 1000;
          full = "";
        }
        nTok++;
        full += delta;
        holder.innerHTML = chatMd(full);
        chatScroll();
      }
    }
    if (ttft === null) {
      holder.innerHTML = `<p class="status-bad">No output from \`${escHtml(model)}\` — check the Launch tab / swap log for load errors.</p>`;
      chatAbort = null;
      sendBtn.disabled = false;
      stopBtn.disabled = true;
      return;
    }
    chatHistory.push({ role: "assistant", content: full });
    const genS = (performance.now() - t0) / 1000;
    const tps = (genS - ttft) > 0 && nTok > 1 ? ((nTok - 1) / (genS - ttft)).toFixed(1) : "0";
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = `${model} — TTFT ${ttft.toFixed(2)}s · ${tps} tok/s · ${nTok} tokens · ${genS.toFixed(1)}s`;
    holder.appendChild(meta);
    chatScroll();
  } catch (e) {
    if (e.name === "AbortError") {
      if (full) {
        chatHistory.push({ role: "assistant", content: full + " …(stopped)" });
        holder.innerHTML = chatMd(full);
      } else {
        holder.innerHTML = `<p class="status-bad">Stopped.</p>`;
      }
    } else {
      holder.innerHTML = `<p class="status-bad">${escHtml(e.message)}</p>`;
    }
  }
  chatAbort = null;
  sendBtn.disabled = false;
  stopBtn.disabled = true;
}
on("#chat-send", "click", sendChat);
on("#chat-stop", "click", () => { if (chatAbort) chatAbort.abort(); });
on("#chat-clear", "click", () => { chatHistory = []; const log = $("#chat-log"); if (log) log.innerHTML = ""; });
on("#chat-refresh", "click", loadChatModels);
on("#chat-input", "keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") sendChat(); });

/* ---------------- settings ---------------- */
/* theme registry: each pair = [darkSuffix, lightSuffix, displayName, swatchColors] */
const THEMES = [
  { id: "nord", name: "Nord", dark: "nord-dark", light: "nord-light",
    swatch: ["#2e3440", "#88c0d0", "#a3be8c"] },
  { id: "tropical", name: "Tropical", dark: "tropical-dark", light: "tropical-light",
    swatch: ["#04262c", "#2ee6c8", "#ff8a5c"] },
  { id: "cyberpunk", name: "Cyberpunk", dark: "cyberpunk-dark", light: "cyberpunk-light",
    swatch: ["#0d0714", "#f9f002", "#00f0ff"] },
  { id: "fiery", name: "Fiery", dark: "fiery-dark", light: "fiery-light",
    swatch: ["#38100c", "#ff8a2a", "#ffd23d"] },
  { id: "emerald", name: "Emerald", dark: "emerald-dark", light: "emerald-light",
    swatch: ["#0d3527", "#2fe89a", "#ffd166"] },
  { id: "mynecraft", name: "Mynecraft", dark: "mynecraft-dark", light: "mynecraft-light",
    swatch: ["#2a1c0e", "#6abe30", "#4de3d1"] },
  { id: "steel", name: "Steel", dark: "steel-dark", light: "steel-light",
    swatch: ["#1b1d20", "#c7cfda", "#ff9a3c"] },  { id: "sakura", name: "Sakura", dark: "sakura-dark", light: "sakura-light",
    swatch: ["#2b1622", "#ff8ac2", "#ff9ecf"] },  { id: "arctic", name: "Arctic", dark: "arctic-dark", light: "arctic-light",
    swatch: ["#20344d", "#6fd3ff", "#d8ecff"] },
  { id: "lavender", name: "Lavender", dark: "lavender-dark", light: "lavender-light",
    swatch: ["#2f2a50", "#c4b5fd", "#f4b5fd"] },
  { id: "everforest", name: "Everforest", dark: "everforest-dark", light: "everforest-light",
    swatch: ["#2a2f38", "#a7c080", "#dbbc7f"] },
  { id: "wicked", name: "Wicked", dark: "wicked-dark", light: "wicked-light",
    swatch: ["#1a0f2e", "#a86bff", "#7ce38b"] },];

function currentTheme() {
  return localStorage.getItem("ui-theme-family") || "nord";
}
function currentMode() {
  const m = localStorage.getItem("ui-theme-mode");
  return m === "light" ? "light" : "dark";
}

function saveThemeToBackend(family, mode) {
  api("POST", "/api/settings", {
    model_dir: $("#set-model-dir")?.value.trim() || "store://llm-models",
    llama_bin: $("#set-llama-bin")?.value.trim() || "~/llama.cpp/build/bin",
    theme_family: family, theme_mode: mode,
  }).catch(err => console.warn("[settings] theme save failed:", err.message));
}

function applyTheme(family, mode) {
  family = family || currentTheme();
  mode = mode || currentMode();
  if (mode !== "light") mode = "dark";
  if (family.startsWith("custom:")) {
    // user-made themes are absolute palettes: the mode toggle never touches them
    const c = customThemes[family.slice(7)];
    if (c) {
      // custom themes ride on the base tokens + inline overrides
      document.documentElement.dataset.theme = "custom";
      for (const [k, v] of Object.entries(c.tokens)) document.documentElement.style.setProperty(k, v);
      localStorage.setItem("ui-theme-family", family);
      paintModeChips(family);
      renderThemeGrid();
      saveThemeToBackend(family, mode);
      return;
    }
    family = "nord";   // custom id gone (deleted) — fall back
  }
  const fam = THEMES.find(t => t.id === family) || THEMES[0];
  clearInlineTheme();
  document.documentElement.dataset.theme = mode === "light" ? fam.light : fam.dark;
  localStorage.setItem("ui-theme-family", fam.id);
  localStorage.setItem("ui-theme-mode", mode);
  paintModeChips(fam.id);
  renderThemeGrid();
  saveThemeToBackend(fam.id, mode);
}
/* sync the segmented control + reflect custom themes (mode chips disabled) */
function paintModeChips(family) {
  const seg = $("#theme-mode-seg");
  if (!seg) return;
  const isCustom = (family || currentTheme()).startsWith("custom:");
  const mode = currentMode();
  seg.querySelectorAll("button").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === mode);
    b.disabled = isCustom;   // customs are absolute — nothing to switch
  });
  seg.style.opacity = isCustom ? "0.45" : "1";
  seg.title = isCustom ? "Custom themes are absolute — the toggle doesn't apply" : "";
}

function clearInlineTheme() {
  for (const t of THEME_TOKENS) document.documentElement.style.removeProperty(t.var);
  for (const f of THEME_FX) document.documentElement.style.removeProperty(f.var);
}

/* ---- custom theme builder ---- */
// Every color token a theme owns. The builder modal edits exactly these;
// live preview writes them straight onto documentElement.
const THEME_TOKENS = [
  { var: "--bg", label: "Background" },
  { var: "--surface", label: "Surface (cards/sidebar)" },
  { var: "--raised", label: "Raised" },
  { var: "--hover", label: "Hover" },
  { var: "--inset", label: "Inset" },
  { var: "--border", label: "Border" },
  { var: "--border-soft", label: "Border soft" },
  { var: "--border-strong", label: "Border strong" },
  { var: "--text", label: "Text" },
  { var: "--text-2", label: "Text secondary" },
  { var: "--muted", label: "Muted" },
  { var: "--accent", label: "Accent" },
  { var: "--ribbon", label: "Ribbons" },
  { var: "--ok", label: "Success" },
  { var: "--warn", label: "Warning" },
  { var: "--bad", label: "Danger" },
];
let customThemes = {};   // id -> {name, tokens}
let themeDraft = null;   // {id|null, name, tokens} while the modal is open

async function loadCustomThemes() {
  try {
    const r = await api("GET", "/api/themes/custom");
    customThemes = (r && r.themes) || {};
  } catch (_) { customThemes = {}; }
  renderThemeGrid();
  // a stored custom theme can't paint until the list arrives — apply now
  if (currentTheme().startsWith("custom:")) applyTheme(currentTheme());
}

// numeric (non-color) theme settings — ribbon FX, tuned with sliders
const THEME_FX = [
  { var: "--ribbon-glow", label: "glow", min: 0, max: 2 },
  { var: "--ribbon-sat", label: "vivid", min: 0, max: 2.5 },
];
function cssToHex(v, fallback) {
  // getComputedStyle returns rgb()/rgba(), color inputs need #rrggbb
  const m = /rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/.exec(v || "");
  if (m) {
    const h = (n) => Math.max(0, Math.min(255, parseInt(n, 10))).toString(16).padStart(2, "0");
    return ("#" + h(m[1]) + h(m[2]) + h(m[3])).toLowerCase();
  }
  const h = /#([0-9a-fA-F]{6})/.exec(v || "");
  return h ? ("#" + h[1]).toLowerCase() : (fallback || "#000000");
}
function computedTokens() {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  for (const t of THEME_TOKENS) out[t.var] = cssToHex(cs.getPropertyValue(t.var), "#000000");
  for (const f of THEME_FX) out[f.var] = (cs.getPropertyValue(f.var) || "1").trim() || "1";
  return out;
}

let themeCssCache = null;   // raw style.css text, fetched once for seeding
async function seedTokensFrom(key) {
  // key: "current" | built-in theme id | "custom:<id>"
  if (key === "current") return computedTokens();
  if (key.startsWith("custom:")) {
    const c = customThemes[key.slice(7)];
    if (!c) return computedTokens();
    const out = { ...c.tokens };
    for (const f of THEME_FX) out[f.var] = out[f.var] ?? "1";   // pre-FX saves
    return out;
  }
  const fam = THEMES.find(t => t.id === key);
  if (!fam) return computedTokens();
  try {
    if (!themeCssCache) {
      const r = await fetch("/static/style.css");
      themeCssCache = await r.text();
    }
    const m = themeCssCache.match(
      new RegExp(`:root\\[data-theme="${fam.dark}"\\]\\s*\\{([^}]*)\\}`));
    if (!m) return computedTokens();
    const out = {};
    for (const t of THEME_TOKENS) {
      const vm = m[1].match(new RegExp(`${t.var}:\\s*(#[0-9a-fA-F]{6})`));
      out[t.var] = vm ? vm[1].toLowerCase() : themeDraft.tokens[t.var];
    }
    // FX sliders are global tuning, not per-built-in: keep the draft's
    for (const f of THEME_FX) out[f.var] = themeDraft.tokens[f.var] ?? "1";
    return out;
  } catch (_) {
    return computedTokens();
  }
}

function paintDraftInputs() {
  const box = $("#theme-builder-rows");
  for (const t of THEME_TOKENS) {
    const inp = box.querySelector(`input[data-tok="${t.var}"]`);
    if (inp) inp.value = themeDraft.tokens[t.var];
    const hex = box.querySelector(`[data-hex="${t.var}"]`);
    if (hex) hex.textContent = themeDraft.tokens[t.var];
  }
}

function previewDraft() {
  // paint the whole draft live (used after template seeding)
  document.documentElement.dataset.theme = "custom";
  for (const [k, v] of Object.entries(themeDraft.tokens)) document.documentElement.style.setProperty(k, v);
}

function openThemeBuilder(editId) {
  const existing = editId ? customThemes[editId] : null;
  themeDraft = {
    id: editId || null,
    name: existing ? existing.name : "My theme",
    tokens: existing ? { ...existing.tokens } : computedTokens(),
  };
  $("#theme-builder-name").value = themeDraft.name;
  const box = $("#theme-builder-rows");
  box.innerHTML = "";
  for (const t of THEME_TOKENS) {
    const row = document.createElement("label");
    row.className = "theme-token-row";
    row.innerHTML = `<input type="color" value="${themeDraft.tokens[t.var]}" data-tok="${t.var}" aria-label="${t.label}">
      <span class="theme-token-name">${t.label}</span>
      <code class="theme-token-hex" data-hex="${t.var}">${themeDraft.tokens[t.var]}</code>`;
    box.appendChild(row);
  }
  box.oninput = (e) => {
    const inp = e.target.closest("input[type=color]");
    if (!inp || !themeDraft) return;
    const v = inp.dataset.tok;
    themeDraft.tokens[v] = inp.value;
    const hex = box.querySelector(`[data-hex="${v}"]`);
    if (hex) hex.textContent = inp.value;
    // LIVE preview: paint immediately so the wheel shows the real result
    document.documentElement.dataset.theme = "custom";
    document.documentElement.style.setProperty(v, inp.value);
  };
  $("#theme-builder-name").oninput = (e) => { if (themeDraft) themeDraft.name = e.target.value; };
  // template picker: current look + every built-in + saved customs
  const seed = $("#theme-builder-seed");
  seed.innerHTML = `<option value="current">Current page colors</option>`
    + THEMES.map(t => `<option value="${t.id}">${escHtml(t.name)}</option>`).join("")
    + Object.entries(customThemes)
      .filter(([id]) => id !== (themeDraft.id || ""))
      .map(([id, c]) => `<option value="custom:${id}">${escHtml(c.name)} (custom)</option>`).join("");
  seed.value = "current";
  seed.onchange = async () => {
    if (!themeDraft) return;
    themeDraft.tokens = await seedTokensFrom(seed.value);
    paintDraftInputs();
    previewDraft();   // whole page follows the template instantly
  };
  // ribbon FX sliders: same live treatment as the color wheels
  const fxEls = {
    "--ribbon-glow": ["#theme-fx-glow", "#theme-fx-glow-v"],
    "--ribbon-sat": ["#theme-fx-sat", "#theme-fx-sat-v"],
  };
  const paintFx = () => {
    for (const f of THEME_FX) {
      const s = $(fxEls[f.var][0]), lab = $(fxEls[f.var][1]);
      const v = parseFloat(themeDraft.tokens[f.var]);
      const num = Number.isFinite(v) ? v : 1;
      if (s) s.value = num;
      if (lab) lab.textContent = Math.round(num * 100) + "%";
    }
  };
  paintFx();
  for (const f of THEME_FX) {
    const s = $(fxEls[f.var][0]);
    if (s) s.oninput = () => {
      if (!themeDraft) return;
      themeDraft.tokens[f.var] = String(s.value);
      const lab = $(fxEls[f.var][1]);
      if (lab) lab.textContent = Math.round(parseFloat(s.value) * 100) + "%";
      document.documentElement.dataset.theme = "custom";
      document.documentElement.style.setProperty(f.var, s.value);
    };
  }
  const _origSeedChange = seed.onchange;
  seed.onchange = async () => { await _origSeedChange(); paintFx(); };
  $("#theme-builder-delete").hidden = !existing;
  const wrap = $("#theme-builder-wrap");
  const title = $("#theme-builder-title");
  if (title) title.textContent = existing ? `Edit theme: ${existing.name}` : "Build a custom theme";
  themeBuilderShow(true);
}
function themeBuilderShow(show) {
  // inline collapsible panel (NOT a modal) so the live preview stays visible
  const wrap = $("#theme-builder-wrap");
  if (!wrap) return;
  wrap.open = !!show;
  if (show) wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function saveThemeBuilder() {
  if (!themeDraft) return;
  const name = ($("#theme-builder-name").value || "").trim() || "Custom";
  try {
    const saved = await api("POST", "/api/themes/custom",
      { id: themeDraft.id || "", name, tokens: themeDraft.tokens });
    customThemes[saved.id] = { name: saved.name, tokens: saved.tokens };
    themeDraft = null;
    themeBuilderShow(false);
    localStorage.setItem("ui-theme-family", "custom:" + saved.id);
    applyTheme("custom:" + saved.id);
  } catch (e) {
    toast("Could not save theme: " + (e.message || e), "bad");
  }
}

async function deleteThemeBuilder(e) {
  if (!themeDraft || !themeDraft.id) return;
  // two-click arm instead of native confirm() (browsers block/suppress it —
  // the delete button would die silently with zero feedback)
  const btn = e && e.currentTarget;
  const label = btn ? btn.textContent : "Delete";
  if (btn && !armed(btn, label)) return;
  try {
    await api("DELETE", "/api/themes/custom/" + encodeURIComponent(themeDraft.id));
  } catch (e2) {
    toast("Could not delete theme: " + (e2.message || e2), "bad");
    if (btn) { btn.disabled = false; btn.textContent = label; }
    return;
  }
  if (btn) { btn.disabled = false; btn.textContent = label; }
  const wasActive = currentTheme() === "custom:" + themeDraft.id;
  delete customThemes[themeDraft.id];
  themeDraft = null;
  themeBuilderShow(false);
  if (wasActive) { localStorage.setItem("ui-theme-family", "nord"); applyTheme("nord"); }
  else renderThemeGrid();
}

function cancelThemeBuilder() {
  // revert the live preview to whatever is actually saved/selected
  themeDraft = null;
  themeBuilderShow(false);
  clearInlineTheme();
  applyTheme(currentTheme());
}

function themeSwatchDots(tokens) {
  const dots = [tokens["--bg"], tokens["--accent"], tokens["--ribbon"]];
  return `<span class="theme-dots">${dots.map(c =>
    `<span style="background:${c}"></span>`).join("")}</span>`;
}

function renderThemeGrid() {
  const grid = $("#theme-grid");
  if (!grid) return;
  const fam = currentTheme();
  grid.innerHTML = "";
  for (const t of THEMES) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "theme-swatch" + (t.id === fam ? " active" : "");
    el.innerHTML = `<span class="theme-dots">${t.swatch.map(c =>
      `<span style="background:${c}"></span>`).join("")}</span>
      <span class="theme-name">${escHtml(t.name)}</span>`;
    el.onclick = () => {
      localStorage.setItem("ui-theme-family", t.id);
      // boot theme: localStorage for instant paint, then backend overrides
// if it differs (theme was saved from another browser)
applyTheme(t.id);
api("GET", "/api/settings").then(s => {
  if (!localStorage.getItem("ui-theme-family") && s.theme_family) {
    localStorage.setItem("ui-theme-family", s.theme_family);
    localStorage.setItem("ui-theme-mode", "dark");
    applyTheme(s.theme_family, s.theme_mode);
  }
}).catch(() => {});
      saveThemeToBackend(t.id, currentMode());
    };
    grid.appendChild(el);
  }
  // custom themes (persisted in config/custom_themes.json)
  for (const [id, c] of Object.entries(customThemes)) {
    const key = "custom:" + id;
    const el = document.createElement("div");
    el.className = "theme-swatch theme-custom" + (key === fam ? " active" : "");
    el.innerHTML = `${themeSwatchDots(c.tokens)}
      <span class="theme-name">${escHtml(c.name)}</span>
      <span class="theme-mini-btns">
        <button type="button" class="theme-mini-btn" data-act="edit" title="Edit"><i class="fa-solid fa-pen"></i></button>
        <button type="button" class="theme-mini-btn" data-act="apply" title="Apply"><i class="fa-solid fa-check"></i></button>
      </span>`;
    el.querySelector('[data-act="apply"]').onclick = (e) => {
      e.stopPropagation();
      localStorage.setItem("ui-theme-family", key);
      applyTheme(key);
    };
    // clicking anywhere on the card (not just ✓) applies the theme
    el.style.cursor = "pointer";
    el.onclick = () => {
      localStorage.setItem("ui-theme-family", key);
      applyTheme(key);
    };
    el.querySelector('[data-act="edit"]').onclick = (e) => {
      e.stopPropagation();
      openThemeBuilder(id);
    };
    grid.appendChild(el);
  }
  // "+" tile: build your own
  const add = document.createElement("button");
  add.type = "button";
  add.className = "theme-swatch theme-add";
  add.title = "Build a custom theme";
  add.innerHTML = `<span class="theme-add-plus"><i class="fa-solid fa-plus"></i></span>
    <span class="theme-name">Custom…</span>`;
  add.onclick = () => openThemeBuilder(null);
  grid.appendChild(add);
}

// boot theme: localStorage for instant paint, then backend overrides
// if it differs (theme was saved from another browser)
applyTheme(currentTheme());
loadCustomThemes();
api("GET", "/api/settings").then(s => {
  if (!localStorage.getItem("ui-theme-family") && s.theme_family) {
    localStorage.setItem("ui-theme-family", s.theme_family);
    localStorage.setItem("ui-theme-mode", "dark");
    applyTheme(s.theme_family, s.theme_mode);
  }
}).catch(() => {});
on("#theme-builder-cancel", "click", cancelThemeBuilder);
on("#theme-builder-save", "click", saveThemeBuilder);
on("#theme-builder-delete", "click", deleteThemeBuilder);
// (light-mode checkbox was replaced by the segmented #theme-mode-seg control)

async function loadSettings() {
  try {
    const s = await api("GET", "/api/settings");
    $("#set-model-dir").value = s.model_dir || "store://llm-models";
    $("#set-llama-bin").value = s.llama_bin || "~/llama.cpp/build/bin";
    const tok = $("#set-hf-token");
    if (tok) tok.value = ""; // never fill the real token — placeholder shows state
    const hint = $("#set-hf-token-hint");
    if (hint) {
      hint.textContent = s.hf_token_set
        ? "Token saved — gated models will use it. Paste a new one to replace."
        : "Not set — gated downloads will 401.";
    }
    appSettings = s;
    applyEngineHints(s);
    if (s.theme_family && !localStorage.getItem("ui-theme-family")) {
      localStorage.setItem("ui-theme-family", s.theme_family);
      localStorage.setItem("ui-theme-mode", "dark");
      applyTheme(s.theme_family, s.theme_mode);
    }
  } catch (err) { /* defaults fine */ }
}

on("#set-save", "click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true; btn.textContent = "saving…";
  const box = $("#set-status");
  try {
    const payload = {
      model_dir: $("#set-model-dir").value.trim(),
      llama_bin: $("#set-llama-bin").value.trim(),
      theme_family: currentTheme(),
      theme_mode: currentMode(),
    };
    // only send the token when the user typed one — empty keeps the saved one
    const tok = $("#set-hf-token")?.value.trim();
    if (tok) payload.hf_token = tok;
    // engine + vllm block; the backend merges the partial vllm dict over
    // the stored one, so only keys the UI owns are sent
    const engSel = $("#set-engine");
    if (engSel?.value) payload.engine = engSel.value;
    if (payload.engine === "vllm") {
      const vllmOut = {};
      const del = $("#set-vllm-delivery")?.value;
      if (del) vllmOut.delivery = del;
      const vb = $("#set-vllm-venv-bin")?.value.trim();
      if (vb) vllmOut.venv_bin = vb;
      const di = $("#set-vllm-docker-image")?.value.trim();
      if (di) vllmOut.docker_image = di;
      const g = parseFloat($("#set-vllm-gpu-mem")?.value);
      if (Number.isFinite(g) && g > 0 && g <= 1) vllmOut.default_gpu_memory_utilization = g;
      if (Object.keys(vllmOut).length) payload.vllm = vllmOut;
    }
    const saved = await api("POST", "/api/settings", payload);
    // engine flip (T4-G7): wholesale refresh, not a merge. The POST
    // response lacks ui_manifest (only GET injects it), and every
    // engine-gated surface — editor sections, launch mode, downloads
    // pane, models/scan lists, hidden tabs — renders from the ACTIVE
    // engine's manifest. Merging kept the OLD engine's manifest until
    // the next Settings visit, so a flip left stale UI everywhere.
    // Re-GET is the only correct source; then re-render all four
    // data-driven views immediately (models/scan/downloads/launch) so
    // no tab keeps the retired engine's rows on screen.
    const prevEngine = (appSettings || {}).engine;
    appSettings = await api("GET", "/api/settings");
    applyEngineHints(appSettings);
    refreshVllmInstallButton();
    if ((appSettings.engine || "llama_cpp") !== (prevEngine || "llama_cpp")) {
      // wholesale re-render: models table + scan list are engine-filtered
      // server-side (DEC-07/DEC-11), downloads pane is manifest-driven
      // (DEC-11), launch renders engine-native views (DEC-12)
      loadModels();
      scanDirty = true;
      if (document.querySelector("#tab-models.active")) {
        runScan();
        scanDirty = false;
      }
      loadDownloads();
      loadLaunch();
    }
    // B8: saved engine change re-gates the machine chips (Windows greys
    // out in vllm mode) — refresh the group wizard list if it's on screen
    if (document.querySelector("#g-machines-chip-list")) renderGroupMachines(readGroupMachines());
    const hint = $("#set-hf-token-hint");
    if (hint) {
      hint.textContent = saved.hf_token_set
        ? "Token saved — gated models will use it. Paste a new one to replace."
        : "Not set — gated downloads will 401.";
    }
    const tokInput = $("#set-hf-token");
    if (tokInput) tokInput.value = "";
    box.innerHTML = `<p class="status-ok">Defaults saved${tok ? " (HF token updated)." : "."}</p>`;
    toast("Defaults saved", "ok");
  } catch (err) {
    box.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
    toast(err.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Save defaults";
});

on("#set-hf-token-toggle", "click", () => {
  const inp = $("#set-hf-token");
  if (!inp) return;
  inp.type = inp.type === "password" ? "text" : "password";
});

on("#set-hf-token-clear", "click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const saved = await api("POST", "/api/settings", {
      model_dir: $("#set-model-dir").value.trim(),
      llama_bin: $("#set-llama-bin").value.trim(),
      theme_family: currentTheme(),
      theme_mode: currentMode(),
      clear_hf_token: true,
    });
    const inp = $("#set-hf-token");
    if (inp) inp.value = "";
    const hint = $("#set-hf-token-hint");
    if (hint) hint.textContent = "Not set — gated downloads will 401.";
    toast(saved.hf_token_set ? "HF token cleared" : "No token was saved", saved.hf_token_set ? "warn" : "ok");
  } catch (err) { toast(err.message, "bad"); }
  btn.disabled = false;
});

/* UX-3 (design §8): the install/uninstall buttons and their handlers are
   RETIRED — the app never executes installs anymore. src/bootstrap.py owns
   installs/uninstalls (copy-paste one-liner from the README/install docs).
   The single "Tool status" panel below is the read-only replacement:
   per-tool found/version/path from GET /api/tools/verify. The settings
   install fields (llama_bin, vllm.venv_bin, swap binary) render read-only
   from the same probed records (readonly inputs in index.html). */

async function loadToolStatus() {
  const panel = document.querySelector("#tool-status-panel");
  if (!panel) return;
  panel.innerHTML = '<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> probing…</p>';
  try {
    const v = await api("GET", "/api/tools/verify");
    const order = [["llama.cpp", "llama.cpp"], ["llama-swap", "llama-swap"], ["vllm", "vLLM"]];
    panel.innerHTML = order.map(([key, label]) => {
      const e = v[key] || {};
      const ok = e.found;
      const icon = ok ? "fa-circle-check status-ok" : "fa-circle-question status-warn";
      const loc = e.path || e.configured || "not found";
      const stale = ok && e.configured && e.configured !== e.path ? " (configured path differs)" : "";
      const ver = e.version ? " · " + escHtml(e.version) : (ok ? " · version unknown" : "");
      return '<div style="display:flex;justify-content:space-between;gap:8px;padding:4px 0">'
        + '<span><i class="fa-solid ' + icon + '"></i> <strong>' + escHtml(label) + '</strong>'
        + '<small style="color:var(--muted)">' + ver + '</small></span>'
        + '<small style="color:var(--muted);text-align:right">' + escHtml(loc) + escHtml(stale) + '</small>'
        + '</div>';
    }).join('');
  } catch (err) {
    panel.innerHTML = '<p class="status-bad">' + escHtml(err.message) + '</p>';
  }
}

/* keep the old element ids alive as no-ops: the buttons are hidden in the
   template; any stray trigger is a no-op instead of a dead fetch. */
["#uninstall-vllm", "#install-llama-cpp", "#install-llama-cpp-all",
 "#install-vllm", "#install-llama-swap"].forEach(sel => {
  on(sel, "click", () => {});
});


/* ---------------- installer modal (UX-2) ----------------
   One modal for ALL tools (llama.cpp / llama-swap / vLLM / ray): node
   picker (machines.yaml, "this machine" first), tool set, role, dry-run.
   Submits POST /api/tools/install and consumes the NDJSON progress
   stream ({event: log|done}) — busy state + per-step log + result
   summary. Skips render as "already installed" (re-run for a provisioned
   node shows only skip lines). No native confirm/alert dialogs. */

function instModalClose() {
  $("#inst-modal").classList.add("hidden");
  $("#inst-modal").setAttribute("aria-hidden", "true");
}

async function openInstModal() {
  const sel = $("#inst-machine");
  try {
    const machines = await api("GET", "/api/machines");
    const opts = [`<option value="__local__">this machine (UI host)</option>`]
      .concat((machines || []).map(m =>
        `<option value="${escHtml(m.name)}">${escHtml(m.name)}${m.role === "master" ? " (master)" : ""} · ${escHtml(m.os || "?")}</option>`));
    sel.innerHTML = opts.join("");
  } catch (_) {
    sel.innerHTML = `<option value="__local__">this machine (UI host)</option>`;
  }
  // reset to form view
  $("#inst-form").hidden = false;
  $("#inst-progress").hidden = true;
  $("#inst-result").innerHTML = "";
  $("#inst-log").textContent = "";
  $("#inst-progress-head").innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Planning…`;
  const submit = $("#inst-submit");
  submit.disabled = false; submit.textContent = "Run install";
  $("#inst-modal").classList.remove("hidden");
  $("#inst-modal").setAttribute("aria-hidden", "false");
}

function fmtInstSummary(summary) {
  const done = (summary.installed || []).length
    + (summary.repaired || []).length;
  const skipped = (summary.skipped || []).length;
  const failed = (summary.failures || []).length;
  const cls = failed ? "status-bad" : done ? "status-ok" : "status-ok";
  const icon = failed ? "fa-triangle-exclamation" : "fa-circle-check";
  const bits = [];
  if (done) bits.push(`${done} installed/repaired`);
  if (skipped) bits.push(`${skipped} skipped (already present)`);
  if (failed) bits.push(`${failed} failed`);
  if (!bits.length) bits.push("nothing to do");
  return `<p class="${cls}"><i class="fa-solid ${icon}"></i> ${escHtml(bits.join(" · "))}</p>`;
}

on("#install-open-modal", "click", openInstModal);
on("#inst-modal-close", "click", instModalClose);
on("#inst-modal-done", "click", instModalClose);
on("#inst-modal-backdrop", "click", instModalClose);
on("#inst-submit", "click", async (e) => {
  const btn = e.currentTarget;
  const machine = $("#inst-machine").value;
  const toolSel = $("#inst-tools").value;
  const role = $("#inst-role").value;
  const dryRun = $("#inst-dry-run").checked;
  const tools = toolSel === "all" ? [] : [toolSel];
  btn.disabled = true; btn.textContent = "installing…";
  $("#inst-form").hidden = true;
  $("#inst-progress").hidden = false;
  $("#inst-result").innerHTML = "";
  const logEl = $("#inst-log");
  try {
    const resp = await fetch("/api/tools/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ machine, tools, role, dry_run: dryRun }),
    });
    if (!resp.ok && !resp.body) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", summary = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line); } catch (_) { continue; }
        if (ev.event === "log") {
          logEl.textContent += ev.message + "\n";
          logEl.scrollTop = logEl.scrollHeight;
        } else if (ev.event === "done") {
          summary = ev.summary || {};
        } else if (ev.event === "timeout") {
          throw new Error("installer timed out (no activity for 5 min)");
        }
      }
    }
    const failures = summary ? (summary.failures || []) : [{ error: "no result from server" }];
    const skipped = summary ? (summary.skipped || []) : [];
    const installed = summary
      ? (summary.installed || []).concat(summary.repaired || [])
      : [];
    $("#inst-progress-head").innerHTML = failures.length
      ? `<i class="fa-solid fa-triangle-exclamation status-bad"></i> finished with failures`
      : `<i class="fa-solid fa-circle-check status-ok"></i> finished`;
    let html = fmtInstSummary(summary || {});
    if (skipped.length) {
      html += `<small class="hint" style="display:block">skipped: ${escHtml(skipped.join(", "))} — already installed, nothing re-run</small>`;
    }
    if (failures.length) {
      const rows = failures.map(f =>
        `<div style="padding:6px 8px;border-bottom:1px solid var(--border-soft);font-size:12px"><i class="fa-solid fa-triangle-exclamation status-bad"></i> ${escHtml(f.error || JSON.stringify(f))}${(f.context || {}).tool ? ` <small>(${escHtml(f.context.tool)})</small>` : ""}</div>`).join("");
      html += `<div class="card" style="padding:0;overflow:hidden;margin-top:6px">${rows}</div>`;
    }
    $("#inst-result").innerHTML = html;
    toast(failures.length ? "install finished with failures"
            : installed.length ? "install complete"
            : "already up to date",
          failures.length ? "bad" : "ok");
  } catch (err) {
    $("#inst-progress-head").innerHTML = `<i class="fa-solid fa-triangle-exclamation status-bad"></i> failed`;
    $("#inst-result").innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
    toast(err.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Run install";
});
on("#install-llama-cpp", "click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true; btn.textContent = "installing…";
  const box = $("#set-status");
  try {
    const resp = await api("POST", "/api/install-llama-cpp", {});
    btn.disabled = false; btn.textContent = "Installed";
    if (resp.status === "success") {
      $("#set-llama-bin").value = resp.llama_bin;
      box.innerHTML = `<p class="status-ok">${resp.message}</p>`;
      toast("llama.cpp installed", "ok");
    } else {
      box.innerHTML = `<p class="status-bad">${resp.message}</p>`;
      toast(resp.message, "bad");
    }
  } catch (err) {
    box.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
    toast(err.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Install llama.cpp (this host)";
});

/* Install-on-machines modal: engine-parameterized checklist. Opened by
   the per-engine "Install on other machines…" buttons; lists every
   configured machine except the local host, auto-checks install state
   (existing installs are reported, never touched), and installs only on
   the ticked machines via the batch endpoints' `machines` subset. */
let installModalEngine = "llama_cpp";
function installModalEndpoint() {
  return installModalEngine === "vllm"
    ? "/api/install-vllm/all"
    : "/api/install-llama-cpp/all";
}
function installModalLabel() {
  return installModalEngine === "vllm" ? "vLLM" : "llama.cpp";
}
function openInstallModal(engine) {
  installModalEngine = engine === "vllm" ? "vllm" : "llama_cpp";
  const modal = $("#install-modal");
  if (!modal) return;
  $("#install-modal-title").innerHTML =
    `<i class="fa-solid fa-server"></i> Install ${escHtml(installModalLabel())} on other machines`;
  $("#install-modal-sub").textContent =
    `Tick machines to check. Existing installs are skipped — only missing ones get the ${installModalLabel()} ${installModalEngine === "vllm" ? "venv" : "prebuilt for its machine type"}.`;
  $("#install-modal-list").innerHTML =
    `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading machines…</p>`;
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  loadInstallModalMachines();
}
function closeInstallModal() {
  const modal = $("#install-modal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}
async function loadInstallModalMachines() {
  const box = $("#install-modal-list");
  try {
    const [machines, settings] = await Promise.all([
      api("GET", "/api/machines"),
      api("GET", "/api/swap/settings"),
    ]);
    const localName = settings.host_name || settings.host;
    const remotes = (machines || []).filter(m => m.name !== localName);
    if (!remotes.length) {
      box.innerHTML = `<p class="hint">No other machines configured — add one on the Machines tab first.</p>`;
      return;
    }
    box.innerHTML = `<div class="checkbox-list">` + remotes.map(m => {
      const gpus = (m.detected && m.detected.gpu_names) || [];
      return `
      <label class="server-chip" data-machine="${escHtml(m.name)}" title="tick to include in check/install">
        <input type="checkbox" class="install-modal-tick" value="${escHtml(m.name)}" checked>
        <span class="chip-name">${escHtml(m.name)}${m.role === "master" ? ' <span class="badge badge-master">master</span>' : ""}</span>
        <span class="chip-sub">${escHtml(m.ip || "")} · ${escHtml(m.os || "linux")} · ${gpus.length || "?"} GPU</span>
        <span class="badge install-modal-status">not checked</span>
      </label>`;
    }).join("") + `</div>`;
    // auto-check on open so the user sees state before ticking
    await runInstallModalCheck();
  } catch (err) {
    box.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
  }
}
function installModalRowStatus(name, kind, text) {
  // badge idiom matches the launch-grid server-chips: ok / down / warn
  const badge = kind === "ok" ? "" : kind === "bad" ? " badge-down" : " badge-local";
  const row = document.querySelector(`#install-modal-list .server-chip[data-machine="${CSS.escape(name)}"] .install-modal-status`);
  if (row) { row.className = "badge install-modal-status" + badge; row.textContent = text; }
}
function installModalTicked() {
  return Array.from(document.querySelectorAll("#install-modal-list .install-modal-tick:checked"))
    .map(el => el.value);
}
async function runInstallModalCheck() {
  const names = installModalTicked();
  names.forEach(n => installModalRowStatus(n, "warn", "checking…"));
  try {
    const resp = await api("POST", installModalEndpoint(), {install_missing: false, machines: names});
    for (const r of (resp.results || [])) {
      const ok = r.status === "already-installed" || r.status === "installed";
      const kind = ok ? "ok" : r.status === "offline" ? "warn" : "bad";
      installModalRowStatus(r.machine, kind, r.status + (r.bin ? ` — ${r.bin}` : r.reason ? ` — ${r.reason}` : ""));
      // pre-untick machines that need nothing, so Install hits only gaps
      if (ok || r.status === "offline") {
        const tick = document.querySelector(`#install-modal-list .server-chip[data-machine="${CSS.escape(r.machine)}"] .install-modal-tick`);
        if (tick) tick.checked = false;
      }
    }
    const missing = (resp.missing || []).length;
    toast(resp.summary || "Check complete", missing ? "warn" : "ok");
  } catch (err) {
    toast(err.message, "bad");
  }
}
async function runInstallModalInstall(btn) {
  const names = installModalTicked();
  if (!names.length) { toast("Tick at least one machine", "warn"); return; }
  btn.disabled = true; btn.textContent = "installing…";
  names.forEach(n => installModalRowStatus(n, "warn", "installing…"));
  try {
    const resp = await api("POST", installModalEndpoint(), {install_missing: true, machines: names});
    for (const r of (resp.results || [])) {
      const ok = r.status === "already-installed" || r.status === "installed";
      const kind = ok ? "ok" : r.status === "offline" ? "warn" : "bad";
      installModalRowStatus(r.machine, kind, r.status + (r.bin ? ` — ${r.bin}` : r.reason ? ` — ${r.reason}` : ""));
    }
    const box = $("#install-all-status");
    if (box) box.innerHTML = `<p class="status-ok">${escHtml(resp.summary || "Install complete")}</p>`;
    toast(resp.summary || "Install complete", "ok");
  } catch (err) {
    toast(err.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Install selected";
}
on("#install-llama-cpp-others", "click", () => openInstallModal("llama_cpp"));
on("#install-vllm-others", "click", () => openInstallModal("vllm"));
on("#install-modal-close", "click", closeInstallModal);
on("#install-modal-backdrop", "click", closeInstallModal);
on("#install-modal-check", "click", runInstallModalCheck);
on("#install-modal-install", "click", (e) => runInstallModalInstall(e.currentTarget));


/* Settings tab: install vLLM (this host) — engine-conditional (B10b).
   data-engine="vllm" on the button makes applyEngineHints show it ONLY
   in vllm mode (hidden entirely on llama_cpp). Capability gate: the
   machine payload's vllm_capable (B14) hides the button on incapable
   hosts (windows/macos/intel/cpu) with the verbatim reason as tooltip. */
async function refreshVllmInstallButton() {
  /* UX-3: retired — the install button is hidden in the template and the
     vllm-plan endpoint no longer exists. Kept as a no-op so the two
     legacy call sites stay valid. */
}

/* tool status rides along with the settings load (same panel, same tab) */
const _loadSettings = loadSettings;
loadSettings = async function () {
  await _loadSettings();
  loadToolStatus();
};

/* Settings tab: install vLLM (this host) — single combined button with
   busy state + run_steps result report (B10b). The endpoint answers 502
   with the full report on a failed/refused install, so raw fetch keeps
   the report visible instead of collapsing it into an error string.
   A refusal (windows, docker, no GPU tooling) is an install failure,
   not a server error. */
on("#install-vllm", "click", async (e) => {
  const btn = e.currentTarget;
  const box = $("#install-vllm-status");
  const statusBox = $("#set-status");
  btn.disabled = true; btn.textContent = "installing vLLM…";
  if (box) box.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> checking for an existing install, then installing — heavy download, may take minutes…</p>`;
  try {
    const machines = await api("GET", "/api/machines");
    const local = machines.find(m => m.role === "master") || machines[0];
    if (!local) throw new Error("no machines configured");
    const res = await fetch(`/api/machines/${encodeURIComponent(local.name)}/install-vllm`, { method: "POST", headers: { "Content-Type": "application/json" } });
    const report = await res.json().catch(() => ({}));
    const already = !!report.skipped
      || (report.steps || []).some(s => s.skipped)
      || (report.detail || "").startsWith("already");
    const cls = already ? "status-ok" : report.ok ? "status-ok" : "status-bad";
    if (box) {
      const head = already
        ? `<p class="status-ok"><i class="fa-solid fa-circle-check"></i> Already installed — ${escHtml((report.detail || "existing install verified").replace(/^already installed:?\s*/i, ""))}</p>`
        : `<p class="${cls}"><i class="fa-solid ${report.ok ? "fa-circle-check" : "fa-triangle-exclamation"}"></i> ${escHtml(report.detail || (report.ok ? "install complete" : "install failed"))}</p>`;
      const steps = (report.steps || []).length
        ? `<div class="card" style="padding:8px 10px;overflow:hidden;margin-top:4px;white-space:pre-wrap;font-size:12px">${escHtml(fmtVllmSteps(report))}</div>`
        : "";
      box.innerHTML = head + steps
        + (report.target ? `<small class="hint" style="display:block;margin-top:4px">vllm bin: ${escHtml(report.target)}</small>` : "");
    }
    if (statusBox) statusBox.innerHTML = `<p class="${cls}">${escHtml(report.detail || (already ? "already installed" : report.ok ? "install complete" : "install failed"))}</p>`;
    toast(already ? "vLLM already installed"
            : report.ok ? "vLLM installed"
            : `vLLM install failed: ${report.detail || "see report"}`,
          already || report.ok ? "ok" : "bad");
  } catch (err) {
    if (box) box.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`;
    toast(err.message, "bad");
  }
  btn.disabled = false; btn.textContent = "Install vLLM (this host)";
});

/* ---------------- file explorer modal (Settings + per-machine) ---------------- */
let fsTargetInput = null;
let fsTargetMachine = null;
let fsCurrentPath = "";
let fsCurrentAlias = "";

async function fsLoad(path) {
  const listEl = $("#fs-list");
  const statusEl = $("#fs-status");
  const pathEl = $("#fs-path");
  fsCurrentPath = path;
  if (pathEl) pathEl.value = path;
  if (listEl) listEl.innerHTML = `<p class="hint" style="padding:14px"><i class="fa-solid fa-spinner fa-spin"></i> Loading…</p>`;
  if (statusEl) statusEl.textContent = "";
  try {
    const qs = new URLSearchParams();
    qs.set("path", path);
    if (fsTargetMachine) qs.set("machine", fsTargetMachine);
    if ($("#fs-show-hidden")?.checked) qs.set("show_hidden", "true");
    const data = await api("GET", `/api/fs/browse?${qs}`);
    fsCurrentPath = data.path;
    fsCurrentAlias = data.alias || data.path;
    if (pathEl) pathEl.value = data.path;
    const parent = data.parent;
    const entries = data.entries || [];
    if (!entries.length) {
      listEl.innerHTML = `<p class="hint" style="padding:14px">Empty folder</p>`;
    } else {
      listEl.innerHTML = entries.map(e => `
        <div class="fs-row ${e.is_dir ? "is-dir" : ""}" data-path="${escHtml(e.path)}" data-dir="${e.is_dir}">
          <span class="fs-icon"><i class="fa-solid ${e.is_dir ? "fa-folder" : "fa-file"}"></i></span>
          <span class="fs-name">${escHtml(e.name)}</span>
          ${e.is_dir ? `<span class="fs-arrow"><i class="fa-solid fa-chevron-right"></i></span>` : ""}
        </div>`).join("");
      listEl.querySelectorAll(".fs-row").forEach(row => {
        row.onclick = () => {
          const p = row.dataset.path;
          const isDir = row.dataset.dir === "true";
          if (isDir) fsLoad(p);
        };
      });
    }
    if (statusEl) statusEl.textContent = `${entries.length} items · ${data.os || ""} · ${data.path}`;
    // store parent for up button
    listEl.dataset.parent = parent || "";
  } catch (err) {
    if (listEl) listEl.innerHTML = `<p class="status-bad" style="padding:14px">${escHtml(err.message)}</p>`;
    if (statusEl) statusEl.textContent = err.message;
  }
}

function openFsBrowse(targetInputId, machineName) {
  fsTargetInput = targetInputId;
  fsTargetMachine = machineName || null;
  const modal = $("#fs-modal");
  if (!modal) return;
  const init = $(`#${targetInputId}`)?.value?.trim() || "";
  const hint = $("#fs-machine-hint");
  if (hint) hint.textContent = fsTargetMachine ? `Browsing on ${fsTargetMachine} via SSH — pick a folder on that machine` : `Browsing on this host (app server) — works on Linux & Windows clients`;
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  fsLoad(init || "~");
}

function closeFsBrowse() {
  const modal = $("#fs-modal");
  if (modal) { modal.classList.add("hidden"); modal.setAttribute("aria-hidden", "true"); }
  fsTargetInput = null;
  fsTargetMachine = null;
}

on("#fs-modal-backdrop", "click", closeFsBrowse);
on("#fs-modal-close", "click", closeFsBrowse);
on("#fs-cancel", "click", closeFsBrowse);
on("#fs-up", "click", () => {
  const parent = $("#fs-list")?.dataset?.parent || "";
  if (parent && parent !== fsCurrentPath) fsLoad(parent);
});
on("#fs-go", "click", () => {
  const p = $("#fs-path")?.value?.trim();
  if (p) fsLoad(p);
});
on("#fs-path", "keydown", (e) => { if (e.key === "Enter") { const p = e.target.value.trim(); if (p) fsLoad(p); }});
on("#fs-show-hidden", "change", () => fsLoad(fsCurrentPath));
on("#fs-select", "click", () => {
  if (!fsTargetInput) return closeFsBrowse();
  const inp = $(`#${fsTargetInput}`);
  if (inp) {
    let chosen = fsCurrentPath;
    // for model directory, prefer portable store:// alias when available
    if (fsTargetInput === "set-model-dir" && fsCurrentAlias && fsCurrentAlias.startsWith("store://")) {
      chosen = fsCurrentAlias;
    }
    inp.value = chosen;
    inp.dispatchEvent(new Event("change", { bubbles: true }));
  }
  closeFsBrowse();
  toast(`Picked ${fsCurrentPath}`, "ok");
});

on("#set-model-dir-browse", "click", () => openFsBrowse("set-model-dir", null));
on("#set-llama-bin-browse", "click", () => openFsBrowse("set-llama-bin", null));
on("#m-llama-bin-browse", "click", () => {
  const orig = $("#machine-original-name")?.value?.trim() || $("#m-name")?.value?.trim() || "";
  const ip = $("#m-ip")?.value?.trim();
  const user = $("#m-username")?.value?.trim();
  // if editing an existing machine, browse that machine
  let machine = null;
  if (orig) machine = orig;
  else if (ip && user) {
    // unsaved machine — cannot SSH without being saved; fall back to local with hint
    toast("Save the machine first to browse its own filesystem via SSH — showing this host for now", "info");
  }
  openFsBrowse("m-llama-bin", machine);
});

// RDMA toggle enables/disables all RDMA sub-fields
on("#m-rdma", "change", () => {
  const on = $("#m-rdma").checked;
  ["#m-rdma-ip", "#m-rdma-dev", "#m-rdma-gid"].forEach(sel => {
    $(sel).disabled = !on;
  });
  $("#m-rdma-cm").disabled = !on;
  if (on && !$("#m-rdma-ip").value && $("#m-ip").value.trim()) {
    $("#m-rdma-ip").focus();
  }
});

// Esc to close modal
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#fs-modal")?.classList.contains("hidden")) closeFsBrowse();
});

/* ---------------- downloads ---------------- */
let downloadsTimer = null;

async function loadDownloads() {
  let stores = [];
  try { stores = await api("GET", "/api/downloads/stores"); } catch (e) { /* non-fatal */ }
  const sel = $("#dl-store");
  if (sel && !sel.options.length) {
    for (const s of stores) {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s;
      sel.appendChild(opt);
    }
  }
  let jobs = [];
  const box = $("#dl-jobs");
  if (!box) return;
  // spinner while the first fetch is in flight (before any data arrived)
  const firstLoad = !box.dataset.loaded && !box._fetched;
  if (firstLoad && !box.querySelector(".dl-loading"))
    box.innerHTML = `<p class="hint dl-loading"><i class="fa-solid fa-spinner fa-spin"></i></p>`;
  try { jobs = await api("GET", "/api/downloads"); } catch (e) { /* ignore */ }
  box._fetched = true;
  box.dataset.loaded = "1";
  // tab header spinner while anything is still in flight
  const h2 = $("#tab-dljobs h2");
  if (h2) {
    const n = jobs.filter(j => j.status === "downloading").length;
    h2.innerHTML = `Downloads${n ? ` <i class="fa-solid fa-spinner fa-spin" title="${n} download(s) in flight"></i>` : ""}`;
  }
  if (!jobs.length) { box.innerHTML = `<p class="hint">No downloads yet.</p>`; }
  else {
    /* each job renders its own card + bar; concurrent jobs update
       independently on every poll. Progress bars must GLIDE, not
       jump-cut: existing cards get their fill patched in place (the
       CSS width transition then runs), only NEW/CHANGED-STATE jobs
       get a full re-render. */
    for (const j of jobs) {
      const total = j.total_bytes || 0;
      const pct = total ? Math.min(100, Math.round((j.done_bytes || 0) / total * 100)) : 0;
      const active = j.status === "downloading";
      const existing = box.querySelector(`.dl-job[data-job="${CSS.escape(j.id)}"]`);
      if (existing && existing.dataset.status === j.status) {
        // in-place patch: bar + bytes line only
        const fill = existing.querySelector(".metric-fill");
        if (fill) {
          fill.style.width = `${pct}%`;
          fill.style.background = pct >= 100 ? "var(--ok)" : "var(--warn)";
        }
        const meta = existing.querySelector("small");
        if (meta) meta.innerHTML = `${fmtBytes(j.done_bytes || 0)} / ${fmtBytes(total)} (${pct}%)${j.error ? ` — <span class="status-bad">${escHtml(j.error)}</span>` : ""}`;
        continue;
      }
      const badge = active
        ? `<i class="fa-solid fa-spinner fa-spin"></i> downloading`
        : escHtml(j.status);
      const html = `
      <div class="card dl-job" data-job="${j.id}" data-status="${j.status}">
        <div class="row">
          <div class="dl-job-info">
            <div><strong>${escHtml(j.filename)}</strong>${active ? ` <i class="fa-solid fa-circle-notch fa-spin dl-spin" title="still coming over the network"></i>` : ""}</div>
            <small>${escHtml(j.repo_id)} → ${escHtml(j.store)} · ${escHtml(j.dest)}</small>
          </div>
          <span class="serving-ind ${j.status === "done" ? "status-ok" : (active ? "status-warn" : "status-bad")}">${badge}</span>
        </div>
        <div class="metric-bar"><div class="metric-fill" style="width:${pct}%; background:${pct >= 100 ? "var(--ok)" : "var(--warn)"}"></div></div>
        <small>${fmtBytes(j.done_bytes || 0)} / ${fmtBytes(total)} (${pct}%)${j.error ? ` — <span class="status-bad">${escHtml(j.error)}</span>` : ""}</small>
        ${(j.log || []).length ? `<pre class="dl-log">${escHtml(j.log.slice(-8).join("\n"))}</pre>` : ""}
        ${active ? `<div class="actions"><button class="small danger" data-act="cancel">Cancel</button></div>` : ""}
      </div>`;
      if (existing) {
        // status changed: replace in place so ordering stays stable
        const tmp = document.createElement("div");
        tmp.innerHTML = html;
        existing.replaceWith(tmp.firstElementChild);
      } else {
        box.insertAdjacentHTML("beforeend", html);
      }
    }
    // remove cards for vanished jobs
    const liveIds = new Set(jobs.map(j => j.id));
    box.querySelectorAll(".dl-job").forEach(c => {
      if (!liveIds.has(c.dataset.job)) c.remove();
    });
    if (!box.querySelector(".dl-job")) box.innerHTML = `<p class="hint">No downloads yet.</p>`;
    box.querySelectorAll('[data-act="cancel"]').forEach(btn => {
      btn.onclick = async () => {
        const id = btn.closest(".dl-job").dataset.job;
        try { await api("POST", `/api/downloads/${id}/cancel`); loadDownloads(); }
        catch (err) { toast(err.message, "bad"); }
      };
    });
  }
  /* keep polling while ANY job is active -- each concurrent job gets
     its own card and its own bar refreshed by the same 1s poll.
     A job leaving "downloading" means store contents changed → flag the
     models scan list for refresh (it picks it up on its poll/tab-enter). */
  const prev = loadDownloads._prev || {};
  for (const j of jobs) {
    if (prev[j.id] === "downloading" && j.status !== "downloading") noteScanDirty();
  }
  loadDownloads._prev = Object.fromEntries(jobs.map(j => [j.id, j.status]));
  const anyActive = jobs.some(j => j.status === "downloading");
  clearInterval(downloadsTimer);
  if (anyActive) downloadsTimer = setInterval(loadDownloads, 1000);
}

on("#dl-start", "click", async (e) => {
  const btn = e.currentTarget;
  const link = $("#dl-link").value.trim();
  const store = $("#dl-store").value;
  if (!link) { toast("paste an hf:// link first", "bad"); return; }
  // DEC-11 (T4-G5): in the engine's vocabulary the custom pane either
  // fetches a .gguf file (llama.cpp) or a whole HF repo (vLLM). A bare
  // hf://owner/name repo link posts repo:true so the vLLM path runs
  // start_repo; the llama.cpp gate would 409 it anyway, so this never
  // changes llama.cpp behavior (its example copy points at files).
  const dlFormats = ((appSettings && appSettings.ui_manifest
                      && appSettings.ui_manifest.downloads) || {}).formats || [];
  const repoLink = dlFormats.includes("hf") && !dlFormats.includes("gguf")
    && !/\.[A-Za-z0-9]+$/.test(link.split("/").pop() || "");
  btn.disabled = true; btn.textContent = "starting…";
  const box = $("#dl-status");
  try {
    const r = await api("POST", "/api/downloads",
                        repoLink ? { link, store, repo: true } : { link, store });
    const jobs = r.jobs || [r];
    const started = jobs.filter(j => j.status !== "done");
    const existed = jobs.filter(j => j.status === "done");
    box.innerHTML = started.length
      ? `<p class="status-ok">${started.length} download${started.length > 1 ? "s" : ""} started${r.sharded ? ` (all ${r.total} shards)` : ""} → ${escHtml(started[0].dest)}${started.length > 1 ? " …" : ""}</p>`
        + (existed.length ? `<p class="status-warn">${existed.length} file${existed.length > 1 ? "s" : ""} already existed — skipped</p>` : "")
      : `<p class="status-warn">File${jobs.length > 1 ? "s" : ""} already exist${jobs.length > 1 ? "" : "s"} at ${escHtml(jobs[0].dest)}</p>`;
    $("#dl-link").value = "";
    loadDownloads();
    // a download starting is the event of interest — jump to its subtab
    if (started.length) dlSubTab("jobs");
  } catch (err) { box.innerHTML = `<p class="status-bad">${escHtml(err.message)}</p>`; }
  btn.disabled = false; btn.textContent = "Start download";
});

/* ---------------- HF GGUF browser (amazing search) ---------------- */
let hfPage = 0, hfQuery = "", hfQuant = "", hfTask = "", hfCompany = "", hfCollection = "", hfSort = "downloads", hfActiveId = null, hfDebounce = null;
let hfFormat = "gguf"; // "gguf" | "safetensors" — which HF repo flavor the browse pane lists
let hfVramTotal = 0; // for recommended highlight
let hfAvatarCache = {}; // author -> url
let hfCompaniesFull = [];
let hfCompanyFilterDebounce = null;
let hfRecommended = false;
const HF_RECOMMENDED = ["openai","google","Qwen","deepseek-ai","microsoft","meta-llama","mistralai","huihui-ai","orcarouter","unsloth","ornith-ai"];
async function getHfAvatar(author){
  if(!author) return null;
  const key = author.toLowerCase();
  if(hfAvatarCache[key] !== undefined) return hfAvatarCache[key];
  try{
    const r = await api("GET", `/api/hf/avatar/${encodeURIComponent(author)}`);
    hfAvatarCache[key] = r.avatarUrl || null;
    return r.avatarUrl || null;
  }catch(_){
    hfAvatarCache[key] = null;
    return null;
  }
}
async function loadHfCompanies(search=""){
  const sel=$("#hf-company"); if(!sel) return;
  try{
    const params = new URLSearchParams({limit:"150"});
    if(search) params.set("search", search);
    if(hfTask) params.set("pipeline_tag", hfTask);
    if(hfQuant) params.set("quant", hfQuant);
    if(hfRecommended) params.set("recommended", "true");
    const data = await api("GET", `/api/hf/companies?${params}`);
    if(!search) hfCompaniesFull = data.companies||[];
    const toShow = search ? (data.companies||[]) : hfCompaniesFull;
    const cur = sel.value;
    sel.innerHTML = '<option value="">All companies</option>';
    for(const c of toShow){
      const o=document.createElement("option");
      o.value=c.author; o.textContent=`${c.author} (${c.count})`;
      if(c.avatarUrl) o.dataset.avatar = c.avatarUrl;
      sel.appendChild(o);
    }
    if(cur && [...sel.options].some(o=>o.value===cur)) sel.value=cur;
    const hint=$("#hf-results-hint");
    if(hint && data.totalUnique) hint.textContent = `${toShow.length} companies · ${data.totalUnique} total unique`;
  }catch(e){ console.warn("companies load failed", e.message); }
}
function filterHfCompaniesLocal(q){
  const sel=$("#hf-company"); if(!sel || !hfCompaniesFull.length) return;
  const cur = sel.value;
  const needle = (q||"").toLowerCase();
  sel.innerHTML = '<option value="">All companies</option>';
  for(const c of hfCompaniesFull){
    if(needle && !c.author.toLowerCase().includes(needle)) continue;
    const o=document.createElement("option");
    o.value=c.author; o.textContent=`${c.author} (${c.count})`;
    if(c.avatarUrl) o.dataset.avatar = c.avatarUrl;
    sel.appendChild(o);
  }
  if(cur && [...sel.options].some(o=>o.value===cur)) sel.value=cur;
}
async function loadHfPipelineTags(){
  const sel=$("#hf-pipeline-select"); if(!sel) return;
  try{
    const data = await api("GET", "/api/hf/pipeline-tags?limit=200");
    const cur = sel.value;
    // keep All option, add tags
    const existing = new Set([...sel.options].map(o=>o.value));
    for(const t of (data.pipeline_tags||[])){
      if(existing.has(t)) continue;
      const o=document.createElement("option");
      o.value=t; o.textContent=t;
      sel.appendChild(o);
    }
    if(cur) sel.value=cur;
  }catch(e){ console.warn("pipeline tags load failed", e.message); }
}
async function loadHfCollections(owner){
  const sel=$("#hf-collection"); if(!sel) return;
  sel.innerHTML = '<option value="">All collections</option>';
  sel.disabled = !owner;
  if(!owner) return;
  try{
    sel.disabled = true;
    const params = new URLSearchParams({owner, limit:"100"});
    if(hfTask) params.set("pipeline_tag", hfTask);
    if(hfQuant) params.set("quant", hfQuant);
    const data = await api("GET", `/api/hf/collections?${params}`);
    for(const c of (data.collections||[])){
      const o=document.createElement("option");
      o.value=c.slug; o.textContent=`${c.title} (${c.itemCount})`;
      sel.appendChild(o);
    }
    sel.disabled = false;
  }catch(e){
    sel.disabled = false;
    console.warn("collections load failed", e.message);
  }
}
/* downloads sub-tabs: Browse GGUFs | Custom link | Downloads (browse
   inits once — filters persist across visits instead of resetting per open) */
let hfInit = false;
function dlSubTab(name) {
  for (const n of ["browse", "custom", "jobs"]) {
    $("#dl-sub-" + n)?.classList.toggle("active", n === name);
    const pane = $("#dl-pane-" + n);
    if (pane) pane.hidden = n !== name;
  }
  // opening the Downloads subtab refreshes its list
  if (name === "jobs") loadDownloads();
}
on("#dl-sub-browse", "click", () => dlSubTab("browse"));
on("#dl-sub-custom", "click", () => dlSubTab("custom"));
on("#dl-sub-jobs", "click", () => dlSubTab("jobs"));
async function initHfBrowse(){
  if (hfInit) return;
  hfInit = true;
  // populate stores
  try{
    const stores = await api("GET","/api/downloads/stores");
    const sel = $("#hf-store"), cur = sel.value;
    if(sel && !sel.options.length){
      for(const s of stores){ const o=document.createElement("option"); o.value=s; o.textContent=s; sel.appendChild(o); }
    } else if(sel && cur) sel.value = cur;
    if(sel && !sel.value && stores.length) sel.value = stores[0];
    // also keep dl-store in sync
    const dlSel = $("#dl-store");
    if(dlSel && sel.value) dlSel.value = sel.value;
  }catch(_){}
  // populate companies + pipeline tags
  loadHfCompanies();
  loadHfPipelineTags();
  // fetch vram for recommendation
  try{
    const st = await api("GET","/api/stats");
    hfVramTotal = (st.totals?.vram?.total) || 0;
  }catch(_){ hfVramTotal = 0; }
  // defaults per user request: empty search, Q4_K_M, Text, trending, Recommended
  $("#hf-query").value = "";
  hfQuant = "Q4_K_M";
  document.querySelectorAll("#hf-quant-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.quant===hfQuant));
  hfTask = "text-generation";
  document.querySelectorAll("#hf-task-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.task===hfTask));
  const pipeSel=$("#hf-pipeline-select"); if(pipeSel) pipeSel.value = hfTask;
  const sortSel=$("#hf-sort"); if(sortSel) sortSel.value = "trending";
  hfSort = "trending";
  hfRecommended = true;
  const recChip=$("#hf-recommended"); if(recChip){ recChip.classList.add("active"); recChip.dataset.recommended="true"; }
  // reset company/collection to All (recommended filter will limit to curated)
  hfCompany = ""; const compSel=$("#hf-company"); if(compSel) compSel.value="";
  hfCollection = ""; const collSel=$("#hf-collection"); if(collSel) collSel.value="";
  if(compSel) await loadHfCollections("");
  doHfSearch(true);
}
async function doHfSearch(resetPage){
  if(resetPage) hfPage = 0;
  hfQuery = $("#hf-query")?.value.trim() || "";
  hfSort = $("#hf-sort")?.value || "downloads";
  hfCompany = $("#hf-company")?.value || "";
  hfCollection = $("#hf-collection")?.value || "";
  const list = $("#hf-results"), status=$("#hf-status"), hint=$("#hf-results-hint");
  if(!list) return;
  // safetensors mode: no collections (they are GGUF-curated), no quant chips
  if(hfFormat === "safetensors" && hfCollection){ hfCollection = ""; const cs=$("#hf-collection"); if(cs) cs.value=""; }
  // if collection selected, show collection items instead of search
  if(hfCollection){
    list.innerHTML = `<div style="padding:12px"><div class="hf-skeleton" style="width:80%"></div><div class="hf-skeleton" style="width:60%"></div></div>`.repeat(2);
    if(status) status.textContent = "Loading collection…";
    try{
      const data = await api("GET", `/api/hf/collection?slug=${encodeURIComponent(hfCollection)}`);
      // backend already verified GGUF siblings (gguf_only) — every item
      // here is directly downloadable, no HF-page dead ends
      const models = (data.items||[]).map(it=> ({
        id: it.id, author: it.author, downloads: it.downloads||0, likes: it.likes||0, pipeline_tag: it.pipeline_tag||"", tags: it.tags||[], ggufCount: it.ggufCount||0, gated: false
      }));
      renderHfResults({models, total: models.length});
      if(status) status.textContent = models.length
        ? `${models.length} GGUF models in "${data.title||hfCollection}"`
        : `No GGUF models in "${data.title||hfCollection}" — only source/safetensors repos`;
      if(hint) hint.textContent = `${models.length} GGUF models in collection`;
      $("#hf-prev").disabled = true;
      $("#hf-next").disabled = true;
      return;
    }catch(e){
      list.innerHTML = `<p class="status-bad" style="padding:14px">${escHtml(e.message)} — try visiting <a href="https://huggingface.co/collections/${escHtml(hfCollection)}" target="_blank">HF collection page</a> to request access if gated.</p>`;
      if(status) status.textContent = e.message;
      return;
    }
  }
  list.innerHTML = `<div style="padding:12px"><div class="hf-skeleton" style="width:80%"></div><div class="hf-skeleton" style="width:60%"></div><div class="hf-skeleton" style="width:90%"></div></div>`.repeat(3);
  if(status) status.textContent = "Searching Hugging Face…";
  try{
    const qs = new URLSearchParams({query: hfQuery, sort: hfSort, limit:"20", page:String(hfPage), quant: hfQuant, pipeline_tag: hfTask, author: hfCompany, recommended: hfRecommended ? "true" : "false", fmt: hfFormat});
    const data = await api("GET", `/api/hf/search?${qs}`);
    renderHfResults(data);
    // gated hint
    // chunked pagination: backend returns exact_total only when it knows
    // the full count (snapshot path / HF exhausted). Otherwise page with
    // has_more — Next fetches the next chunk on demand.
    const gatedCount = (data.models||[]).filter(m=>m.gated).length;
    const recHint = hfRecommended ? " · ★ Recommended" : "";
    const exact = data.exact_total && data.total != null;
    const totalAll = data.total ?? 0;
    const hasMore = data.has_more ?? ((data.models||[]).length >= 20);
    const lastPage = exact ? Math.max(1, Math.ceil(totalAll/20)) : null;
    const pageLabel = exact ? `page ${hfPage+1}/${lastPage}` : `page ${hfPage+1}${hasMore ? "" : " (end)"}`;
    const countLabel = exact ? `${data.models?.length||0} of ${totalAll} models` : `${data.models?.length||0} models${hasMore ? " · more…" : ""}`;
    if(status) status.textContent = `${countLabel} · ${pageLabel}` + (gatedCount?` · ${gatedCount} gated`:"") + recHint;
    if(hint) hint.textContent = exact ? `${totalAll} ${hfFormat === "safetensors" ? "safetensors" : "GGUF"} models` + (hfCompany?` — ${hfCompany}`: hfRecommended?` — Recommended`:"") : `${hfFormat === "safetensors" ? "safetensors" : "GGUF"} models${hasMore ? " · more available" : ""}` + (hfCompany?` — ${hfCompany}`: hfRecommended?` — Recommended`:"");
    $("#hf-prev").disabled = hfPage===0;
    $("#hf-next").disabled = !hasMore;
  }catch(e){
    const msg = e.message||"";
    const isGated = /gated|401|403|access.*restricted|invite/i.test(msg);
    list.innerHTML = isGated
      ? `<p class="status-bad" style="padding:14px">This model is invite-only or requires agreement. Visit <a href="https://huggingface.co/${escHtml(hfQuery||hfCompany||'')}" target="_blank">huggingface.co/${escHtml(hfQuery||hfCompany||'')}</a> and request access from the maintainer, then paste your token in <code>Settings → Defaults → Hugging Face token</code> (or set <code>HF_TOKEN</code> env).</p><p class="hint" style="padding:0 14px">${escHtml(msg)}</p>`
      : `<p class="status-bad" style="padding:14px">${escHtml(msg)}</p>`;
    if(status) status.textContent = e.message;
  }
}
function renderHfResults(data){
  const list=$("#hf-results"); if(!list) return;
  const models = data.models||[];
  const st = hfFormat === "safetensors";
  if(!models.length){ list.innerHTML = `<p class="hint" style="padding:14px">No ${st?"safetensors":"GGUF"} models found for “${escHtml(hfQuery)}”. Try Qwen, Mistral, Phi.</p>`; $("#hf-detail").style.display="none"; return; }
  list.innerHTML = models.map(m=>`
    <div class="hf-card ${hfActiveId===m.id?'active':''}" data-id="${escHtml(m.id)}">
      <img class="hf-avatar" data-author="${escHtml(m.author)}" src="${escHtml(proxiedImgSrc(`https://ui-avatars.com/api/?name=${encodeURIComponent(m.author)}&background=171c29&color=9aa5bd&size=56&bold=true`))}" alt="${escHtml(m.author)}" loading="lazy">
      <div class="hf-card-body">
        <div class="hf-card-title">${escHtml(m.id)} <span class="hf-quant-badge">${st?(m.safetensorsCount?"ST":""):(m.ggufCount||"")+" GGUF"}</span> ${m.gated?'<span class="badge" style="background:var(--warn-dim);color:var(--warn)"><i class="fa-solid fa-lock"></i> gated</span>':""}</div>
        <div class="hf-card-meta">
          <span><i class="fa-solid fa-download"></i> ${m.downloads.toLocaleString()}</span>
          <span><i class="fa-solid fa-heart"></i> ${m.likes.toLocaleString()}</span>
          ${m.pipeline_tag?`<span class="badge" style="background:var(--accent-dim);color:var(--accent)">${escHtml(m.pipeline_tag)}</span>`:""}
          ${m.tags?.includes("gguf")?`<span class="badge">gguf</span>`:""}
        </div>
        <div class="hint" style="margin-top:4px; font-size:11px">${escHtml((m.tags||[]).slice(0,4).join(" · "))}</div>
      </div>
    </div>`).join("");
  list.querySelectorAll(".hf-card").forEach(card=>{
    card.onclick = ()=> selectHfModel(card.dataset.id);
  });
  // lazy load real company avatars
  const uniq = [...new Set(models.map(m=>m.author).filter(Boolean))];
  uniq.forEach(async author=>{
    const url = await getHfAvatar(author);
    if(url){
      list.querySelectorAll(`img.hf-avatar[data-author="${CSS.escape(author)}"]`).forEach(img=>{
        img.src = proxiedImgSrc(url);
        img.onerror = ()=> { img.src = proxiedImgSrc(`https://ui-avatars.com/api/?name=${encodeURIComponent(author)}&background=171c29&color=9aa5bd&size=56&bold=true`); };
      });
    }
  });
}
async function selectHfModel(modelId){
  hfActiveId = modelId;
  document.querySelectorAll("#hf-results .hf-card").forEach(c=> c.classList.toggle("active", c.dataset.id===modelId));
  if(hfFormat === "safetensors") return selectHfSafetensorsModel(modelId);
  const detail=$("#hf-detail"); if(!detail) return;
  detail.style.display="block";
  detail.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading GGUF files…</p>`;
  try{
    const [author, name] = modelId.split("/");
    const data = await api("GET", `/api/hf/models/${encodeURIComponent(author)}/${encodeURIComponent(name)}/gguf`);
    const files = data.siblings||[];
    if(!files.length){
      detail.innerHTML = `<p class="status-bad" style="padding:14px">No GGUF files in <strong>${escHtml(data.id)}</strong> — this is a source (safetensors) repo, not a downloadable quant. Pick a <code>-GGUF</code> repo from the list instead, or grab the files from <a href="https://huggingface.co/${escHtml(data.id)}" target="_blank">the HF page</a>.</p>`;
      return;
    }
    // group split shards into one combined entry:
    // base-00001-of-00002.gguf + base-00002-of-00002.gguf → single
    // "base.gguf (2 shards)" row whose button downloads all shards together.
    const entries = [];
    {
      const bySplit = new Map();
      for (const f of files) {
        const m = (f.rfilename || "").match(/^(.*)-(\d+)-of-(\d+)\.gguf$/i);
        if (!m) {
          entries.push({ single: true, file: f.rfilename, quant: f.quant || "", size: f.size || 0, shards: [f] });
          continue;
        }
        const stem = m[1];
        const total = (m[3] || "").replace(/^0+/, "") || m[3];
        const key = `${stem.toLowerCase()}__of__${total}`;
        let g = bySplit.get(key);
        if (!g) {
          g = { single: false, stem, total: parseInt(total, 10) || 0, shards: [] };
          bySplit.set(key, g);
          entries.push(g);
        }
        g.shards.push(f);
      }
      for (const e of entries) {
        if (e.single) continue;
        e.shards.sort((a, b) => String(a.rfilename).localeCompare(String(b.rfilename)));
        e.file = e.shards[0].rfilename; // backend all_shards:true resolves the rest
        e.quant = e.shards[0].quant || "";
        e.size = e.shards.reduce((a, s) => a + (s.size || 0), 0);
        const base = e.stem.includes("/") ? e.stem.split("/").pop() : e.stem;
        e.display = `${base}.gguf (${e.shards.length} shards)`;
      }
      entries.sort((a, b) => (a.size || 0) - (b.size || 0));
    }
    // sort by size asc, find recommended (largest that fits 90% vram, else smallest)
    let recIdx = -1;
    if (hfVramTotal && entries.length) {
      const limit = hfVramTotal * 0.9 * 1e9;
      let best = -1;
      for (let i = 0; i < entries.length; i++) { const sz = entries[i].size || 0; if (sz <= limit && sz > best) { best = sz; recIdx = i; } }
      if (recIdx === -1) { // all too large, pick smallest
        recIdx = entries.reduce((a, b, i, arr) => (arr[a].size || Infinity) < (b.size || Infinity) ? a : i, 0);
      }
    }
    const storeVal = $("#hf-store")?.value || $("#dl-store")?.value || "llm-models";
    const avatarUrl = await getHfAvatar(data.author) || `https://ui-avatars.com/api/?name=${encodeURIComponent(data.author)}&background=171c29&color=9aa5bd&size=56&bold=true`;
    detail.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:start; gap:8px; margin-bottom:10px">
        <div style="display:flex; gap:10px; align-items:center"><img class="hf-avatar" src="${escHtml(proxiedImgSrc(avatarUrl))}" alt="${escHtml(data.author)}" style="width:32px;height:32px"><div><strong>${escHtml(data.id)}</strong> ${data.gated?'<span class="badge" style="background:var(--warn-dim);color:var(--warn)"><i class="fa-solid fa-lock"></i> gated</span>':''}<br><small class="hint">${escHtml(data.pipeline_tag||"")} · <i class="fa-solid fa-download"></i> ${data.downloads.toLocaleString()} · <i class="fa-solid fa-heart"></i> ${data.likes.toLocaleString()}</small></div></div>
        <a href="https://huggingface.co/${escHtml(data.id)}" target="_blank" class="small"><i class="fa-solid fa-external-link"></i> HF</a>
      </div>
      ${data.gated ? `<div id="hf-access-box" style="margin-bottom:10px"><p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Checking saved token against this repo…</p></div>` : ""}
      <div class="hint" style="margin-bottom:8px">${entries.length} download option(s)${files.length !== entries.length ? ` (combined from ${files.length} files)` : ` — ${files.length} GGUF file(s)`} — pick quant/size. Highlighted = fits your VRAM (~${hfVramTotal || "?"} GB). Split models show as one row and download <strong>all shards</strong> together.</div>
      ${entries.map((e, i) => {
        const label = e.single ? "Download" : `Download all ${e.shards.length} shards`;
        const sub = e.single
          ? `${escHtml(e.quant)} · ${fmtBytes(e.size)}`
          : `${escHtml(e.quant)} · ${fmtBytes(e.size)} total`;
        const shardTip = e.single ? "" : ` <span class="badge">${e.shards.length} shards</span>`;
        const shardList = e.single ? "" : `<small class="hint" style="display:block; word-break:break-all" title="${escHtml(e.shards.map(s => s.rfilename).join("\n"))}">${escHtml(e.shards[0].rfilename)} … +${e.shards.length - 1} more</small>`;
        return `
        <div class="hf-file-row ${i === recIdx ? 'recommended' : ''}">
          <div style="min-width:0; flex:1">
            <div class="hf-detail-file">${escHtml(e.single ? e.file : e.display)}</div>
            <small class="hint">${sub} ${i === recIdx ? '<span class="badge" style="background:var(--ok-dim);color:var(--ok)">recommended</span>' : ''}${shardTip}</small>
            ${shardList}
          </div>
          <button class="small" data-file="${escHtml(e.file)}" style="background:var(--accent-dim);border-color:var(--accent);color:var(--accent); margin-left:8px">${label}</button>
        </div>`;}).join("")}
      <small class="hint" style="display:block; margin-top:8px">Downloads via <code>hf://</code> to <code>store://${escHtml(storeVal)}</code> — streamed to disk, bypasses HF cache (same as custom URL path).</small>`;
    detail.querySelectorAll("button[data-file]").forEach(btn=>{
      const origLabel = btn.textContent;
      btn.onclick = async ()=>{
        const file = btn.dataset.file;
        const store = $("#hf-store")?.value || $("#dl-store")?.value || "llm-models";
        const link = `hf://${data.id}/${file}`;
        btn.disabled=true; btn.textContent="starting…";
        try{
          const r = await api("POST","/api/downloads",{link, store, all_shards:true});
          const jobs = r.jobs || [r];
          const started = jobs.filter(j => j.status !== "done");
          const existed = jobs.filter(j => j.status === "done");
          toast(started.length
            ? `Download started (${started.length}/${jobs.length} shard${jobs.length > 1 ? "s" : ""}) → ${started[0].dest}${existed.length ? ` · ${existed.length} already existed` : ""}`
            : `Already exists: ${jobs[0].dest}`, started.length ? "ok" : "warn");
          loadDownloads();
          // a download starting is the event of interest — jump to its subtab
          if (started.length) dlSubTab("jobs");
        }catch(e){
          const msg = e.message||"";
          const isGated = /gated|401|403|access.*restricted|invite|agree/i.test(msg);
          if(isGated){
            toast(`Gated model — visit https://huggingface.co/${data.id} to request access, paste your token in Settings → Defaults → Hugging Face token, then retry. ${msg}`, "bad", 9000);
            detail.insertAdjacentHTML("afterbegin", `<p class="status-bad">Download failed: gated. <a href="https://huggingface.co/${escHtml(data.id)}" target="_blank">Open HF page to request access</a></p>`);
          } else toast(msg,"bad");
        }
        btn.disabled=false; btn.textContent=origLabel;
      };
    });
    if (data.gated) checkHfAccess(data, entries.length ? entries[0].file : "");
  }catch(e){
    const msg = e.message||"";
    const isGated = /gated|401|403|access.*restricted|invite|agree/i.test(msg);
    detail.innerHTML = isGated
      ? `<p class="status-bad">This model is invite-only/gated. Visit <a href="https://huggingface.co/${escHtml(modelId)}" target="_blank">huggingface.co/${escHtml(modelId)}</a> and request access from the maintainer, then paste your token in <code>Settings → Defaults → Hugging Face token</code> and retry.</p><p class="hint">${escHtml(msg)}</p>`
      : `<p class="status-bad">${escHtml(msg)}</p>`;
  }
}
/* Safetensors detail: whole-repo download (config.json + *.safetensors
   land in store/<ModelName>/ — what model_parser scans and `vllm serve`
   takes as a path). Sizes come from the detail endpoint (blobs=true). */
async function selectHfSafetensorsModel(modelId){
  const detail=$("#hf-detail"); if(!detail) return;
  detail.style.display="block";
  detail.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Loading repo files…</p>`;
  try{
    const [author, name] = modelId.split("/");
    const data = await api("GET", `/api/hf/models/${encodeURIComponent(author)}/${encodeURIComponent(name)}/safetensors`);
    const files = data.siblings||[];
    const totalSize = files.reduce((a, f) => a + (f.size || 0), 0);
    const storeVal = $("#hf-store")?.value || $("#dl-store")?.value || "llm-models";
    const avatarUrl = await getHfAvatar(data.author) || `https://ui-avatars.com/api/?name=${encodeURIComponent(data.author)}&background=171c29&color=9aa5bd&size=56&bold=true`;
    detail.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:start; gap:8px; margin-bottom:10px">
        <div style="display:flex; gap:10px; align-items:center"><img class="hf-avatar" src="${escHtml(proxiedImgSrc(avatarUrl))}" alt="${escHtml(data.author)}" style="width:32px;height:32px"><div><strong>${escHtml(data.id)}</strong> ${data.gated?'<span class="badge" style="background:var(--warn-dim);color:var(--warn)"><i class="fa-solid fa-lock"></i> gated</span>':''}<br><small class="hint">${escHtml(data.pipeline_tag||"")} · <i class="fa-solid fa-download"></i> ${data.downloads.toLocaleString()} · <i class="fa-solid fa-heart"></i> ${data.likes.toLocaleString()}</small></div></div>
        <a href="https://huggingface.co/${escHtml(data.id)}" target="_blank" class="small"><i class="fa-solid fa-external-link"></i> HF</a>
      </div>
      ${data.gated ? `<div id="hf-access-box" style="margin-bottom:10px"><p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Checking saved token against this repo…</p></div>` : ""}
      <p class="hint" style="margin-bottom:8px">Source repo (vLLM format). Downloads <strong>config.json + *.safetensors</strong> (+ tokenizer) into <code>store://${escHtml(storeVal)}/${escHtml(name)}/</code> — the format vLLM serves directly and model discovery picks up.</p>
      <div class="hf-file-row recommended">
        <div style="min-width:0; flex:1">
          <div class="hf-detail-file">${escHtml(name)}/</div>
          <small class="hint">${files.length} safetensors file${files.length===1?"":"s"} · ${fmtBytes(totalSize)} total${files.length !== files.filter(f=>f.size).length ? " · some sizes unknown" : ""}</small>
          <small class="hint" style="display:block; word-break:break-all" title="${escHtml(files.map(f => f.rfilename).join("\n"))}">${escHtml(files[0]?.rfilename || "")}${files.length > 1 ? ` … +${files.length - 1} more` : ""}</small>
        </div>
        <button id="hf-st-download" style="background:var(--accent-dim);border-color:var(--accent);color:var(--accent); margin-left:8px">Download repo</button>
      </div>
      <details style="margin-top:8px">
        <summary class="hint" style="cursor:pointer">${files.length} file(s)</summary>
        ${files.map(f => `<small class="hint" style="display:block; word-break:break-all">${escHtml(f.rfilename)} — ${fmtBytes(f.size||0)}</small>`).join("")}
      </details>
      <small class="hint" style="display:block; margin-top:8px">Whole-repo download via <code>snapshot_download</code> — GGUF/bin/checkpoint files in the repo are skipped automatically.</small>`;
    const btn = detail.querySelector("#hf-st-download");
    if (btn) btn.onclick = async ()=>{
      btn.disabled=true; btn.textContent="starting…";
      const store = $("#hf-store")?.value || $("#dl-store")?.value || "llm-models";
      try{
        await api("POST","/api/downloads",{link:`hf://${data.id}`, store, repo:true});
        toast(`Repo download started → ${data.id} into store://${store}`, "ok");
        loadDownloads();
        dlSubTab("jobs");
      }catch(e){
        const msg = e.message||"";
        const isGated = /gated|401|403|access.*restricted|invite|agree/i.test(msg);
        if(isGated){
          toast(`Gated model — visit https://huggingface.co/${data.id} to request access, paste your token in Settings → Defaults → Hugging Face token, then retry. ${msg}`, "bad", 9000);
        } else toast(msg,"bad");
      }
      btn.disabled=false; btn.textContent="Download repo";
    };
    if (data.gated) checkHfAccess(data, files[0]?.rfilename || "");
  }catch(e){
    const msg = e.message||"";
    const isGated = /gated|401|403|access.*restricted|invite|agree/i.test(msg);
    detail.innerHTML = isGated
      ? `<p class="status-bad">This model is invite-only/gated. Visit <a href="https://huggingface.co/${escHtml(modelId)}" target="_blank">huggingface.co/${escHtml(modelId)}</a> and request access from the maintainer, then paste your token in <code>Settings → Defaults → Hugging Face token</code> and retry.</p><p class="hint">${escHtml(msg)}</p>`
      : `<p class="status-bad">${escHtml(msg)}</p>`;
  }
}
/* Live token check for gated repos: 1-byte range GET proves the saved
   token can download. Granted → green banner, warning dismissed.
   Denied → warning with the concrete reason + a Re-check button (use
   right after clicking Agree on the HF page). */
async function checkHfAccess(data, testFile) {
  const box = $("#hf-access-box");
  if (!box) return;
  const hfLink = `https://huggingface.co/${data.id}`;
  const deniedHTML = (reason) => `
    <div class="hint-box" style="border-left:3px solid var(--warn); padding-left:8px">
      <i class="fa-solid fa-lock"></i> <strong>Gated model</strong> — requires agreement.
      Visit <a href="${escHtml(hfLink)}" target="_blank">${escHtml(data.id)}</a>, click “Agree and access repository”,
      then paste your token in <code>Settings → Defaults → Hugging Face token</code> and retry.
      <br><small class="hint">Check: ${escHtml(reason)}</small>
      <div style="margin-top:6px"><button class="small" id="hf-recheck">Re-check access</button></div>
    </div>`;
  try {
    const parts = (data.id || "").split("/");
    const author = parts.shift() || "";
    const name = parts.join("/");
    const q = testFile ? `?file=${encodeURIComponent(testFile)}` : "";
    const r = await api("GET", `/api/hf/access/${encodeURIComponent(author)}/${encodeURIComponent(name)}${q}`);
    if (r.access === "granted") {
      box.innerHTML = `
        <div class="hint-box" style="border-left:3px solid var(--ok); padding-left:8px">
          <i class="fa-solid fa-circle-check" style="color:var(--ok)"></i>
          <strong>Access granted</strong> — your saved token works for this repo. Downloads will succeed.
        </div>`;
    } else {
      box.innerHTML = deniedHTML(r.reason || `HF answered ${r.status || "denied"}`);
      on("#hf-recheck", "click", () => {
        box.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Checking saved token against this repo…</p>`;
        checkHfAccess(data, testFile);
      });
    }
  } catch (e) {
    box.innerHTML = deniedHTML(`couldn't verify: ${e.message}`);
    on("#hf-recheck", "click", () => {
      box.innerHTML = `<p class="hint"><i class="fa-solid fa-spinner fa-spin"></i> Checking saved token against this repo…</p>`;
      checkHfAccess(data, testFile);
    });
  }
}
function hfSetQuant(q){
  hfQuant = q||"";
  document.querySelectorAll("#hf-quant-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.quant===hfQuant));
  // reflective: company/collection lists depend on quant
  loadHfCompanies();
  if(hfCompany) loadHfCollections(hfCompany);
  doHfSearch(true);
}
function hfSetTask(t){
  hfTask = t||"";
  document.querySelectorAll("#hf-task-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.task===hfTask));
  const sel=$("#hf-pipeline-select"); if(sel) sel.value = hfTask;
  // reflective: company/collection lists depend on task
  loadHfCompanies();
  if(hfCompany) loadHfCollections(hfCompany);
  doHfSearch(true);
}
on("#hf-search-btn","click", ()=> doHfSearch(true));
on("#hf-query","keydown", (e)=>{ if(e.key==="Enter") doHfSearch(true); });
on("#hf-query","input", ()=>{
  clearTimeout(hfDebounce);
  hfDebounce = setTimeout(()=> doHfSearch(true), 400);
});
on("#hf-sort","change", ()=> doHfSearch(true));
on("#hf-store","change", ()=>{
  const v=$("#hf-store").value; if(v && $("#dl-store")) $("#dl-store").value=v;
  // re-render detail to update store hint
  if(hfActiveId) selectHfModel(hfActiveId);
});
document.querySelectorAll("#hf-quant-chips .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=> hfSetQuant(b.dataset.quant||""));
});
document.querySelectorAll("#hf-task-chips .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=> hfSetTask(b.dataset.task||""));
});
on("#hf-pipeline-select","change", ()=>{
  const v=$("#hf-pipeline-select").value || "";
  hfTask = v;
  document.querySelectorAll("#hf-task-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.task===hfTask));
  loadHfCompanies();
  if(hfCompany) loadHfCollections(hfCompany);
  doHfSearch(true);
});
on("#hf-recommended","click", ()=>{
  hfRecommended = !hfRecommended;
  const chip=$("#hf-recommended"); if(chip){ chip.classList.toggle("active", hfRecommended); chip.dataset.recommended = hfRecommended?"true":"false"; }
  loadHfCompanies();
  if(hfCompany) loadHfCollections(hfCompany);
  doHfSearch(true);
});
on("#hf-company","change", async ()=>{
  hfCompany = $("#hf-company").value || "";
  hfCollection = "";
  const sel=$("#hf-collection"); if(sel) sel.value="";
  await loadHfCollections(hfCompany);
  doHfSearch(true);
});
on("#hf-collection","change", ()=>{
  hfCollection = $("#hf-collection").value || "";
  doHfSearch(true);
});
on("#hf-company-search","input", (e)=>{
  clearTimeout(hfCompanyFilterDebounce);
  hfCompanyFilterDebounce = setTimeout(()=> filterHfCompaniesLocal(e.target.value), 200);
});
on("#hf-prev","click", ()=>{ if(hfPage>0){ hfPage--; doHfSearch(false); } });
on("#hf-next","click", ()=>{ hfPage++; doHfSearch(false); });
/* format toggle: GGUF quants (llama.cpp) vs source repos (vLLM). Swaps the
   quant-row/task visibility, resets filters that don't apply, re-searches. */
function hfSetFormat(fmt){
  if(hfFormat === fmt) return;
  hfFormat = fmt;
  document.querySelectorAll("#hf-fmt-chips .hw-chip").forEach(b=> b.classList.toggle("active", b.dataset.fmt===hfFormat));
  const quantRow = $("#hf-quant-row");
  if(quantRow) quantRow.style.display = hfFormat === "safetensors" ? "none" : "";
  const hint = $("#hf-fmt-hint");
  if(hint) hint.textContent = hfFormat === "safetensors"
    ? "Safetensors = source weights (config.json + *.safetensors) for vLLM — bigger downloads, no quantization"
    : "GGUF = ready-to-run quantized weights for llama.cpp";
  const collectionSel = $("#hf-collection"); if(collectionSel){ collectionSel.value=""; hfCollection=""; }
  hfActiveId = null;
  const detail = $("#hf-detail"); if(detail){ detail.style.display="none"; }
  doHfSearch(true);
}
document.querySelectorAll("#hf-fmt-chips .hw-chip").forEach(b=>{
  b.addEventListener("click", ()=> hfSetFormat(b.dataset.fmt||"gguf"));
});

/* ---------------- flow visualizer ---------------- */
let flowActive = false;
let flowMachines = [];   // machine topology for the diagram
let flowGpuActivity = {}; // machine -> gpu key -> {t: last activity ts, s: model_id source or null}
let flowModelLoadedAt = {}; // model_id -> ts when first seen loaded (load-order sched attribution)
let flowRenderTimer = null;
let flowSchedWs = null;
let flowUpstreamWs = null;
let flowSchedEvents = [];   // recent per-tensor backend events
let flowRpcMaps = null;     // model_id -> {file, maps: {variantLabel: {RPC0: {machine, resource}}}} from /api/rpc-map
let flowLastServers = null; // servers of the model that streamed most recently (sched attribution hint)
let flowLastModel = null;   // model_id that streamed most recently
let flowModelLastSeen = {}; // model_id -> ts of last upstream chunk (per-model serving state)
let flowFileToId = {};      // gguf filename -> model_id (incl. variant keys)
let flowModelMachines = {}; // model filename -> servers
let flowModelExposes = {}; // model_id -> expose map
let flowModelIdServers = {}; // model_id -> servers
let flowLoadedModels = [];  // model_ids currently loaded (left side)
window.__flowDebugGet = () => ({loaded: flowLoadedModels, machines: flowModelMachines, idServers: flowModelIdServers, activity: flowGpuActivity});
let flowViewport = { x: 0, y: 0, scale: 1 };
let flowPanSetup = false;
let flowLayout = {};        // node id -> {x,y} persisted to backend
let flowDrag = null;        // {id, startX, startY, origX, origY}

// window after which a model is considered no longer streaming (ms).
// upstream chunks arrive per-token, so 2.5s of silence = stream ended.
const FLOW_SERVING_WINDOW = 2500;

// Rebuild all model->servers/expose/file maps from /api/models.
// Called on init and every 3s poll so newly-saved models map correctly.
function buildFlowModelMaps(models) {
  flowModelMachines = {};
  flowModelIdServers = {};
  flowModelExposes = {};
  flowFileToId = {};
  for (const m of models) {
    const fname = (m.base_path || "").split("/").pop();
    const mid = (m.model_id || "").trim();
    if (fname && (m.servers || []).length) flowModelMachines[fname] = m.servers;
    if (fname && m.expose) flowModelExposes[fname] = m.expose;
    if (mid && (m.servers || []).length) flowModelIdServers[mid] = m.servers;
    if (mid && fname) flowFileToId[fname] = mid;
    if (m.expose) flowModelExposes[mid] = m.expose;
    for (const v of (m.variants || [])) {
      const vf = (v.base_path || m.base_path || "").split("/").pop();
      const vsrv = v.servers || m.servers || [];
      const vexp = v.expose || m.expose;
      const lab = (v.label || "").trim();
      const vkey = lab ? `${mid}--${lab}` : mid;
      // variants share the base model's GGUF filename — never let variant
      // metadata overwrite the BASE fname keys, or streaming the plain
      // single-node model lights the variant's extra machines (e.g. the
      // rpc-X3 variant lighting aibox3 while only aibox1 serves)
      const vfIsBase = !vf || vf === fname;
      if (!vfIsBase && vsrv.length) flowModelMachines[vf] = vsrv;
      if (!vfIsBase && vexp) flowModelExposes[vf] = vexp;
      if (vkey && vsrv.length) flowModelIdServers[vkey] = vsrv;
      if (!vfIsBase && vf && vkey) flowFileToId[vf] = vkey;
      if (vexp) flowModelExposes[vkey] = vexp;
    }
  }
}

// model_id of the loaded model with the freshest upstream chunk, or null.
// Sched log lines are anonymous, so only a still-streaming model is a safe attribution.
function flowFreshestModelId() {
  let best = null, bestTs = 0;
  const now = Date.now();
  for (const id of flowLoadedModels) {
    const ts = flowModelLastSeen[id] || 0;
    if (ts > bestTs && (now - ts) < FLOW_SERVING_WINDOW) { bestTs = ts; best = id; }
  }
  return best;
}

function flowActTs(v) {
  return (v && typeof v === "object") ? (v.t || 0) : (v || 0);
}

function flowActSrc(v) {
  return (v && typeof v === "object") ? (v.s || null) : null;
}

function flowSameModel(a, b) {
  if (!a || !b || a === b) return !!a && a === b;
  const ab = a.indexOf("--"), bb = b.indexOf("--");
  const aBase = ab > 0 ? a.slice(0, ab) : a;
  const bBase = bb > 0 ? b.slice(0, bb) : b;
  return aBase === bBase;
}

function flowMarkActivity(machine, key, modelId) {
  if (!machine) return;
  flowGpuActivity[machine] = flowGpuActivity[machine] || {};
  flowGpuActivity[machine][key] = { t: Date.now(), s: modelId || null };
}

// prune stale activity entries + unseen models so the activity maps can't
// grow without bound over long sessions (memory-leak guard)
setInterval(() => {
  const now = Date.now();
  for (const [mn, slots] of Object.entries(flowGpuActivity)) {
    for (const [k, v] of Object.entries(slots)) {
      if (now - flowActTs(v) > 60000) delete slots[k];
    }
    if (!Object.keys(slots).length) delete flowGpuActivity[mn];
  }
  for (const [id, ts] of Object.entries(flowModelLastSeen)) {
    if (now - ts > 300000) delete flowModelLastSeen[id];
  }
}, 60000);

function flowTrackLoadedModels(list) {
  const now = Date.now();
  const set = new Set(list || []);
  for (const id of set) {
    if (!(id in flowModelLoadedAt)) flowModelLoadedAt[id] = now;
  }
  for (const id of Object.keys(flowModelLoadedAt)) {
    if (!set.has(id)) delete flowModelLoadedAt[id];
  }
}

function flowNewestLoadedId() {
  let best = null, bestTs = 0;
  for (const id of flowLoadedModels) {
    const ts = flowModelLoadedAt[id] || 0;
    if (ts >= bestTs) { bestTs = ts; best = id; }
  }
  return best;
}

async function initFlowVisualizer() {
  if (flowActive) {
    // re-entering the tab: waves may have sized 0x0 while hidden
    startFlowWaves();
    requestFlowRender();
    return;
  }
  flowActive = true;
  startFlowWaves();   // waves are traffic-independent — run from first open
  const status = $("#flow-status");
  if (status) status.textContent = "connecting…";
  // load machine topology for the diagram
  try { flowMachines = await api("GET", "/api/machines"); } catch (_) { flowMachines = []; }
  // build model filename -> servers map so we know which machines to light for each model
  try {
    const models = await api("GET", "/api/models");
    buildFlowModelMaps(models);
  } catch (_) { buildFlowModelMaps([]); }
  // exact RPC<N> -> (machine, resource) maps for sched attribution
  try { flowRpcMaps = await api("GET", "/api/rpc-map"); } catch (_) { flowRpcMaps = null; }
  // loaded models for the left side
  try {
    const l = await api("GET", "/api/swap/loaded");
    flowLoadedModels = l.loaded || [];
  } catch (_) { flowLoadedModels = []; }
  flowTrackLoadedModels(flowLoadedModels);
  // persisted node positions
  try { flowLayout = await api("GET", "/api/flow/layout"); } catch (_) { flowLayout = {}; }
  renderFlowDiagram();
  // upstream token stream -> which model is currently serving (direct or RPC)
  try { connectUpstreamWs(); } catch (_) {}
  // real per-tensor backend assignment (GGML_SCHED_DEBUG=2) -> cross-device movement
  try {
    flowSchedWs = new WebSocket(`ws://${location.host}/ws/flow/sched`);
    flowSchedWs.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch (_) { return; }
      handleSchedEvent(ev);
    };
    flowSchedWs.onopen = () => { if (status) status.textContent = "live"; };
    flowSchedWs.onclose = (e) => {
      console.warn("[flow] sched ws closed", e.code, e.reason);
      flowActive = false;
      if (status) status.textContent = "disconnected";
      if ($("#tab-visualizer")?.classList.contains("active")) {
        setTimeout(initFlowVisualizer, 2000);
      }
    };
    flowSchedWs.onerror = (e) => { console.error("[flow] sched ws error", e); if (status) status.textContent = "error"; };
  } catch (e) { console.error("[flow] sched ws failed", e); }
  // warn if a loaded model is actively serving but no sched events arrive
  // (single-GPU models legitimately have no cross-device sched, so only warn when streaming)
  let flowWarnedNoSched = false;
  let flowMachinesPollTick = 0;
  setTimeout(() => {
    const streaming = Object.keys(flowModelLastSeen).filter(id =>
      Date.now() - (flowModelLastSeen[id] || 0) < FLOW_SERVING_WINDOW);
    if (streaming.length && flowLoadedModels.length && !flowSchedEvents.length && !flowWarnedNoSched) {
      flowWarnedNoSched = true;
      console.warn("[flow] serving", streaming, "but no sched events — single-GPU models have no cross-device sched; upstream token stream drives the animation");
    }
  }, 8000);
  // keep the diagram fresh and keep loaded-model list live. 500ms so the
  // 1500ms activity decay visibly turns off promptly — a slow poll made
  // lights/edges linger long after activity stopped (laggy feel).
  if (flowRenderTimer) clearInterval(flowRenderTimer);
  flowRenderTimer = setInterval(async () => {
    if (!document.getElementById("tab-visualizer")?.classList.contains("active")) return;
    // machine topology refresh (every ~6th tick): a new card / RAM change
    // on any machine shows up in the diagram without a page reload
    if (!flowMachinesPollTick) flowMachinesPollTick = 0;
    if (++flowMachinesPollTick % 6 === 0) {
      try {
        const fresh = await api("GET", "/api/machines");
        if (JSON.stringify(fresh) !== JSON.stringify(flowMachines)) {
          flowMachines = fresh;
          flowLastSvg = null;  // force redraw even if layout looks same
        }
      } catch (_) {}
    }
    try {
      const l = await api("GET", "/api/swap/loaded");
      const loaded = l.loaded || [];
      if (JSON.stringify(loaded) !== JSON.stringify(flowLoadedModels)) flowLoadedModels = loaded;
      flowTrackLoadedModels(flowLoadedModels);
    } catch (_) {}
    try {
      const models = await api("GET", "/api/models");
      buildFlowModelMaps(models);
    } catch (_) {}
    // vLLM refinement: scrape the upstream /metrics (per-model counters ->
    // token deltas, KV usage, queue depth). Drives the same
    // flowMarkActivity path the ws stats pulse uses but at scrape
    // cadence with REAL throughput; the ws remains the activity signal
    // between scrapes. ~every 4th tick (2s) is plenty — counters move
    // with request traffic, not per token.
    if ((appSettings && appSettings.engine) === "vllm") {
      if (++flowVllmMetricsTick % 4 === 0) pollVllmFlowMetrics();
    }
    if (!flowDrag) renderFlowDiagram();
  }, 500);
  // n8n-style pan/zoom: wheel to zoom toward cursor, drag to pan
  if (!flowPanSetup) {
    flowPanSetup = true;
    const canvas = $("#flow-canvas");
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const newScale = Math.min(3, Math.max(0.3, flowViewport.scale * factor));
      flowViewport.x = mx - (mx - flowViewport.x) * (newScale / flowViewport.scale);
      flowViewport.y = my - (my - flowViewport.y) * (newScale / flowViewport.scale);
      flowViewport.scale = newScale;
      const vp = document.getElementById("flow-viewport");
      if (vp) vp.setAttribute("transform", `translate(${flowViewport.x},${flowViewport.y}) scale(${flowViewport.scale})`);
    }, {passive: false});
    let panning = false, startX, startY, startPanX, startPanY;
    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      panning = true;
      canvas.classList.add("grabbing");
      startX = e.clientX; startY = e.clientY;
      startPanX = flowViewport.x; startPanY = flowViewport.y;
    });
    window.addEventListener("mousemove", (e) => {
      if (!panning) return;
      flowViewport.x = startPanX + (e.clientX - startX);
      flowViewport.y = startPanY + (e.clientY - startY);
      const vp = document.getElementById("flow-viewport");
      if (vp) vp.setAttribute("transform", `translate(${flowViewport.x},${flowViewport.y}) scale(${flowViewport.scale})`);
    });
    window.addEventListener("mouseup", () => {
      if (panning) { panning = false; canvas.classList.remove("grabbing"); }
    });
    document.getElementById("flow-zoom-in")?.addEventListener("click", () => {
      flowViewport.scale = Math.min(3, flowViewport.scale * 1.2);
      document.getElementById("flow-viewport")?.setAttribute("transform", `translate(${flowViewport.x},${flowViewport.y}) scale(${flowViewport.scale})`);
    });
    document.getElementById("flow-zoom-out")?.addEventListener("click", () => {
      flowViewport.scale = Math.max(0.3, flowViewport.scale / 1.2);
      document.getElementById("flow-viewport")?.setAttribute("transform", `translate(${flowViewport.x},${flowViewport.y}) scale(${flowViewport.scale})`);
    });
    document.getElementById("flow-zoom-fit")?.addEventListener("click", () => {
      flowViewport = {x:0, y:0, scale:1};
      document.getElementById("flow-viewport")?.setAttribute("transform", `translate(0,0) scale(1)`);
    });
    document.getElementById("flow-zoom-reset")?.addEventListener("click", () => {
      flowViewport = {x:0, y:0, scale:1};
      document.getElementById("flow-viewport")?.setAttribute("transform", `translate(0,0) scale(1)`);
    });
    document.getElementById("flow-auto")?.addEventListener("click", async () => {
      flowLayout = {};
      try { await api("POST", "/api/flow/layout", {layout:{}}); } catch(_){}
      renderFlowDiagram();
    });
  }
}

function connectUpstreamWs() {
  const ws = new WebSocket(`ws://${location.host}/ws/flow/upstream`);
  flowUpstreamWs = ws;
  // ignore the first second after open — belt+braces against any backend
  // history replay so a page reload never flares up old activity
  const openedAt = Date.now();
  ws.onmessage = (e) => {
    if (Date.now() - openedAt < 1000) return;
    let ev;
    try { ev = JSON.parse(e.data); } catch (_) { console.warn("[flow] upstream bad json", e.data); return; }
    handleUpstreamEvent(ev);
  };
  // reconnect on drop: llama-swap restarts / model swaps end the upstream
  // SSE tail, which closes this ws — without reconnect the upstream
  // fallback (the ONLY driver for single-node single-GPU models) stays
  // dead until a manual page reload
  ws.onclose = () => {
    if (flowUpstreamWs !== ws) return; // replaced by a newer connection
    flowUpstreamWs = null;
    console.warn("[flow] upstream ws closed — reconnecting in 2s");
    if ($("#tab-visualizer")?.classList.contains("active")) {
      setTimeout(() => {
        if (!flowUpstreamWs && $("#tab-visualizer")?.classList.contains("active")) {
          try { connectUpstreamWs(); } catch (_) {}
        }
      }, 2000);
    }
  };
  ws.onerror = (e) => console.error("[flow] upstream ws error", e);
}

function handleUpstreamEvent(ev) {
  // vLLM engine-stats events (10s cadence): same "is this model serving"
  // signal as a llama.cpp token chunk, but carrying real throughput
  // numbers. The pulse decays over the 10s gap via FLOW_SERVING_WINDOW.
  if (ev.type === "vllm_stats") {
    handleVllmStatsEvent(ev);
    return;
  }
  // hidden/inactive tab: still record lastSeen (cheap) but skip the
  // machine-lighting work — nothing is visible to animate
  if (document.hidden || !$("#tab-visualizer")?.classList.contains("active")) {
    const raw0 = ev.model || "";
    const fname0 = raw0.split("/").pop();
    const mid0 = flowFileToId[fname0] || fname0;
    flowModelLastSeen[mid0] = Date.now();
    return;
  }
  const raw = ev.model || "";
  const fname = raw.split("/").pop();
  // upstream gives GGUF path, but also handle bare model_id
  const servers = flowModelMachines[fname] || flowModelIdServers[fname] || flowModelIdServers[raw] || flowModelMachines[raw];
  if (!servers) console.warn("[flow] upstream no servers for", fname, "known:", Object.keys(flowModelMachines).slice(0, 3));
  if (servers && servers.length) {
    // resolve to a model_id (so per-model serving state is keyed consistently
    // even when several models co-serve). fall back to the raw id/fname.
    const mid = flowFileToId[fname] || (flowModelIdServers[raw] ? raw : null) || (flowModelIdServers[fname] ? fname : null) || fname;
    const now = Date.now();
    flowModelLastSeen[mid] = now;
    // variant keys ("id--label") may show up in the loaded list as just "id"
    // (llama-swap registers under the base id), so record both so the node
    // lights regardless of which form /api/swap/loaded returned.
    const dash = mid.indexOf("--");
    if (dash > 0) flowModelLastSeen[mid.slice(0, dash)] = now;
    flowLastModel = mid;          // most-recently-streaming model (sched attribution hint)
    flowLastServers = servers;    // kept for sched fallback + legacy references
    // light up the serving machine(s) in real time (fallback when sched debug is quiet)
    // SOURCE-TAGGED with this model so other models' edges can't claim it
    for (const srv of servers) {
      flowGpuActivity[srv] = flowGpuActivity[srv] || {};
      const m = flowMachines.find(x => x.name === srv);
      const nGpus = ((m && m.detected) || {}).gpu_names || [];
      const alloc = (flowModelExposes[mid] || flowModelExposes[fname] || {})[srv];
      const gi = alloc && alloc.gpus && alloc.gpus.length
        ? parseInt((alloc.gpus[0].match(/\d+/) || [0])[0], 10)
        : (nGpus.length ? 0 : "RAM");
      flowMarkActivity(srv, gi, mid);
    }
    // HOST-ONLY models: upstream logs barely stream, so simulate the
    // machine-style continuous GPU light — keep the host's GPU/RAM cells
    // marked for 2.5s after each token, refreshed on a fast interval
    // while the model is the most recent streamer
    const isHostOnly = servers.length === 1 && flowMachines.find(x => x.name === servers[0] && x.role === "master");
    if (isHostOnly && !window._hostSimTimer) {
      // capture loop-scoped values — the interval closure outlives them
      const hSrv = servers[0];
      const hMid = mid;
      const hAlloc = (flowModelExposes[mid] || flowModelExposes[fname] || {})[hSrv];
      const hM = flowMachines.find(x => x.name === hSrv);
      const hNGpus = ((hM && hM.detected) || {}).gpu_names || [];
      const hGi = hAlloc && hAlloc.gpus && hAlloc.gpus.length
        ? parseInt((hAlloc.gpus[0].match(/\d+/) || [0])[0], 10)
        : (hNGpus.length ? 0 : "RAM");
      window._hostSimTimer = setInterval(() => {
        const seen = Date.now() - (flowModelLastSeen[hMid] || 0);
        if (seen > 3000 || !$("#tab-visualizer")?.classList.contains("active")) {
          clearInterval(window._hostSimTimer);
          window._hostSimTimer = null;
          return;
        }
        flowMarkActivity(hSrv, hGi, hMid);
        requestFlowRender();
      }, 400);
    }
    requestFlowRender();
  }
}

/* vLLM serving pulse (engine=vllm): the upstream ws forwards the 10s
   engine-stats line with the model bound from the stream's "<id>" tag.
   Same lights-up-the-servers behavior as a llama.cpp token chunk, plus
   a stats readout (throughput / KV / cache) on the model card. When the
   stats line carries no model (no tag seen yet), fall back to the
   single loaded model — the common single-model case. */
function handleVllmStatsEvent(ev) {
  let mid = ev.model || null;
  if (!mid && flowLoadedModels.length === 1) mid = flowLoadedModels[0];
  if (!mid) return;
  const now = Date.now();
  flowModelLastSeen[mid] = now;
  const dash = mid.indexOf("--");
  if (dash > 0) flowModelLastSeen[mid.slice(0, dash)] = now;
  flowLastModel = mid;
  const servers = flowModelIdServers[mid]
    || flowModelIdServers[mid.slice(0, dash)] || flowLastServers;
  if (!servers || !servers.length) return;
  flowLastServers = servers;
  // stats line alone tells us nothing about WHICH gpu computed — light
  // the model's full server set (the /metrics poller below refines this
  // with real per-model token deltas at scrape granularity)
  for (const srv of servers) {
    const m = flowMachines.find(x => x.name === srv);
    const nGpus = ((m && m.detected) || {}).gpu_names || [];
    const gi = nGpus.length ? 0 : "RAM";
    flowMarkActivity(srv, gi, mid);
  }
  // live readout in the header badge
  updateVllmStatsReadout(ev, mid);
  requestFlowRender();
}

function updateVllmStatsReadout(ev, mid) {
  const badge = $("#flow-status");
  if (!badge) return;
  const tps = `${ev.gen_tps.toFixed(1)} tok/s`;
  badge.textContent = `${mid} · gen ${tps} · prompt ${ev.prompt_tps.toFixed(1)} · KV ${ev.kv_pct.toFixed(0)}% · cache ${ev.cache_hit_pct.toFixed(0)}%`;
}

/* /api/flow/vllm-metrics poller (engine=vllm): per-model counter deltas
   from the upstream Prometheus scrape. Feeds flowMarkActivity with the
   same (machine, gpu/RAM) targets the ws stats pulse uses — the metrics
   scrape adds REAL tokens/s per model, so the animation reflects actual
   work between the 10s ws stats lines. Errors stay silent (vLLM may
   briefly not answer /metrics during weight load). */
let flowVllmMetricsTick = 0;
async function pollVllmFlowMetrics() {
  let data;
  try { data = await api("GET", "/api/flow/vllm-metrics"); } catch (_) { return; }
  const models = data.models || {};
  for (const [mid, m] of Object.entries(models)) {
    if (!m) continue;
    const servers = flowModelIdServers[mid] || [];
    if (!servers.length) continue;
    const active = (m.gen_tps > 0 || m.prompt_tps > 0
      || (m.num_requests_running || 0) > 0);
    if (!active) continue;
    const now = Date.now();
    flowModelLastSeen[mid] = now;
    flowLastModel = mid;
    flowLastServers = servers;
    for (const srv of servers) {
      const mm = flowMachines.find(x => x.name === srv);
      const nGpus = ((mm && mm.detected) || {}).gpu_names || [];
      flowMarkActivity(srv, nGpus.length ? 0 : "RAM", mid);
    }
    updateVllmStatsReadout({
      prompt_tps: m.prompt_tps || 0, gen_tps: m.gen_tps || 0,
      kv_pct: m.kv_cache_usage_perc || 0,
      cache_hit_pct: (m.prefix_cache_queries > 0
        ? 100 * m.prefix_cache_hits / m.prefix_cache_queries : 0),
    }, mid);
  }
  requestFlowRender();
}

function handleSchedEvent(ev) {
  // hidden/inactive tab: drop the event entirely — sched floods are the
  // heaviest stream (thousands of lines per load) and nothing is visible
  if (document.hidden || !$("#tab-visualizer")?.classList.contains("active")) return;
  // keep a rolling buffer of recent per-tensor backend assignments
  flowSchedEvents.push(ev);
  if (flowSchedEvents.length > 200) flowSchedEvents.shift();
  // sched lines arrive at thousands/sec during loads — cap processing so
  // the render loop can't be starved (memory leak fuel: every event bumps
  // activity timestamps, forcing the SVG diff to always differ)
  const t = Date.now();
  if (t - (handleSchedEvent._last || 0) < 250) return;
  handleSchedEvent._last = t;
  const backend = (ev.backend || "").toLowerCase();
  const now = Date.now();
  // Sched lines carry NO model id, so attribution decides which model's
  // edges light up. Policy:
  //   1. A model still streaming (upstream chunk < SERVING_WINDOW old)
  //      owns the sched — tokens flowing = graph is computing.
  //   2. Nothing streaming: single loaded model -> it owns the sched
  //      (LOAD-time events must light edges too).
  //   3. Multiple loaded, none streaming: most-recently-loaded model.
  const freshest = flowFreshestModelId();
  let owner = freshest;
  let targets;
  if (owner) {
    targets = flowModelIdServers[owner] || flowLastServers;
  } else if (flowLoadedModels.length === 1) {
    owner = flowLoadedModels[0];
    targets = flowModelIdServers[owner] || flowLastServers;
  } else if (flowLoadedModels.length > 1) {
    owner = flowNewestLoadedId();
    targets = flowModelIdServers[owner] || flowLastServers;
  } else {
    targets = flowLastServers && flowLastServers.length ? flowLastServers : null;
  }
  if (!targets) return;
  // Exact RPC<N> -> {machine, resource} map from the backend (the same
  // numbering the generated scripts count devices with). Fetched on
  // visualizer init; falls back to the old modulo guess only if absent.
  if (!flowRpcMaps) return;
  const mapFor = flowRpcMaps[owner] || (owner && owner.indexOf("--") > 0
    ? (flowRpcMaps[owner.split("--")[0]] || {}).maps
    : null) || null;
  const variantLabel = owner && owner.indexOf("--") > 0 ? owner.split("--")[1] : "";
  const rpcMap = mapFor && mapFor.maps
    ? (mapFor.maps[variantLabel] || mapFor.maps[""] || {})
    : (mapFor || {});
  if (backend.startsWith("rpc")) {
    // Exact resolution: RPC<N> -> machine + resource (GPU index or RAM).
    // No modulo guessing — an unknown id simply doesn't light anything.
    const entry = rpcMap[backend.toUpperCase()];
    if (entry && entry.machine) {
      const gi = entry.resource === "RAM"
        ? "RAM"
        : parseInt((entry.resource.match(/\d+/) || [0])[0], 10);
      flowMarkActivity(entry.machine, gi, owner);
      requestFlowRender();
      return;
    }
    // no map entry for this token: don't guess, don't light
    return;
  }
  if (backend.startsWith("cuda") || backend.startsWith("rocm") || backend.startsWith("metal")) {
    const gi = parseInt((backend.match(/\d+/) || [0])[0], 10);
    // CUDA/ROCm/Metal lives on the host (first server) of the model
    const host = targets[0];
    const m = flowMachines.find(x => x.name === host);
    const nGpus = ((m && m.detected) || {}).gpu_names || [];
    if (nGpus.length) {
      flowMarkActivity(host, Math.min(gi, nGpus.length - 1), owner);
    }
  } else if (backend === "cpu") {
    // A CPU-backend node runs only where its graph split lives. With a
    // known owner's rpc_map, RAM devices it names own the CPU work;
    // without one, attribute to the host only (never broadcast to all).
    const ramMachines = new Set(
      Object.values(rpcMap || {})
        .filter(e => e && e.resource === "RAM")
        .map(e => e.machine));
    if (ramMachines.size) {
      for (const name of ramMachines) flowMarkActivity(name, "RAM", owner);
    } else if (targets.length) {
      flowMarkActivity(targets[0], "RAM", owner);
    }
  }
  requestFlowRender();
}

// Coalesce render requests: WS event floods (sched/upstream) each used to
// call renderFlowDiagram() directly — thousands of full innerHTML rebuilds
// per load. RAF batches them to at most one draw per frame.
let flowRenderPending = false;
let flowLastSvg = null; // last rendered SVG string (skip identical rebuilds)
function requestFlowRender() {
  if (!window._flowWavesStarted) {
    window._flowWavesStarted = true;
    startFlowWaves();
  }
  if (flowRenderPending) return;
  flowRenderPending = true;
  requestAnimationFrame(() => {
    flowRenderPending = false;
    renderFlowDiagram();
  });
}
// waves start lazily on first render (see requestFlowRender)

/* Packet particles: glowing dots riding active edges. RAF loop walks
   each active <line> with SVG geometry + spawns SVG circles that fade. */
let flowPacketRAF = 0;
/* Unified comet simulation: bright dots flowing along the chain rails.
   Per leg: 4 comets at varying speeds, staggered, positioned via getPointAtLength on the
   leg's measurer polyline — always exactly on the path. */
let flowCometsRAF = 0;
function startFlowComets() {
  cancelAnimationFrame(flowCometsRAF);
  const tick = (t) => {
    flowCometsRAF = requestAnimationFrame(tick);
    if (document.hidden || !$("#tab-visualizer")?.classList.contains("active")) return;
    const liveVp = document.getElementById("flow-viewport");
    if (!liveVp) return;
    // every leg: independent comets, staggered, simple and stable.
    // no cross-leg chaining — each rail owns its own dots.
    // 4 comets per leg at varying speeds (read from data-speed so each dot
    // keeps its own pace across re-renders).
    const tNow = t / 1000;
    for (const cm of [...liveVp.querySelectorAll("line.flow-chain-comet")]) {
      const pathEl = [...liveVp.querySelectorAll(".flow-chain-path")].find(
        p => p.getAttribute("data-flow-id") === cm.dataset.legId);
      if (!pathEl || !pathEl.getTotalLength) continue;
      const len = pathEl.getTotalLength();
      if (!len || !isFinite(len) || len <= 0) continue;
      const slot = parseFloat(cm.dataset.slot || "0");
      const speed = parseFloat(cm.dataset.speed || "0.35");
      const prog = ((tNow * speed + slot * 0.25) % 1);
      const head = prog * len;
      const tail = Math.max(0, head - 10);
      const a = pathEl.getPointAtLength(tail);
      const b = pathEl.getPointAtLength(head);
      cm.setAttribute("x1", a.x); cm.setAttribute("y1", a.y);
      cm.setAttribute("x2", b.x); cm.setAttribute("y2", b.y);
      const edgeFade = Math.min(1, Math.min(prog, 1 - prog) * 5);
      cm.setAttribute("opacity", (0.95 * edgeFade).toFixed(2));
    }
  };
  flowCometsRAF = requestAnimationFrame(tick);
}

/* PS3-XMB-style flowing ribbon waves: layered translucent sine bands
   drifting slowly behind the graph. Interval-driven; pauses when tab hidden.
   Shared renderer: the visualizer's #flow-waves canvas AND the global
   #bg-waves canvas (fixed behind every page) both use it. */
let flowWavesRAF = 0;
let bgWavesRAF = 0;
function startWaveRenderer(canvasId, alphaScale, requireVisualizerTab) {
  const cv = document.getElementById(canvasId);
  if (!cv || cv._wavesStarted) return cv ? null : null;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  cv._wavesStarted = true;
  let W = 0, H = 0;
  const resize = () => {
    const r = cv.getBoundingClientRect();
    W = cv.width = Math.max(1, Math.round(r.width * devicePixelRatio));
    H = cv.height = Math.max(1, Math.round(r.height * devicePixelRatio));
  };
  resize();
  window.addEventListener("resize", resize);
  if (requireVisualizerTab) window.addEventListener("resize", () => requestFlowRender());
  const rootEl = document.documentElement;
  const readRibbon = () => {
    const rs = getComputedStyle(rootEl);
    return (rs.getPropertyValue("--ribbon").trim()
      || rs.getPropertyValue("--accent").trim() || "#7aa2ff");
  };
  let accent = readRibbon();
  let lastTheme = rootEl.getAttribute("data-theme");
  let waveFrame = 0;
  // ribbon FX: glow (alpha multiplier) + vivid (saturation multiplier),
  // tunable per theme via --ribbon-glow / --ribbon-sat
  const readFx = () => {
    const rs = getComputedStyle(rootEl);
    const num = (v, d) => {
      const f = parseFloat(v);
      return Number.isFinite(f) ? Math.min(2.5, Math.max(0, f)) : d;
    };
    return {
      glow: num(rs.getPropertyValue("--ribbon-glow"), 1),
      sat: num(rs.getPropertyValue("--ribbon-sat"), 1),
    };
  };
  let ribbonFx = readFx();
  const vivid = (hex, sat) => {
    const m = /^#([0-9a-fA-F]{6})$/.exec((hex || "").trim());
    if (!m || sat === 1) return hex;
    let r = parseInt(m[1].substr(0, 2), 16) / 255,
        g = parseInt(m[1].substr(2, 2), 16) / 255,
        b = parseInt(m[1].substr(4, 2), 16) / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    let h = 0, s = 0; const l = (mx + mn) / 2;
    if (mx !== mn) {
      const d = mx - mn;
      s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
      if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h /= 6;
    }
    s = Math.min(1, s * sat);
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    let rr, gg, bb;
    if (s === 0) { rr = gg = bb = l; }
    else {
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      rr = hue2rgb(p, q, h + 1/3); gg = hue2rgb(p, q, h); bb = hue2rgb(p, q, h - 1/3);
    }
    return `rgb(${Math.round(rr*255)},${Math.round(gg*255)},${Math.round(bb*255)})`;
  };
  const start = performance.now();
  // 4 ribbon layers, each an independent phase/speed/opacity — organic
  // overlap like the XMB background
  const bands = [
    { amp: 40, speed: 0.00014, phase: 0.9, width: 1.8, alpha: 0.12, yOff: 0.08 },
    { amp: 52, speed: 0.00011, phase: 2.4, width: 1.4, alpha: 0.10, yOff: 0.18 },
    { amp: 46, speed: 0.00016, phase: 0.0, width: 2.2, alpha: 0.16, yOff: 0.30 },
    { amp: 62, speed: 0.00012, phase: 1.7, width: 1.6, alpha: 0.11, yOff: 0.44 },
    { amp: 62, speed: 0.00020, phase: 3.1, width: 1.2, alpha: 0.08, yOff: 0.58 },
    { amp: 38, speed: 0.00024, phase: 4.6, width: 3.0, alpha: 0.13, yOff: 0.72 },
    { amp: 55, speed: 0.00010, phase: 5.9, width: 1.0, alpha: 0.07, yOff: 0.50 },
    { amp: 50, speed: 0.00013, phase: 4.1, width: 1.6, alpha: 0.11, yOff: 0.84 },
    { amp: 42, speed: 0.00018, phase: 2.8, width: 2.0, alpha: 0.12, yOff: 0.93 },
  ];
  const tick = (t) => {
    if (document.hidden) return;
    if (requireVisualizerTab && !$("#tab-visualizer")?.classList.contains("active")) return;
    // theme switch = new ribbon color: re-read when data-theme changes.
    // Custom-theme color dragging never changes data-theme, so also
    // re-check the computed value every 15th frame (cheap, catches drags).
    const th = rootEl.getAttribute("data-theme");
    if (th !== lastTheme) { lastTheme = th; accent = readRibbon(); ribbonFx = readFx(); }
    else if (++waveFrame % 15 === 0) {
      const rb = readRibbon(); if (rb !== accent) accent = rb;
      const fx = readFx();
      if (fx.glow !== ribbonFx.glow || fx.sat !== ribbonFx.sat) ribbonFx = fx;
    }
    // self-heal size: the tab starts hidden, so the first measure was 0x0
    const r = cv.getBoundingClientRect();
    if (Math.round(r.width * devicePixelRatio) !== W || Math.round(r.height * devicePixelRatio) !== H) resize();
    ctx.clearRect(0, 0, W, H);
    const time = t - start;
    ctx.save();
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const w = W / devicePixelRatio, h = H / devicePixelRatio;
    for (const b of bands) {
      const path = () => {
        ctx.beginPath();
        const yBase = h * b.yOff;
        for (let x = -20; x <= w + 20; x += 12) {
          const y = yBase
            + Math.sin(x * 0.006 + time * b.speed * 2 + b.phase) * b.amp
            + Math.sin(x * 0.0021 - time * b.speed * 1.4 + b.phase * 2) * b.amp * 0.45;
          if (x <= 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      };
      const ribbonColor = vivid(accent, ribbonFx.sat);
      // FX (halo + breathing) ONLY when the theme actually tunes glow/sat —
      // defaults render exactly like the original plain ribbons
      const fxOn = (ribbonFx.glow !== 1 || ribbonFx.sat !== 1);
      // same glow fade in/out as the llama-swap core (flow-core-pulse):
      // each band breathes on the same 2.4s rhythm, staggered by phase
      const breathe = (!fxOn || ribbonFx.glow === 0)
        ? 1 : (0.7 + 0.3 * Math.sin(time * (2 * Math.PI / 2400) + b.phase));
      // glow 0 = halo off but core stays full plain brightness
      const glowBoost = ribbonFx.glow === 0 ? 1 : (0.75 + 0.25 * ribbonFx.glow);
      const baseAlpha = fxOn
        ? Math.min(1, b.alpha * alphaScale * glowBoost * breathe)
        : b.alpha * alphaScale;
      // pass 1 — glow fade: nested soft shells (wide→narrow, faint→less
      // faint) so the halo FADES out instead of ending in a hard edge.
      // No shadowBlur (it tanks the frame rate); layered round strokes
      // approximate the same blurred falloff, like the visualizer halos.
      if (fxOn && ribbonFx.glow > 0.02) {
        const shells = [[7, 0.10], [4.5, 0.20], [2.5, 0.32]];
        for (const [wm, am] of shells) {
          path();
          ctx.strokeStyle = ribbonColor;
          ctx.shadowBlur = 0;
          ctx.globalAlpha = Math.min(0.5, baseAlpha * am * ribbonFx.glow * breathe);
          ctx.lineWidth = b.width * wm;
          ctx.lineCap = "round";
          ctx.stroke();
        }
      }
      // pass 2 — crisp core line
      path();
      ctx.shadowBlur = 0;
      ctx.lineCap = "round";
      ctx.strokeStyle = ribbonColor;
      ctx.globalAlpha = baseAlpha;
      ctx.lineWidth = b.width;
      ctx.stroke();
    }
    ctx.restore();
  };
  // 60fps: matches the comet RAF and keeps the breathing perfectly smooth
  return setInterval(() => tick(performance.now()), 16);
}
function startFlowWaves() {
  if (!flowWavesRAF) flowWavesRAF = startWaveRenderer("flow-waves", 1, true);
}
function startAmbientWaves() {
  // global backdrop: subtler than the visualizer's so text stays readable
  if (!bgWavesRAF) bgWavesRAF = startWaveRenderer("bg-waves", 0.55, false);
}
/* master ribbons toggle — independent of theme: kills the paint loops
   entirely when off (not just hidden) */
function ribbonsOn() {
  return localStorage.getItem("ui-ribbons") !== "0";
}
function applyRibbons() {
  const on = ribbonsOn();
  const chip = $("#set-ribbons-chip");
  if (chip) chip.setAttribute("aria-pressed", String(on));
  const cb = $("#set-ribbons");
  if (cb) cb.checked = on;
  const cv = document.getElementById("bg-waves");
  if (!on) {
    clearInterval(bgWavesRAF);
    bgWavesRAF = 0;
    if (cv) { cv._wavesStarted = false; cv.style.display = "none"; }
    return;
  }
  if (cv) cv.style.display = "";
  startAmbientWaves();
}

/* Morph the live SVG toward the freshly rendered string instead of
   innerHTML-swapping: reuses existing nodes (so CSS animations keep
   running and nodes never blink), updates attributes in place, adds
   new / removes gone elements. Keyed by data-flow-id for nodes, by
   index for edges/packets (edges have data-packet, GPU rings by path). */
function flowMorphSvg(canvas, svgStr) {
  const tmp = document.createElement("div");
  tmp.innerHTML = svgStr;
  const freshSvg = tmp.firstElementChild;
  if (!freshSvg) return;
  // "Waiting for traffic…" placeholder: hide once the graph exists
  const empty = canvas.querySelector(".flow-empty");
  if (empty) empty.style.display = "none";
  let live = canvas.querySelector("svg.flow-svg");
  if (!live) { canvas.appendChild(freshSvg); return; }
  // defs: replace wholesale (cheap, no animation state)
  const liveDefs = live.querySelector("defs");
  const freshDefs = freshSvg.querySelector("defs");
  if (liveDefs && freshDefs && liveDefs.innerHTML !== freshDefs.innerHTML) liveDefs.innerHTML = freshDefs.innerHTML;
  const liveVp = live.querySelector("#flow-viewport");
  const freshVp = freshSvg.querySelector("#flow-viewport");
  if (!liveVp || !freshVp) return;
  // keyed diff of the viewport's children (nodes, edges, packets)
  const freshKids = [...freshVp.children];
  const liveById = new Map();
  for (const k of [...liveVp.children]) {
    const key = k.getAttribute("data-flow-id") || k.getAttribute("data-packet")
      || (k.tagName === "line" ? `edge:${k.getAttribute("x1")},${k.getAttribute("y1")},${k.getAttribute("x2")},${k.getAttribute("y2")}` : null)
      || `idx:${[...liveVp.children].indexOf(k)}`;
    if (!liveById.has(key)) liveById.set(key, k);
  }
  const used = new Set();
  for (const fk of freshKids) {
    const key = fk.getAttribute("data-flow-id") || fk.getAttribute("data-packet")
      || (fk.tagName === "line" ? `edge:${fk.getAttribute("x1")},${fk.getAttribute("y1")},${fk.getAttribute("x2")},${fk.getAttribute("y2")}` : `idx:${freshKids.indexOf(fk)}`);
    let target = liveById.get(key);
    if (!target || target.tagName !== fk.tagName) {
      // for stateless elements reuse an orphan of the SAME kind only —
      // never let a comet steal a flow-edge (or vice versa), that halves
      // lines/glows when keys churn.
      if (fk.classList.contains("flow-chain-comet")) {
        target = [...liveVp.querySelectorAll("line.flow-chain-comet")].find(l => !l.dataset.used);
      } else if (fk.classList.contains("flow-chain-path")) {
        target = [...liveVp.querySelectorAll(".flow-chain-path")].find(l => !l.dataset.used);
      } else if (fk.tagName === "line") {
        target = [...liveVp.querySelectorAll("line.flow-edge")].find(l => !l.dataset.used);
      } else if (fk.classList.contains("flow-packet")) {
        target = [...liveVp.querySelectorAll("circle.flow-packet")].find(l => !l.dataset.used);
      }
    }
    if (target && target.tagName === fk.tagName) {
      target.dataset.used = "1";
      // copy attributes except style (drag persists via attr anyway) and
      // data-prog (comet live progress must survive re-renders).
      // For live comets also skip geometry (x1/y1/x2/y2/opacity): the RAF
      // positions them every frame, and re-copying the fresh 0,0/opacity-0
      // values on every render makes them flicker/disappear.
      const isLiveComet = target.classList.contains("flow-chain-comet");
      for (const a of fk.attributes) {
        if (a.name === "style" || a.name === "data-prog") continue;
        if (isLiveComet && (a.name === "x1" || a.name === "y1" || a.name === "x2" || a.name === "y2" || a.name === "opacity")) continue;
        if (target.getAttribute(a.name) !== a.value) target.setAttribute(a.name, a.value);
      }
      // carry progress over from the fresh render's perspective
      if (fk.dataset.prog !== undefined && target.dataset.prog === undefined) target.dataset.prog = fk.dataset.prog;
      // inner content (texts, icons)
      if (target.innerHTML !== fk.innerHTML) target.innerHTML = fk.innerHTML;
      if (target.parentNode !== liveVp) liveVp.appendChild(target);
    } else {
      liveVp.appendChild(fk);   // brand new node — entrance animation plays
    }
    if (target) target.dataset.used = "1";
  }
  // remove elements that disappeared — including comets whose leg
  // measurer is gone (otherwise they linger at stale/0,0 coords)
  for (const [key, el] of liveById) {
    if (el.classList.contains("flow-packet")) continue;
    if (el.classList.contains("flow-chain-comet")) {
      const legStillThere = liveVp.querySelector('[data-flow-id="' + (el.dataset.legId || "").replace(/"/g, '\\"') + '"]');
      if (!legStillThere) { el.remove(); continue; }
    }
    if (!el.dataset.used && el.parentNode === liveVp) el.remove();
    delete el.dataset.used;
  }
}

function renderFlowDiagram() {
  const canvas = $("#flow-canvas");
  if (!canvas) return;
  // DEBUG: force all legs/edges active for testing comets without traffic
  const forceAllActive = localStorage.getItem("flow-debug-all-active") === "1";
  // (visualizer ribbon loop retired: the transparent wrap shows the single
  // global #bg-waves loop instead — see style.css)
  // background tabs / hidden visualizer: don't draw at all — a flood of
  // WS events (sched lines fire thousands of times during a load) used
  // to trigger full innerHTML rebuilds per event in every open tab,
  // which is what piled up CPU/RAM on the peers' browsers
  if (document.hidden) return;
  if (!$("#tab-visualizer")?.classList.contains("active")) return;
  if (!flowMachines.length) {
    canvas.innerHTML = `<div class="flow-empty"><i class="fa-solid fa-diagram-project"></i><span>No machines configured</span></div>`;
    return;
  }
  const now = Date.now();
  const W = canvas.clientWidth || 900;
  const H0 = 520;
  // left-to-right: llama-swap (left) -> loaded models (center) -> machines (right)
  const nCenter = flowLoadedModels.length || 1;
  const nRight = flowMachines.length;
  const longestLoaded = Math.max(12, ...flowLoadedModels.map(m => m.length));
  const boxWCenter = Math.min(420, Math.max(110, longestLoaded * 6.8 + 28));
  const boxW = nRight > 6 ? 100 : 120;
  const leftX = 90;
  const rightX = W - 90;
  // dynamic height so all nodes fit without clipping — like n8n's fitView with padding
  const maxBoxH = Math.max(...flowMachines.map(m =>
    46 + ((m.detected || {}).gpu_names || []).length * 26 + (((m.detected || {}).ram_total_gb > 0) ? 26 : 0)));
  const needH = Math.max(nCenter * 44 + 80, nRight * (maxBoxH + 8) + 80, H0);
  const H = needH;
  const cx = W / 2, cy = H / 2;
  const gapCenter = nCenter > 1 ? 44 : 0;
  const gapRight = nRight > 1 ? maxBoxH + 8 : 0;
  let svg = `<svg viewBox="0 0 ${W} ${H}" class="flow-svg">
    <defs>
      <clipPath id="flow-icon-clip"><rect x="0" y="0" width="18" height="18" rx="4"/></clipPath>
      <linearGradient id="flow-edge-grad" x1="0%" y1="0%" x2="100%" y2="0%" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.35"/>
        <stop offset="50%" stop-color="var(--accent)" stop-opacity="1"/>
        <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.35"/>
      </linearGradient>
      <radialGradient id="flow-core-grad">
        <stop offset="0%" stop-color="color-mix(in srgb, var(--accent) 35%, var(--raised))"/>
        <stop offset="100%" stop-color="var(--raised)"/>
      </linearGradient>
      <filter id="flow-glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <g id="flow-viewport" transform="translate(${flowViewport.x},${flowViewport.y}) scale(${flowViewport.scale})">`;
  // positions: left (llama-swap single), center (loaded models), right (machines)
  const centerPos = [];
  for (let i = 0; i < nCenter; i++) {
    const id = flowLoadedModels[i] || 'idle';
    const key = `model:${id}`;
    if (flowLayout[key]) { centerPos.push({ x: flowLayout[key].x, y: flowLayout[key].y }); }
    else { const y = cy - (nCenter - 1) * gapCenter / 2 + i * gapCenter; centerPos.push({ x: cx, y }); }
  }
  const rightPos = [];
  for (let i = 0; i < nRight; i++) {
    const key = `machine:${flowMachines[i].name}`;
    if (flowLayout[key]) rightPos.push({ x: flowLayout[key].x, y: flowLayout[key].y });
    else { const y = cy - (nRight - 1) * gapRight / 2 + i * gapRight; rightPos.push({ x: rightX, y }); }
  }
  // left single may also be dragged
  let leftPos = { x: leftX, y: cy };
  if (flowLayout['swap']) leftPos = { x: flowLayout['swap'].x, y: flowLayout['swap'].y };
  // edges: drawn for any LOADED model whose servers have recent GPU
  // activity — during load, sched events light GPUs before any upstream
  // chunk streams, so gating on "streaming" alone would leave the GPU
  // lit but no edge (the original bug). Per-model so several co-served
  // models light independently.
  const serving = (label) => {
    if (!label) return false;
    if (forceAllActive) return true;   // debug: all lines active
    const ts = flowModelLastSeen[label] || 0;
    return (now - ts) < FLOW_SERVING_WINDOW;
  };
  // a model "has activity" if it's loaded AND one of ITS OWN servers'
  // GPU/RAM slots is recently lit AND that activity belongs to this
  // model (source-tagged). Unattributed entries (s=null, e.g. legacy
  // or unknown-model) still count for any loaded model on that server,
  // but entries owned by ANOTHER model never do — this is what stops
  // model B's edges lighting up while only model A is streaming.
  const hasActivity = (label) => {
    if (!label) return false;
    if (forceAllActive) return true;   // debug: glow all machines/nodes
    const servers = flowModelIdServers[label] || flowModelMachines[label] || [];
    return servers.some(srv => {
      const a = flowGpuActivity[srv] || {};
      return Object.entries(a).some(([k, v]) => {
        if (now - flowActTs(v) >= 1500) return false;
        const src = flowActSrc(v);
        return !src || flowSameModel(src, label);
      });
    });
  };
  // swap -> model rail: always while loaded
  const chainPolylines = [];  // per loaded model: {label, trunk, branches, modelRight}
  for (let i = 0; i < nCenter; i++) {
    const label = flowLoadedModels[i];
    if (!label) continue;
    const y = centerPos[i].y;
    svg += `<line class="flow-edge active" data-flow-id="edge-swap:${escHtml(label)}" x1="${leftPos.x + 34}" y1="${leftPos.y}" x2="${centerPos[i].x - boxWCenter/2}" y2="${y}"/>`;
  }
  // model -> machine rails: only while that machine has per-model activity.
  // The chain comet always exists while the model is loaded (minimum chain:
  // llama-swap -> model); active machine hops extend the chain live.
  for (let ci = 0; ci < nCenter; ci++) {
    const label = flowLoadedModels[ci];
    if (!label) continue;
    const cp = centerPos[ci];
    // DEBUG all-active = preview as if every GPU were streaming: draw a rail
    // from this model to EVERY machine, not just its assigned servers.
    // Normal mode keeps per-model servers so model B never lights model A's
    // machines.
    const servers = forceAllActive
      ? flowMachines.map(m => m.name)
      : (flowModelIdServers[label] || flowModelMachines[label] || []);
    // STAR topology: trunk (swap->model-left) + one branch per machine
    // (model-right->machine). Never daisy-chain machine endpoints together —
    // that built bogus vertical rails down the machine column with comets
    // flying machine-to-machine instead of along the visible edge lines.
    const trunk = [[leftPos.x + 34, leftPos.y], [cp.x - boxWCenter/2, cp.y]];
    const branches = [];  // [srv, [x, y]] — only fresh machines get comets
    for (const srv of servers) {
      const ri = flowMachines.findIndex(m => m.name === srv);
      if (ri === -1) continue;
      const rp = rightPos[ri];
      const active = flowGpuActivity[flowMachines[ri].name] || {};
      const fresh = forceAllActive || Object.entries(active).some(([k, v]) => {
        if (now - flowActTs(v) >= 1500) return false;
        const src = flowActSrc(v);
        return !src || flowSameModel(src, label);
      });
      console.debug("[flow-debug] machine hop:", label, srv, "fresh:", fresh);
      svg += `<line class="flow-edge${fresh ? " active" : ""}" data-flow-id="edge-machine:${escHtml(label)}:${escHtml(srv)}" data-packet="${escHtml(label)}" x1="${cp.x + boxWCenter/2}" y1="${cp.y}" x2="${rp.x - boxW/2}" y2="${rp.y}"/>`;
      if (forceAllActive || fresh) branches.push([srv, [rp.x - boxW/2, rp.y]]);
    }
    chainPolylines.push({label, trunk, branches, modelRight: [cp.x + boxWCenter/2, cp.y]});
  }
  // comets per leg: leg "trunk" = swap->model (always, while loaded);
  // one leg per fresh machine, each riding EXACTLY its visible edge line.
  // Leg ids are stable per machine (not positional) so comets never jump
  // rails when machines appear/disappear.
  for (const ch of chainPolylines) {
    const legs = [["trunk", ch.trunk[0], ch.trunk[1]]];
    for (const [srv, pt] of ch.branches) {
      legs.push([`m:${srv}`, ch.modelRight, pt]);
    }
    legs.forEach(([legKey, a, b]) => {
      const legId = `leg:${escHtml(ch.label)}:${escHtml(legKey)}`;
      svg += `<polyline class="flow-chain-path" points="${Math.round(a[0])},${Math.round(a[1])} ${Math.round(b[0])},${Math.round(b[1])}" data-flow-id="${legId}" fill="none"/>`;
      // 4 comets per leg at varying speeds so the flow feels organic
      const speeds = [0.42, 0.30, 0.36, 0.24];
      for (let k = 0; k < 4; k++) {
        svg += `<line class="flow-chain-comet" data-flow-id="comet:${legId}:${k}" data-slot="${k}" data-speed="${speeds[k]}" data-leg-id="${legId}" opacity="0" x1="0" y1="0" x2="0" y2="0"/>`;
      }
    });
  }
  // left: llama-swap (single)
  const coreActive = flowLoadedModels.some(l => hasActivity(l));
  svg += `<g class="flow-node flow-center ${coreActive ? "active" : ""}" data-flow-id="swap" transform="translate(${leftPos.x},${leftPos.y})">
    <circle r="34" class="flow-core-halo ${coreActive ? "active" : ""}"/>
    <circle r="46" class="flow-core-halo ${coreActive ? "active" : ""}" style="animation-delay:0.8s"/>
    <circle r="34" class="flow-node-ring" fill="url(#flow-core-grad)"/>
    <circle r="24" class="flow-core-inner"/>
    <text y="-2" text-anchor="middle" class="flow-node-label">llama</text>
    <text y="10" text-anchor="middle" class="flow-node-label">-swap</text>
  </g>`;
  // center: loaded models — hidden when idle (position remembered in flowLayout)
  for (let i = 0; i < nCenter; i++) {
    const label = flowLoadedModels[i];
    if (!label) continue;
    const isIdle = false;
    if (!isIdle) resolveModelIcon(label);
    const iconUrl = !isIdle ? (modelIconCache[label] || "") : "";
    const p = centerPos[i];
    const active = !isIdle && (serving(label) || hasActivity(label));
    svg += `<g class="flow-node ${active ? "active" : "loaded"}" data-flow-id="model:${escHtml(label)}" transform="translate(${p.x},${p.y})">
      <rect x="${-boxWCenter/2}" y="-20" width="${boxWCenter}" height="40" rx="10" class="flow-box-halo ${active ? "active" : ""}"/>
      <rect x="${-boxWCenter/2}" y="-20" width="${boxWCenter}" height="40" rx="10" class="flow-node-box"/>
      ${iconUrl ? `<image href="${escHtml(iconUrl)}" x="${-boxWCenter/2 + 6}" y="-11" width="18" height="18" clip-path="url(#flow-icon-clip)"/>` : ""}
      <text y="5" text-anchor="middle" class="flow-node-name" dx="${iconUrl ? "10" : "0"}">${escHtml(label)}</text>
    </g>`;
  }
  // machine nodes (right side) with GPU sub-nodes INSIDE each node box
  for (let i = 0; i < nRight; i++) {
    const m = flowMachines[i];
    const p = rightPos[i];
    const gpus = (m.detected || {}).gpu_names || [];
    const active = flowGpuActivity[m.name] || {};
    const nodeActive = forceAllActive || Object.values(active).some(v => now - flowActTs(v) < 1500);
    // box height grows with GPU count + RAM row
    const hasRam = (m.detected || {}).ram_total_gb > 0;
    const boxH = 46 + gpus.length * 26 + (hasRam ? 26 : 0);
    const topY = p.y - boxH / 2;
    svg += `<g class="flow-node ${nodeActive ? "active" : ""}" data-flow-id="machine:${escHtml(m.name)}" transform="translate(${p.x},${p.y})">
      <rect x="${-boxW / 2}" y="${-boxH / 2}" width="${boxW}" height="${boxH}" rx="12" class="flow-box-halo ${nodeActive ? "active" : ""}"/>
      <rect x="${-boxW / 2}" y="${-boxH / 2}" width="${boxW}" height="${boxH}" rx="12" class="flow-node-box"/>
      <text y="${-boxH / 2 + 16}" text-anchor="middle" class="flow-node-name">${escHtml(m.name)}</text>
      <text y="${-boxH / 2 + 30}" text-anchor="middle" class="flow-node-sub">${escHtml(m.ip)}</text>`;
    // GPU sub-nodes inside the box (no connecting lines — circles light up)
    gpus.forEach((g, gi) => {
      const gy = -boxH / 2 + 44 + gi * 26;
      const gActive = forceAllActive || now - flowActTs(active[gi]) < 1500;
      svg += `<g class="flow-gpu ${gActive ? "active" : ""}" transform="translate(${-boxW / 2 + 20},${gy})">
        <circle r="9" class="flow-gpu-ring"/>
        <text y="3" text-anchor="middle" class="flow-gpu-label">${gi}</text>
      </g>`;
      svg += `<text x="${-boxW / 2 + 34}" y="${gy + 3}" class="flow-gpu-name">${escHtml(g.split(" ").slice(-2).join(" "))}</text>`;
    });
    // RAM sub-node inside the box
    if (hasRam) {
      const ry = -boxH / 2 + 44 + gpus.length * 26;
      const ramActive = forceAllActive || now - flowActTs(active["RAM"]) < 1500;
      svg += `<g class="flow-gpu flow-ram ${ramActive ? "active" : ""}" transform="translate(${-boxW / 2 + 20},${ry})">
        <circle r="9" class="flow-gpu-ring"/>
        <text y="3" text-anchor="middle" class="flow-gpu-label">R</text>
      </g>`;
      svg += `<text x="${-boxW / 2 + 34}" y="${ry + 3}" class="flow-gpu-name">RAM ${(m.detected || {}).ram_total_gb}GB</text>`;
    }
    svg += `</g>`;
  }
  svg += `</g></svg>`;
  // skip identical rebuilds: the 500ms poll + RAF coalescing often produce
  // byte-identical SVGs; reassigning innerHTML destroys and recreates every
  // node (plus rebinds drag listeners + restarts CSS transitions), which is
  // the expensive part
  flowLastSvg = svg;
  window.__lastSvg = svg;   // debug hook
  window.__flowBuild = "star-3-4comets";  // bump on every visualizer fix; verifies browser runs latest
  if (!window.__flowBuildLogged) { window.__flowBuildLogged = true; console.info("[flow] build", window.__flowBuild); }
  flowMorphSvg(canvas, svg);
  // reconcile: every chain rail must own exactly 4 live comets. If the morph
  // dropped one (stale keys from an older build, orphan-steal, render race),
  // re-create it so no line is ever missing its comet set.
  try {
    const liveVp = canvas.querySelector("#flow-viewport");
    if (liveVp) {
      const speeds = ["0.42", "0.30", "0.36", "0.24"];
      for (const p of liveVp.querySelectorAll(".flow-chain-path")) {
        const legId = p.getAttribute("data-flow-id");
        if (!legId) continue;
        const have = liveVp.querySelectorAll(`line.flow-chain-comet[data-leg-id="${CSS.escape(legId)}"]`).length;
        for (let k = have; k < 4; k++) {
          const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
          ln.setAttribute("class", "flow-chain-comet");
          ln.setAttribute("data-flow-id", `comet:${legId}:${k}`);
          ln.setAttribute("data-slot", String(k));
          ln.setAttribute("data-speed", speeds[k % speeds.length]);
          ln.setAttribute("data-leg-id", legId);
          ln.setAttribute("opacity", "0");
          ln.setAttribute("x1", "0"); ln.setAttribute("y1", "0");
          ln.setAttribute("x2", "0"); ln.setAttribute("y2", "0");
          liveVp.appendChild(ln);
        }
        if (have > 4) console.debug("[flow] leg has extra comets:", legId, have);
      }
    }
  } catch (e) { console.debug("[flow] comet reconcile failed:", e); }
  startFlowComets();
    // draggable — simple persist
  try {
    const vp2=document.getElementById('flow-viewport');
    if(vp2){
      const toSvg=(e)=>{const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-flowViewport.x)/flowViewport.scale,y:(e.clientY-r.top-flowViewport.y)/flowViewport.scale}};
      canvas.querySelectorAll('.flow-node').forEach(n=>{
        // use the explicit data-flow-id so long/truncated model names still
        // persist under the right key (textContent matching broke for labels >16 chars)
        const id=n.getAttribute('data-flow-id');
        if(!id)return;
        const key=id;  // already "swap" | "model:<id>" | "machine:<name>"
        n.style.cursor='grab';
        n.addEventListener('mousedown',(e)=>{e.stopPropagation();const s=toSvg(e);const tr=n.getAttribute('transform')||'';const mm=tr.match(/translate\(([^,]+),([^)]+)\)/);const ox=mm?parseFloat(mm[1]):0,oy=mm?parseFloat(mm[2]):0;flowDrag={id:key,startX:s.x,startY:s.y,origX:ox,origY:oy,node:n};n.style.cursor='grabbing';canvas.classList.add('grabbing');});
      });
      if(!canvas._flowDragBound){canvas._flowDragBound=true;
        window.addEventListener('mousemove',(e)=>{if(!flowDrag)return;const c=toSvg(e);const dx=c.x-flowDrag.startX,dy=c.y-flowDrag.startY;flowLayout[flowDrag.id]={x:flowDrag.origX+dx,y:flowDrag.origY+dy};flowDrag.node.setAttribute('transform',`translate(${flowLayout[flowDrag.id].x},${flowLayout[flowDrag.id].y})`);requestFlowRender();});
        window.addEventListener('mouseup',()=>{if(!flowDrag)return;const id=flowDrag.id;flowDrag.node.style.cursor='grab';canvas.classList.remove('grabbing');flowDrag=null;requestFlowRender();api('POST','/api/flow/layout',{layout:flowLayout}).catch(()=>{});});
      }
    }
  } catch(e){}

}

/* ---------------- collapsible close animation ----------------
   Native <details> un-opens instantly; intercept the toggle, play the
   close animation, then actually close. Applies to every details element
   on the page (theme builder, defaults, model advanced, accordions). */
document.addEventListener("click", (e) => {
  const sum = e.target.closest("details > summary");
  if (!sum) return;
  const d = sum.parentElement;
  if (!d || !d.open) return;          // opening animates via CSS keyframes
  e.preventDefault();
  d.classList.add("details-closing");
  setTimeout(() => {
    d.classList.remove("details-closing");
    d.open = false;
  }, 170);
}, true);

/* ---------------- init ---------------- */
startAutoRefresh();
applyRibbons();   // global ribbon backdrop behind every page (toggleable)
on("#set-ribbons-chip", "click", () => {
  localStorage.setItem("ui-ribbons", ribbonsOn() ? "0" : "1");
  applyRibbons();
});
document.querySelectorAll("#theme-mode-seg button").forEach(b => {
  b.addEventListener("click", () => applyTheme(currentTheme(), b.dataset.mode));
});
/* ---------------- live logs (Fleet > Logs) ----------------
   Real-time raw tail of llama-swap's proxy + upstream feeds over
   /ws/logs/stream. Lines arrive as {src, line} JSON; kept in a bounded
   buffer and rendered as diff-friendly rows (append-only, trim from top)
   so a fast stream can't rebuild the whole DOM per line. */
const LOGS_BUFFER_MAX = 2000;   // lines kept in the buffer
const LOGS_DOM_MAX = 600;       // rows kept in the DOM (older ones pruned)
let logsWs = null;
let logsBuffer = [];            // [{src, line, ts, hit}]
let logsPaused = false;
let logsAutoScroll = true;
let logsFilterText = "";
let logsFilterRe = null;
let logsInitDone = false;

function logsStatus(text, cls) {
  const el = $("#logs-status");
  if (el) { el.textContent = text; el.className = `serving-ind ${cls || ""}`; }
}

function logsLineMatches(entry) {
  const srcOk = (entry.src === "proxy")
    ? $("#logs-src-proxy")?.checked !== false
    : $("#logs-src-upstream")?.checked !== false;
  if (!srcOk) return false;
  if (!logsFilterText) return true;
  if (logsFilterRe) return logsFilterRe.test(entry.line);
  return entry.line.toLowerCase().includes(logsFilterText);
}

function logsRenderAppend(entry) {
  const view = $("#logs-view");
  if (!view) return;
  const empty = view.querySelector(".log-empty");
  if (empty) empty.remove();
  const row = document.createElement("span");
  row.className = "logs-line" + (entry.hit ? " log-hit" : "");
  const t = new Date(entry.ts);
  const hh = `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}:${String(t.getSeconds()).padStart(2, "0")}`;
  row.innerHTML = `<span class="log-src ${escHtml(entry.src)}">${escHtml(entry.src)}</span>` +
    `<span class="log-time">${hh}</span>` +
    escHtml(entry.line);
  view.appendChild(row);
  while (view.children.length > LOGS_DOM_MAX) view.firstChild.remove();
}

function logsRefilter() {
  // full pass (filter edits / src toggles): rebuild the visible tail
  const view = $("#logs-view");
  if (!view) return;
  view.innerHTML = "";
  const tail = logsBuffer.slice(-LOGS_DOM_MAX);
  let shown = 0;
  for (const e of tail) {
    e.hit = logsLineMatches(e);
    if (e.hit) { logsRenderAppend(e); shown++; }
  }
  if (!shown) view.innerHTML = '<span class="log-empty">no matching lines</span>';
  const count = $("#logs-count");
  if (count) count.textContent = `${shown} shown / ${logsBuffer.length} buffered`;
  if (logsAutoScroll) view.scrollTop = view.scrollHeight;
}

function logsUpdateCount() {
  const count = $("#logs-count");
  if (count) {
    const shown = [...$("#logs-view").children].filter(c => c.classList.contains("logs-line")).length;
    count.textContent = `${shown} shown / ${logsBuffer.length} buffered`;
  }
}

function handleLogLine(src, line) {
  if (!line) return;
  const entry = { src, line, ts: Date.now(), hit: false };
  logsBuffer.push(entry);
  if (logsBuffer.length > LOGS_BUFFER_MAX) logsBuffer.splice(0, logsBuffer.length - LOGS_BUFFER_MAX);
  if (logsPaused) return;
  entry.hit = logsLineMatches(entry);
  if (!entry.hit) { logsUpdateCount(); return; }
  const view = $("#logs-view");
  const empty = view?.querySelector(".log-empty");
  if (empty) empty.remove();
  logsRenderAppend(entry);
  const stick = logsAutoScroll && (view.scrollHeight - view.scrollTop - view.clientHeight < 60);
  if (stick || logsAutoScroll) view.scrollTop = view.scrollHeight;
  logsUpdateCount();
}

function logsConnect() {
  if (logsWs) { try { logsWs.close(); } catch (_) {} logsWs = null; }
  const ws = new WebSocket(`ws://${location.host}/ws/logs/stream`);
  logsWs = ws;
  const openedAt = Date.now();
  ws.onopen = () => logsStatus("live", "status-ok");
  ws.onmessage = (e) => {
    if (Date.now() - openedAt < 1500) return;  // backend priming window
    let ev;
    try { ev = JSON.parse(e.data); } catch (_) { return; }
    if (ev.ping) { logsStatus("live", "status-ok"); return; }
    if (typeof ev.line === "string") handleLogLine(ev.src || "proxy", ev.line);
  };
  ws.onclose = () => {
    if (logsWs !== ws) return;
    logsWs = null;
    logsStatus("disconnected — retrying", "status-warn");
    if ($("#tab-logs")?.classList.contains("active")) setTimeout(logsConnect, 2000);
  };
  ws.onerror = () => logsStatus("error", "status-bad");
}

function initLogsTab() {
  logsStatus("connecting…");
  if (!logsInitDone) {
    logsInitDone = true;
    $("#logs-clear")?.addEventListener("click", () => {
      logsBuffer = [];
      const view = $("#logs-view");
      if (view) view.innerHTML = '<span class="log-empty">buffer cleared — waiting for lines…</span>';
      logsUpdateCount();
    });
    $("#logs-copy")?.addEventListener("click", async () => {
      const text = logsBuffer.filter(x => logsLineMatches(x)).map(x => `[${x.src}] ${x.line}`).join("\n");
      try { await navigator.clipboard.writeText(text); toast(`copied ${logsBuffer.length} lines`, "ok"); }
      catch (_) { toast("clipboard unavailable", "bad"); }
    });
    $("#logs-pause")?.addEventListener("click", (e) => {
      logsPaused = !logsPaused;
      e.currentTarget.classList.toggle("active", logsPaused);
      e.currentTarget.innerHTML = logsPaused
        ? '<i class="fa-solid fa-play"></i>' : '<i class="fa-solid fa-pause"></i>';
      if (!logsPaused) logsRefilter();
    });
    $("#logs-scroll")?.addEventListener("click", (e) => {
      logsAutoScroll = !logsAutoScroll;
      e.currentTarget.classList.toggle("active", logsAutoScroll);
      if (logsAutoScroll) {
        const view = $("#logs-view");
        if (view) view.scrollTop = view.scrollHeight;
      }
    });
    $("#logs-filter")?.addEventListener("input", (e) => {
      const v = e.target.value;
      logsFilterText = v;
      // literal search by default; /regex/ syntax opts into regex mode
      if (v.length > 1 && v.startsWith("/") && v.endsWith("/") && v.length > 2) {
        try { logsFilterRe = new RegExp(v.slice(1, -1), "i"); }
        catch (_) { logsFilterRe = null; }
      } else logsFilterRe = null;
      logsRefilter();
    });
    for (const id of ["logs-src-proxy", "logs-src-upstream"]) {
      document.getElementById(id)?.addEventListener("change", logsRefilter);
    }
    // user scrolled up = manual reading; re-enable only by clicking the button
    $("#logs-view")?.addEventListener("scroll", (e) => {
      const el = e.target;
      if (el.scrollHeight - el.scrollTop - el.clientHeight > 60) logsAutoScroll = false;
    });
    document.getElementById("logs-scroll")?.classList.add("active");
  }
  const view = $("#logs-view");
  if (view && !view.children.length) {
    view.innerHTML = '<span class="log-empty">connecting — lines appear as llama-swap serves traffic…</span>';
  }
  logsConnect();
}

/* ---------------- user page usage graphs (pure SVG, no deps) -------- */

let usagePeriod = "day";

function _uc_sub(el, text) {
  const e = $(el);
  if (e) e.textContent = text;
}

/* ── gauge skeleton (theme classed) ──────────────────────────────
   Uniform dial for every speedometer: 270° arc track (neutral),
   value arc relative to max, big center value, unit sub, peak note.
   `color` = css-Class suffix (accent|ok|ribbon|warn): arcs/texts use
   currentColor; theme re-skin is automatic on change. */
function _gaugeSkel(value, max, color, fmt, unit, label) {
  const W = 300, H = 175, cx = W/2, cy = H*0.78, R = 105;
  const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
  const frac = Math.max(0, Math.min(1, max > 0 ? value / max : 0));
  const arc = (f0, f1, wdt, op) => {
    const r0 = a0 + (a1 - a0) * f0, r1 = a0 + (a1 - a0) * f1;
    const large = (r1 - r0) > Math.PI ? 1 : 0;
    const x0 = cx + R * Math.cos(r0), y0 = cy + R * Math.sin(r0);
    const x1 = cx + R * Math.cos(r1), y1 = cy + R * Math.sin(r1);
    return `<path d="M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${R} ${R} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}" fill="none" stroke-width="${wdt}" stroke-linecap="round" opacity="${op}"/>`;
  };
  const v = Math.max(0, value || 0);
  return `<span class="gauge-wrap gauge-${color}" style="display:contents">
    <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto">
      ${arc(0, 1, 14, 1)}
      ${frac > 0 ? arc(0, frac, 14, 0.95) : ""}
      <text class="gauge-val" x="${cx}" y="${cy - 34}" font-size="30" font-weight="700" text-anchor="middle">${escHtml(fmt(v))}</text>
      <text x="${cx}" y="${cy - 15}" font-size="10.5" text-anchor="middle" fill="var(--muted)">${escHtml(unit || label || "")}</text>
      <text x="${cx}" y="${cy + 14}" font-size="9.5" text-anchor="middle" fill="var(--muted)">peak ${escHtml(fmt(max))}</text>
    </svg></span>`;
}

function renderDecodeSpeedByModel(r, speed) {
  /* per-model dials: one gauge card per model, arc vs fastest×1.15;
     fastest = accent badge; all colors inherit the theme (accent). */
  const box = $("#user-dspeed");
  const sub = $("#udsp-sub");
  if (!box) return;
  const entries = Object.entries(speed || {}).
    filter(([m, v]) => v && v > 0 && Number.isFinite(v)).
    sort((a, b) => b[1] - a[1]);
  if (sub) sub.textContent = entries.length
    ? `${entries[0][0]} fastest · ${entries[0][1]} tok/s · ${entries.length} model(s)`
    : "";
  if (!entries.length) {
    box.innerHTML = '<div class="card gauge-card accent" style="display:flex; align-items:center; justify-content:center"><span class="hint">no decode volume yet — chat a decoder model through the gateway; decision models (laya) emit no completion tokens</span></div>';
    return;
  }
  const max = Math.max(...entries.map(e => e[1])) * 1.15;
  box.innerHTML =
    entries.map(([m, v], idx) => `
    <div class="card gauge-card accent" style="display:flex; flex-direction:column;
      align-items:center; justify-content:center;">
      <div style="font-size:0.95em; font-weight:600; width:100%; text-align:center;
        overflow:hidden; text-overflow:ellipsis" title="${escHtml(m)}">
        ${escHtml(m.length > 30 ? m.slice(0,29) + "…" : m)}${idx === 0 ? ' <span class="badge" style="background:var(--accent-dim);color:var(--accent)">fastest</span>' : ""}
      </div>
      ${_gaugeSkel(v, max, "accent", x => x.toFixed(1), "tok/s", "")}
    </div>`).join("");
}

function _fmtTok(n) {
  n = n || 0;
  return n >= 1e9 ? (n/1e9).toFixed(2)+"B"
       : n >= 1e6 ? (n/1e6).toFixed(2)+"M"
       : n >= 1e3 ? (n/1e3).toFixed(1)+"K" : String(n);
}
function _fmtHours(h) {
  h = h || 0;
  if (h >= 1) return h.toFixed(1) + " h";
  const m = h * 60;
  if (m >= 1) return m.toFixed(1) + " min";
  return Math.round(h * 3600) + " s";
}

/* ── chart panel ─────────────────────────────────────────────
   One polished panel: y ticks with VALUES, subtle grid, a metric
   rendered as bars or a smooth area/line, hover pill readout via
   title on an invisible column rect (full-height hit area), date
   labels sparse + always-last. */
function _chartPanel({buckets, metric, mode, color, fmt, unit, H: H0}) {
  /* READABLE-AT-A-GLANCE micro-chart (sparkline school):
     - NO y-axis lines/labels — the max value sits top-left once
     - x labels: first + last only
     - bars: full-width rounded columns, one series
     - area: smooth line + gradient fill
     - current (last) bucket's value = the big readout, right-top
     - hover: full-height hit column + tooltip */
  const H = H0 || 150, W = 560, padL = 10, padR = 10, padT = 26, padB = 20;
  const data = buckets.slice();
  if (!data.length) return '<div class="hint">no data</div>';
  const vals = data.map(d => d[metric] || 0);
  const max = Math.max(1, ...vals);
  const iw = W - padL - padR, ih = H - padT - padB;
  const step = iw / data.length;
  const v2y = v => padT + ih - (v / max) * ih;
  const x2c = i => padL + i * step + step / 2;

  // max marker (single label, top-left)
  let marks = `<text x="${padL}" y="12" font-size="10" font-weight="600"
    fill="var(--muted, #999)">max ${escHtml(fmt(max))}</text>`;
  // current readout (last nonzero bucket), top-right — the ticker number
  const lastIdx = (() => { for (let i = data.length-1; i >= 0; i--) if (vals[i] > 0) return i; return -1; })();
  if (lastIdx >= 0) {
    marks += `<text x="${W-padR}" y="12" font-size="13" font-weight="700"
      text-anchor="end" fill="${color}">${escHtml(fmt(vals[lastIdx]))}${escHtml(unit||"")}</text>`;
  }

  if (mode === "bar") {
    const bw = Math.max(2, step * 0.62);
    data.forEach((d, i) => {
      const v = vals[i];
      const y = v2y(v), h = padT + ih - y;
      marks += `<rect x="${(x2c(i)-bw/2).toFixed(1)}" y="${y.toFixed(1)}"
        width="${bw.toFixed(1)}" height="${Math.max(v>0?2.5:0, h).toFixed(1)}"
        rx="${Math.min(4, bw/3)}"
        fill="${color}" opacity="${i === lastIdx ? 1 : 0.72}"/>`;
    });
  } else {
    const pts = data.map((d, i) => [x2c(i), v2y(vals[i])]);
    if (pts.length === 1) {
      marks += `<circle cx="${pts[0][0]}" cy="${pts[0][1]}" r="3.5" fill="${color}"/>`;
    } else {
      let path = `M ${pts[0][0]} ${pts[0][1]}`;
      for (let i = 1; i < pts.length; i++) {
        const [x0, y0] = pts[i-1], [x1, y1] = pts[i];
        const cx = (x0 + x1) / 2;
        path += ` C ${cx} ${y0}, ${cx} ${y1}, ${x1} ${y1}`;
      }
      const gid = `g${metric}${Math.round(Math.random()*1e4)}`;
      marks += `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${color}" stop-opacity="0.35"/>
        <stop offset="1" stop-color="${color}" stop-opacity="0.02"/>
      </linearGradient></defs>`;
      marks += `<path d="${path} L ${pts[pts.length-1][0]} ${padT+ih}
        L ${pts[0][0]} ${padT+ih} Z" fill="url(#${gid})"/>
        <path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"
        stroke-linejoin="round" opacity="0.95"/>`;
      pts.forEach(([x, y], i) => {
        if (vals[i] > 0)
          marks += `<circle cx="${x}" cy="${y}" r="3" fill="${color}"
          stroke="var(--bg, #fff)" stroke-width="1.4"/>`;
      });
    }
  }
  // hover hit columns (full height)
  data.forEach((d, i) => {
    const x = padL + i * step;
    marks += `<rect x="${x}" y="${padT}" width="${step}" height="${ih}"
      fill="transparent"><title>${escHtml(d.label)} — ${escHtml(fmt(vals[i]))}${escHtml(unit||"")}</title></rect>`;
  });
  // x labels: first + last
  const labs = [0, data.length - 1].filter((v, i, a) => a.indexOf(v) === i);
  let labels = "";
  labs.forEach(i => {
    labels += `<text x="${Math.min(Math.max(x2c(i), 18), W-18)}" y="${H-5}"
    font-size="10" text-anchor="${i === 0 ? "start" : "end"}"
    fill="var(--muted, #999)">${escHtml(_xlab(data[i].label, usagePeriod))}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; font-family:inherit; overflow:visible">${marks}${labels}</svg>`;
}

function _xlab(label, period) {
  // day 2026-09-22 → "22 Sep" · month 2026-09 → "Sep" · year → "2026"
  const m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  if (period === "day") {
    const [y, mo, d] = label.split("-");
    return `${parseInt(d)} ${m[(parseInt(mo)||1)-1]||""}`;
  }
  if (period === "month") {
    const [y, mo] = label.split("-");
    return m[(parseInt(mo)||1)-1] || label;
  }
  return label;
}

function _panelTitle(txt, tip) {
  return `<div class="hint" style="margin:10px 0 3px; display:flex; justify-content:space-between">
    <span>${escHtml(txt)}</span>${tip ? `<span>${escHtml(tip)}</span>` : ""}</div>`;
}



function _withAvg(buckets) {
  return buckets.map(b => ({...b,
    _avglat: b.requests ? +((b.wall_s || 0) / b.requests).toFixed(2) : 0,
    _spd: (b.wall_s > 0 && (b.tok_out || 0) > 0)
      ? +(b.tok_out / b.wall_s).toFixed(1) : 0}));
}

async function loadUsageTotals(light = false) {
  /* all-time pane = GAUGE CARDS ONLY (same card anatomy as the over-time
     cards: 2-across, each metric in its own card with gauge + note).
     Per-model detail lives on the Decode speed pane. light=true: only
     update values in existing cards (no DOM rebuild — zero flicker). */
  const box = $("#usage-totals"), per = $("#usage-permodel");
  if (!box) return;
  const pane = $("#user-pane-alltime");
  if (light && pane?.querySelectorAll(".gauge-card").length === 5) {
    // LIGHT: fetch fresh data, patch in place
    try {
      const r = await api("GET", "/api/usage?period=all");
      const t = r.total || {};
      let peakDay = {requests: 1, tok_out: 1, tok_in: 1, wall_s: 1, days: 1};
      try {
        const dr = await api("GET", "/api/usage?period=day&count=90");
        const buckets = (dr.buckets || []);
        const pk = (k) => Math.max(1, ...buckets.map(b => b[k] || 0));
        peakDay = {requests: pk("requests"), tok_out: pk("tok_out"),
                   tok_in: pk("tok_in"),
                   wall_s: Math.max(1, ...buckets.map(b => b.wall_s || 0)),
                   days: buckets.filter(b => (b.requests||0) > 0).length || 1};
      } catch (_) {}
      const spd = r.avg_tok_per_s || 0;
      const cards = pane.querySelectorAll(".gauge-card");
      const specs = [
        [t.requests || 0, peakDay.requests * 1.15, _fmtTok],
        [t.tok_out || 0, peakDay.tok_out * 1.15, _fmtTok],
        [t.tok_in || 0, peakDay.tok_in * 1.15, _fmtTok],
        [t.hours || 0, (peakDay.wall_s / 3600) * (peakDay.days || 1) || (t.hours || 0.01), _fmtHours],
        [spd, Math.max(spd * 1.15, 1), v => v.toFixed(1)],
      ];
      cards.forEach((c, i) => {
        const [v, m, f] = specs[i] || [];
        if (v !== undefined) _gaugePatch(c, v, m, f);
      });
      return;
    } catch (err) { /* fall through to full render */ }
  }
  try {
    const r = await api("GET", "/api/usage?period=all");
    const t = r.total || {};
    let peakDay = {requests: 1, tok_out: 1, tok_in: 1};
    try {
      const dr = await api("GET", "/api/usage?period=day&count=90");
      const buckets = (dr.buckets || []);
      const pk = (k) => Math.max(1, ...buckets.map(b => b[k] || 0));
      peakDay = {requests: pk("requests"), tok_out: pk("tok_out"),
                 tok_in: pk("tok_in"),
                 wall_s: Math.max(1, ...buckets.map(b => b.wall_s || 0)),
                 days: buckets.filter(b => (b.requests||0) > 0).length || 1};
    } catch (_) {}
    const G = (value, max, cls, fmt, unit) => `
      <div class="card gauge-card ${cls}">
        ${_gaugeSkel(value, Math.max(max, 1e-6), cls, fmt, unit, "")}
      </div>`;
    const spd = r.avg_tok_per_s || 0;
    box.innerHTML =
      G(t.requests || 0, peakDay.requests * 1.15, "accent",
        v => _fmtTok(v), "requests all-time") +
      G(t.tok_out || 0, peakDay.tok_out * 1.15, "accent",
        _fmtTok, "tokens out") +
      G(t.tok_in || 0, peakDay.tok_in * 1.15, "accent", _fmtTok, "tokens in") +
      G(t.hours || 0,
        (peakDay.wall_s / 3600) * (peakDay.days || 1) || (t.hours || 0.01),
        "accent", _fmtHours, "compute time") +
      G(spd, Math.max(spd * 1.15, 1), "accent", v => v.toFixed(1), "tok/s avg");
    // per-model bar rows removed from this pane (decode speed pane owns per-model detail)
    if (per) per.innerHTML = "";
  } catch (err) {
    box.textContent = "usage failed: " + err.message;
  }
}




async function loadUsageSeries(period, light = false) {
  usagePeriod = period || usagePeriod;
  const panes = {
    requests: "#uc-requests", latency: "#uc-latency",
    tokens: "#uc-tokens",  errors: "#uc-errors",
  };
  const paneEl = $("#user-pane-overtime");
  const LIGHT = light && paneEl && paneEl.querySelectorAll(".gauge-wrap").length >= 3;
  try {
    const q = `period=${usagePeriod}&count=${usagePeriod === "day" ? 14 : usagePeriod === "month" ? 12 : 6}`;
    const r = await api("GET", "/api/usage?" + q);
    const buckets = _withAvg(r.buckets || []);
    const sum = k => buckets.reduce((a, b) => a + (b[k] || 0), 0);
    const peakOf = k => Math.max(1, ...buckets.map(b => b[k] || 0));
    const reqs = sum("requests");
    if (!reqs) {
      Object.values(panes).forEach(sel => {
        const e = $(sel);
        if (e) e.innerHTML = '<span class="hint">no usage in this window yet</span>';
      });
      ["#uc-requests-sub", "#uc-latency-sub", "#uc-tokens-sub",
       "#uc-errors-sub"].forEach(s2 => _uc_sub(s2, ""));
      return;
    }
    const avgLat = buckets.reduce((a, b) =>
      a + ((b.requests) ? (b._avglat || 0) * b.requests : 0), 0) /
      Math.max(1, reqs);
    const latRef = Math.max(0.5, Math.max(...buckets.map(b => b._avglat || 0)));
    const wallSum = buckets.reduce((a, b) => a + (b.wall_s || 0), 0);
    const tokSum = sum("tok_out");
    const errPct = reqs ? 100 * sum("errors") / reqs : 0;
    // (selector, cardClass, value, ref, fmt, unitLabel)
    const specs = [
      ["#uc-requests", "accent", reqs, peakOf("requests") * 1.15,
       v => _fmtTok(v), "requests"],
      ["#uc-latency", "accent", avgLat, latRef, v => v.toFixed(1), "s / request"],
      ["#uc-tokens", "accent",
       tokSum > 0 ? tokSum : sum("tok_in"),
       tokSum > 0 ? peakOf("tok_out") * 1.15 : peakOf("tok_in") * 1.15,
       _fmtTok, tokSum > 0 ? "tokens out" : "tokens in (no decode yet)"],
      ["#uc-errors", "accent", errPct, Math.max(20, errPct * 1.5),
       v => v.toFixed(1) + "%", "of requests failed"],
    ];
    for (const [sel, cls, v, m, f, lbl] of specs) {
      const e = $(sel);
      if (!e) continue;
      const card = e.closest(".card");
      if (card) card.classList.add("gauge-card", cls);
      if (LIGHT && e.querySelector(".gauge-wrap")) {
        // tokens card may FLIP metric (out↔in) — rebuild only then
        const wrap = e.querySelector(".gauge-wrap");
        const nowLbl = (wrap.querySelector("text:nth-of-type(2)") || {}).textContent || "";
        const wantsLbl = lbl;
        const flip = sel === "#uc-tokens" &&
          ((wantsLbl.startsWith("tokens out") && !nowLbl.startsWith("tokens out")) ||
           (wantsLbl.startsWith("tokens in") && !nowLbl.startsWith("tokens")));
        if (flip) e.innerHTML = _gaugeSkel(v, m, cls, f, lbl, "");
        else _gaugePatch(card, v, m, f);
      } else {
        e.innerHTML = _gaugeSkel(v, m, cls, f, lbl, "");
      }
    }
    _uc_sub("#uc-requests-sub", `${reqs} total`);
    _uc_sub("#uc-latency-sub", `${avgLat.toFixed(1)}s avg`);
    _uc_sub("#uc-tokens-sub", `${_fmtTok(sum("tok_in"))} in`);
    _uc_sub("#uc-errors-sub", sum("errors") ? `${sum("errors")} errors` : "");
  } catch (err) {
    Object.values(panes).forEach(sel => {
      const e = $(sel);
      if (e) e.textContent = "failed: " + err.message;
    });
  }
}

async function loadDecodeSpeed() {
  try {
    const r = await api("GET", "/api/usage?period=all");
    renderDecodeSpeedByModel(r, r.tok_out_per_s || {});
  } catch (err) {
    const box = $("#user-dspeed");
    if (box) box.textContent = "speed data failed: " + err.message;
  }
}
