/* Chess Volatility Bar — frontend controller */
/* eslint-disable no-undef */
(function () {
  "use strict";

  // ── Vendor check ────────────────────────────────────────────────────────── //
  const missing = [];
  if (typeof window.jQuery    === "undefined") missing.push("jQuery");
  if (typeof window.Chessboard === "undefined") missing.push("chessboard.js");
  if (typeof window.Chess     === "undefined") missing.push("chess.js");
  if (typeof window.Chart     === "undefined") missing.push("Chart.js");
  if (missing.length) {
    const msg = `Frontend failed to load: ${missing.join(", ")}. Check /vendor/* is served.`;
    const el = document.getElementById("bootError");
    if (el) { el.textContent = msg; el.classList.remove("hidden"); }
    console.error(msg);
    return;
  }

  // ── Constants ────────────────────────────────────────────────────────────  //
  const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const DEFAULT_FEN_TAIL = "w KQkq - 0 1";
  const AUTO_DEBOUNCE_MS = 450;

  // ── DOM refs ─────────────────────────────────────────────────────────────  //
  const $ = (s) => document.querySelector(s);
  const fenInput      = $("#fenInput");
  const copyFenBtn    = $("#copyFen");
  const deepToggle    = $("#deepToggle");
  const deepToggleGame = $("#deepToggleGame");
  const evalBarEl     = $("#evalBar");
  const evalLabelEl   = $("#evalLabel");
  const volBarEl      = $("#volBar");
  const volLabelEl    = $("#volLabel");

  const btnStart      = $("#btnStart");
  const btnClear      = $("#btnClear");
  const btnFlip       = $("#btnFlip");
  const btnAnalyzeFen = $("#btnAnalyzeFen");
  const autoAnalyze   = $("#autoAnalyze");
  const editorStatus  = $("#editorStatus");
  const turnWhiteBtn  = $("#turnWhite");
  const turnBlackBtn  = $("#turnBlack");

  const pgnFileInput  = $("#pgnFile");
  const pgnInput      = $("#pgnInput");
  const btnLoadPgn    = $("#btnLoadPgn");
  const btnAnalyzePgn = $("#btnAnalyzePgn");
  const btnStopPgn    = $("#btnStopPgn");
  const gameStatus    = $("#gameStatus");
  const plyStatus     = $("#plyStatus");
  const moveListEl    = $("#moveList");
  const chartWrap     = $("#chartWrap");
  const moveListWrap  = $("#moveListWrap");
  const chartCanvas   = $("#chart");

  // ── Tab switching ─────────────────────────────────────────────────────── //
  function setTab(name) {
    document.querySelectorAll(".tab").forEach((t) => {
      const active = t.dataset.tab === name;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-for]").forEach((el) => {
      const match = el.dataset.for === name;
      el.classList.toggle("hidden", !match);
    });
    // Re-show conditional children inside game-panel only if they have data
    if (name === "game") {
      if (loadedPlies && loadedPlies.length) moveListWrap.classList.remove("hidden");
      if (chart) chartWrap.classList.remove("hidden");
    }
  }

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => setTab(btn.dataset.tab));
  });

  // ── Shared deep toggle state ──────────────────────────────────────────── //
  function syncDeep(source) {
    const val = source.checked;
    if (deepToggle && deepToggle !== source) deepToggle.checked = val;
    if (deepToggleGame && deepToggleGame !== source) deepToggleGame.checked = val;
  }
  if (deepToggle)     deepToggle.addEventListener("change",     () => syncDeep(deepToggle));
  if (deepToggleGame) deepToggleGame.addEventListener("change", () => syncDeep(deepToggleGame));

  function deepEnabled() {
    return !!(deepToggle && deepToggle.checked);
  }

  // ── Board ─────────────────────────────────────────────────────────────── //
  let suppressSync = false;

  const board = Chessboard("board", {
    draggable:    true,
    sparePieces:  true,
    dropOffBoard: "trash",
    position:     "start",
    pieceTheme:   "/vendor/img/pieces/{piece}.png",
    onChange: () => {
      if (suppressSync) return;
      syncFenFromBoard();
      scheduleAutoAnalyze();
    },
  });

  window.addEventListener("resize", () => board.resize());

  function assembleFen() {
    const parts = (fenInput.value || "").trim().split(/\s+/);
    const tail = parts.length > 1 ? parts.slice(1).join(" ") : DEFAULT_FEN_TAIL;
    return `${board.fen()} ${tail}`;
  }

  function syncFenFromBoard() {
    const fen = assembleFen();
    fenInput.value = fen;
    editorStatus.textContent = validateFen(fen) ? "" : "⚠ Incomplete or illegal position";
    syncTurnToggleFromFen();
  }

  function syncBoardFromFen(fen) {
    const parts = fen.trim().split(/\s+/);
    if (parts.length >= 1) {
      suppressSync = true;
      try { board.position(parts[0], false); } finally { suppressSync = false; }
    }
  }

  function validateFen(fen) {
    try { const g = new Chess(); return g.load(fen); } catch (_) { return false; }
  }

  function getTurn() {
    const parts = (fenInput.value || "").trim().split(/\s+/);
    return parts[1] === "b" ? "b" : "w";
  }

  function setTurn(color) {
    const next = color === "b" ? "b" : "w";
    const parts = (fenInput.value || "").trim().split(/\s+/);
    const tail = DEFAULT_FEN_TAIL.split(/\s+/);
    const placement = parts[0] || board.fen();
    const castling  = parts[2] ?? tail[1];
    const ep        = parts[3] ?? tail[2];
    const halfmove  = parts[4] ?? tail[3];
    const fullmove  = parts[5] ?? tail[4];
    fenInput.value = `${placement} ${next} ${castling} ${ep} ${halfmove} ${fullmove}`;
    editorStatus.textContent = validateFen(fenInput.value) ? "" : "⚠ Incomplete or illegal position";
  }

  function syncTurnToggleFromFen() {
    const turn = getTurn();
    if (turnWhiteBtn) {
      const white = turn === "w";
      turnWhiteBtn.classList.toggle("active", white);
      turnWhiteBtn.setAttribute("aria-checked", white ? "true" : "false");
    }
    if (turnBlackBtn) {
      const black = turn === "b";
      turnBlackBtn.classList.toggle("active", black);
      turnBlackBtn.setAttribute("aria-checked", black ? "true" : "false");
    }
  }

  function onTurnBtnClick(color) {
    if (getTurn() === color) return;
    setTurn(color);
    syncTurnToggleFromFen();
    scheduleAutoAnalyze();
  }

  if (turnWhiteBtn) turnWhiteBtn.addEventListener("click", () => onTurnBtnClick("w"));
  if (turnBlackBtn) turnBlackBtn.addEventListener("click", () => onTurnBtnClick("b"));

  // ── Auto-analyze (debounced) ─────────────────────────────────────────── //
  let autoTimer = null;

  function scheduleAutoAnalyze() {
    if (!autoAnalyze || !autoAnalyze.checked) return;
    if (autoTimer) clearTimeout(autoTimer);
    autoTimer = setTimeout(() => {
      autoTimer = null;
      const fen = fenInput.value.trim();
      if (!validateFen(fen)) return;
      analyzeFen(fen).catch((err) => {
        editorStatus.textContent = `Error: ${err.message || err}`;
      });
    }, AUTO_DEBOUNCE_MS);
  }

  btnStart.addEventListener("click", () => {
    syncBoardFromFen(STARTING_FEN);
    fenInput.value = STARTING_FEN;
    syncTurnToggleFromFen();
    scheduleAutoAnalyze();
  });

  btnClear.addEventListener("click", () => {
    suppressSync = true;
    board.clear(false);
    suppressSync = false;
    syncFenFromBoard();
    scheduleAutoAnalyze();
  });

  btnFlip.addEventListener("click", () => board.flip());

  btnAnalyzeFen.addEventListener("click", () => {
    const fen = fenInput.value.trim();
    if (!validateFen(fen)) { editorStatus.textContent = "Invalid FEN."; return; }
    analyzeFen(fen).catch((err) => { editorStatus.textContent = `Error: ${err.message || err}`; });
  });

  fenInput.addEventListener("change", () => {
    const fen = fenInput.value.trim();
    if (validateFen(fen)) {
      syncBoardFromFen(fen);
      syncTurnToggleFromFen();
      editorStatus.textContent = "";
      scheduleAutoAnalyze();
    } else {
      syncTurnToggleFromFen();
      editorStatus.textContent = "Invalid FEN.";
    }
  });

  copyFenBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(fenInput.value);
      copyFenBtn.title = "Copied!";
      setTimeout(() => (copyFenBtn.title = "Copy FEN to clipboard"), 1200);
    } catch (_) { /* denied */ }
  });

  // ── Bar rendering ─────────────────────────────────────────────────────── //

  function evalCpToFill(cpWhitePov) {
    if (cpWhitePov == null) return 0.5;
    return Math.min(0.97, Math.max(0.03, 0.5 + 0.5 * Math.tanh(cpWhitePov / 400)));
  }

  function formatEval(cp, turn) {
    if (cp == null) return "—";
    const w = turn === "b" ? -cp : cp;
    if (Math.abs(w) >= 1000) return (w > 0 ? "+" : "−") + "M";
    const abs = (Math.abs(w) / 100).toFixed(2);
    return (w >= 0 ? "+" : "−") + abs;
  }

  function scoreToColor(score) {
    if (score < 25) return "low";
    if (score < 60) return "medium";
    return "high";
  }

  function renderEvalBar(cpSideToMove, turn) {
    const w = turn === "b" ? -cpSideToMove : cpSideToMove;
    evalBarEl.style.setProperty("--fill", evalCpToFill(w));
    evalLabelEl.textContent = formatEval(cpSideToMove, turn);
  }

  function renderVolBar(result) {
    if (!result || result.score == null) {
      volBarEl.style.setProperty("--fill", 0);
      volBarEl.style.setProperty("--local", 0);
      volBarEl.style.setProperty("--split-visible", 0);
      volBarEl.dataset.color = "low";
      volBarEl.dataset.decided = "false";
      volLabelEl.textContent = result && result.reason ? `— ${result.reason}` : "—";
      return;
    }

    const score = result.score;
    const fill  = Math.max(0, Math.min(1, score / 100));
    volBarEl.style.setProperty("--fill", fill);
    volBarEl.dataset.color   = scoreToColor(score);
    volBarEl.dataset.decided = result.decided ? "true" : "false";

    const deep = result.recurse_depth_used > 0 && result.raw_cp > 0 && result.local_raw_cp != null;
    if (deep) {
      const localFrac = Math.max(0, Math.min(1, result.local_raw_cp / result.raw_cp));
      volBarEl.style.setProperty("--local", fill * localFrac);
      volBarEl.style.setProperty("--split-visible", 1);
      const lPct = Math.round(100 * fill * localFrac);
      const rPct = Math.round(100 * fill * (1 - localFrac));
      volLabelEl.textContent = `${score.toFixed(1)} L${lPct}/R${rPct}`;
    } else {
      volBarEl.style.setProperty("--local", 0);
      volBarEl.style.setProperty("--split-visible", 0);
      volLabelEl.textContent = score.toFixed(1);
    }
  }

  function setBarsLoading(on) {
    [evalBarEl, volBarEl].forEach((el) =>
      el.classList.toggle("bar-loading", on)
    );
  }

  // ── Analyze FEN ──────────────────────────────────────────────────────── //
  let inflightFen = null;

  async function analyzeFen(fen) {
    if (inflightFen) inflightFen.abort();
    const ctrl = new AbortController();
    inflightFen = ctrl;
    editorStatus.textContent = "Analyzing…";
    setBarsLoading(true);

    try {
      const resp = await fetch("/analyze/fen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen, deep: deepEnabled() }),
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const data = await resp.json();
      const turn = fen.trim().split(/\s+/)[1] || "w";
      renderEvalBar(data.volatility.best_eval_cp, turn);
      renderVolBar(data.volatility);
      const modeStr = data.mode === "deep" ? "deep" : "shallow";
      editorStatus.textContent =
        `${modeStr} · ${data.volatility.analyses} engine call${data.volatility.analyses !== 1 ? "s" : ""}`;
    } catch (err) {
      if (err.name === "AbortError") return;
      throw err;
    } finally {
      setBarsLoading(false);
      if (inflightFen === ctrl) inflightFen = null;
    }
  }

  // ── PGN / Game ───────────────────────────────────────────────────────── //
  let loadedPlies   = [];
  let plyResults    = [];
  let chart         = null;
  let pgnController = null;

  function resetGame() {
    loadedPlies = [];
    plyResults  = [];
    moveListEl.innerHTML = "";
    gameStatus.textContent  = "";
    plyStatus.textContent   = "";
    chartWrap.classList.add("hidden");
    moveListWrap.classList.add("hidden");
    destroyChart();
  }

  function destroyChart() {
    if (chart) { chart.destroy(); chart = null; }
  }

  function parsePgn(text) {
    try {
      const g = new Chess();
      if (!g.load_pgn(text, { sloppy: true })) return null;
      const history = g.history({ verbose: true });
      const replay  = new Chess();
      const plies   = [];
      for (const mv of history) {
        const fenBefore = replay.fen();
        const san = replay.move({ from: mv.from, to: mv.to, promotion: mv.promotion }).san;
        plies.push({ san, fen_before: fenBefore, fen_after: replay.fen() });
      }
      return plies;
    } catch (_) { return null; }
  }

  function renderMoveList() {
    moveListEl.innerHTML = "";
    if (!loadedPlies.length) return;

    const table = document.createElement("table");
    table.className = "move-table";

    const pairCount = Math.ceil(loadedPlies.length / 2);
    for (let i = 0; i < pairCount; i++) {
      const tr = document.createElement("tr");

      const numTd = document.createElement("td");
      numTd.className = "move-num";
      numTd.textContent = `${i + 1}.`;
      tr.appendChild(numTd);

      tr.appendChild(makeMoveCell(i * 2));

      if (loadedPlies[i * 2 + 1]) {
        tr.appendChild(makeMoveCell(i * 2 + 1));
      } else {
        tr.appendChild(document.createElement("td"));
      }

      table.appendChild(tr);
    }

    moveListEl.appendChild(table);
    moveListWrap.classList.remove("hidden");
  }

  function makeMoveCell(idx) {
    const ply = loadedPlies[idx];
    const td  = document.createElement("td");
    td.className    = "move-cell";
    td.dataset.idx  = String(idx);

    const sanSpan = document.createElement("span");
    sanSpan.className   = "move-san";
    sanSpan.textContent = ply.san;

    const vSpan = document.createElement("span");
    vSpan.className = "move-vscore";
    vSpan.id        = `mv-v-${idx}`;
    vSpan.textContent = "—";

    td.appendChild(sanSpan);
    td.appendChild(vSpan);
    td.addEventListener("click", () => jumpToPly(idx));
    return td;
  }

  function updateMoveVol(idx, score) {
    const span = document.getElementById(`mv-v-${idx}`);
    if (!span) return;
    if (score == null) { span.textContent = "—"; delete span.dataset.color; return; }
    span.textContent = Math.round(score).toString();
    span.dataset.color = scoreToColor(score);
  }

  function jumpToPly(idx) {
    if (idx < 0 || idx >= loadedPlies.length) return;
    const entry = loadedPlies[idx];

    suppressSync = true;
    try { board.position(entry.fen_after.split(/\s+/)[0], true); }
    finally { suppressSync = false; }

    fenInput.value = entry.fen_after;
    syncTurnToggleFromFen();

    const r = plyResults[idx];
    if (r) {
      const turn = entry.fen_before.split(/\s+/)[1] || "w";
      renderEvalBar(r.ply.volatility.best_eval_cp, turn);
      renderVolBar(r.ply.volatility);
    }

    document.querySelectorAll(".move-cell").forEach((c) =>
      c.classList.toggle("active", Number(c.dataset.idx) === idx)
    );

    const active = moveListEl.querySelector(".move-cell.active");
    if (active) active.scrollIntoView({ block: "nearest" });

    if (chart) {
      chart.data.datasets.forEach((ds) => {
        ds.pointRadius = ds.data.map((_, i) => (i === idx ? 5 : 2));
        ds.pointHoverRadius = ds.data.map((_, i) => (i === idx ? 7 : 4));
      });
      chart.update("none");
    }
  }

  // ── PGN load / analyze ───────────────────────────────────────────────── //
  btnLoadPgn.addEventListener("click", async () => {
    let text = pgnInput.value.trim();
    if (!text && pgnFileInput.files && pgnFileInput.files[0]) {
      text = await pgnFileInput.files[0].text();
      pgnInput.value = text;
    }
    if (!text) { gameStatus.textContent = "Paste a PGN or pick a file first."; return; }
    const plies = parsePgn(text);
    if (!plies) { gameStatus.textContent = "Could not parse that PGN."; return; }
    resetGame();
    loadedPlies = plies;
    renderMoveList();
    gameStatus.textContent = `Loaded ${plies.length} plies — click Analyze to compute volatility.`;
    if (plies.length) jumpToPly(0);
  });

  pgnFileInput.addEventListener("change", async () => {
    if (!pgnFileInput.files || !pgnFileInput.files[0]) return;
    pgnInput.value = await pgnFileInput.files[0].text();
  });

  btnAnalyzePgn.addEventListener("click", () => {
    const text = pgnInput.value.trim();
    if (!text) { gameStatus.textContent = "Paste a PGN or pick a file first."; return; }
    if (!loadedPlies.length) {
      const plies = parsePgn(text);
      if (plies) { loadedPlies = plies; renderMoveList(); }
    }
    startPgnStream(text);
  });

  btnStopPgn.addEventListener("click", () => {
    if (pgnController) pgnController.abort();
  });

  async function startPgnStream(pgnText) {
    if (pgnController) pgnController.abort();
    const ctrl = new AbortController();
    pgnController = ctrl;
    plyResults = [];
    destroyChart();

    btnAnalyzePgn.disabled = true;
    btnStopPgn.disabled    = false;
    gameStatus.textContent = "Starting…";
    plyStatus.classList.remove("hidden");

    try {
      const resp = await fetch("/analyze/pgn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pgn: pgnText, deep: deepEnabled() }),
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
      await consumeSse(resp.body, ctrl);
    } catch (err) {
      if (err.name === "AbortError") {
        gameStatus.textContent = "Analysis stopped.";
      } else {
        gameStatus.textContent = `Error: ${err.message || err}`;
      }
    } finally {
      btnAnalyzePgn.disabled = false;
      btnStopPgn.disabled    = true;
      if (pgnController === ctrl) pgnController = null;
    }
  }

  // ── SSE streaming ────────────────────────────────────────────────────── //
  async function consumeSse(stream, ctrl) {
    const reader  = stream.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const splitRe = /\r\n\r\n|\n\n/;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let m;
      while ((m = splitRe.exec(buf))) {
        const chunk = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        handleChunk(chunk);
      }
      if (ctrl.signal.aborted) {
        try { reader.cancel(); } catch (_) {}
        return;
      }
    }
  }

  function handleChunk(chunk) {
    let event = "message";
    const dataLines = [];
    for (const line of chunk.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    let payload;
    try { payload = JSON.parse(dataLines.join("\n")); } catch (_) { return; }
    if (event === "start") onStart(payload);
    else if (event === "ply")  onPly(payload);
    else if (event === "done") onDone(payload);
    else if (event === "error") onErr(payload);
  }

  function onStart(p) {
    gameStatus.textContent = `Analyzing (${p.mode})…`;
    ensureChart();
    chartWrap.classList.remove("hidden");
  }

  function onPly(p) {
    const plyData = p.ply;
    plyResults[plyData.ply - 1] = p;
    plyStatus.textContent = `${p.done} / ${p.total} plies`;
    updateMoveVol(plyData.ply - 1, plyData.volatility.score);
    appendChartPoint(plyData);
    jumpToPly(plyData.ply - 1);
  }

  function onDone(p) {
    gameStatus.textContent =
      `Done (${p.mode}) · ${p.plies_analysed} plies · ${p.total_analyses} engine calls`;
    plyStatus.textContent = "";
  }

  function onErr(p) {
    gameStatus.textContent = `Server error: ${p.message}`;
  }

  // ── Chart ────────────────────────────────────────────────────────────── //
  function ensureChart() {
    if (chart) return chart;

    Chart.defaults.color          = "#8a8a8a";
    Chart.defaults.borderColor    = "#262626";
    Chart.defaults.font.family    = "system-ui, sans-serif";
    Chart.defaults.font.size      = 11;

    const ctx = chartCanvas.getContext("2d");

    const volGrad = ctx.createLinearGradient(0, 0, 0, 220);
    volGrad.addColorStop(0,   "rgba(224, 68, 58, 0.38)");
    volGrad.addColorStop(0.45,"rgba(224, 179, 48, 0.22)");
    volGrad.addColorStop(1,   "rgba(57, 255, 20, 0.10)");

    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Volatility",
            data: [],
            yAxisID: "yV",
            borderColor: "#39ff14",
            backgroundColor: volGrad,
            borderWidth: 1.8,
            pointRadius: 2,
            pointHoverRadius: 5,
            pointBackgroundColor: "#39ff14",
            tension: 0.3,
            fill: true,
            spanGaps: true,
          },
          {
            label: "Eval (white, cp)",
            data: [],
            yAxisID: "yE",
            borderColor: "#6aa3ff",
            backgroundColor: "rgba(106, 163, 255, 0.1)",
            borderWidth: 1.5,
            pointRadius: 2,
            pointHoverRadius: 4,
            pointBackgroundColor: "#6aa3ff",
            tension: 0.3,
            fill: false,
            spanGaps: true,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           { duration: 0 },
        interaction:         { mode: "index", intersect: false },
        onClick: (_evt, elements) => {
          if (elements && elements.length) jumpToPly(elements[0].index);
        },
        plugins: {
          legend: {
            labels: { boxWidth: 12, padding: 14, color: "#a0a0a0" },
          },
          tooltip: {
            backgroundColor: "#141414",
            borderColor:     "#39ff14",
            borderWidth:     1,
            titleColor:      "#ececec",
            bodyColor:       "#a0a0a0",
            padding:         10,
            callbacks: {
              label: (ctx) =>
                `${ctx.dataset.label}: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(1) : "—"}`,
            },
          },
        },
        scales: {
          yV: {
            type: "linear",
            position: "left",
            min: 0, max: 100,
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: { color: "#39ff14", stepSize: 25 },
            title: { display: true, text: "Volatility", color: "#39ff14", font: { size: 10 } },
          },
          yE: {
            type: "linear",
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: { color: "#6aa3ff" },
            title: { display: true, text: "Eval (cp)", color: "#6aa3ff", font: { size: 10 } },
          },
          x: {
            grid:  { color: "rgba(255,255,255,0.04)" },
            ticks: {
              color: "#6a6a6a",
              maxRotation: 45,
              autoSkip: true,
              maxTicksLimit: 20,
            },
          },
        },
      },
    });
    return chart;
  }

  function appendChartPoint(plyJson) {
    ensureChart();
    const label = `${plyJson.ply}. ${plyJson.san}`;
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(plyJson.volatility.score ?? null);
    const turn   = plyJson.fen_before.split(/\s+/)[1] || "w";
    const cpWhite = turn === "b" ? -plyJson.eval_cp : plyJson.eval_cp;
    chart.data.datasets[1].data.push(cpWhite);
    chart.update("none");
  }

  // ── Bootstrap ────────────────────────────────────────────────────────── //
  try {
    setTab("editor");
    syncFenFromBoard();
    scheduleAutoAnalyze();
  } catch (err) {
    const msg = `Startup failed: ${err && err.message ? err.message : err}`;
    if (editorStatus) editorStatus.textContent = msg;
    console.error(msg, err);
  }
})();
