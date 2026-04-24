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
  const btnFlipGame   = $("#btnFlipGame");
  const gameStatus    = $("#gameStatus");
  const plyStatus     = $("#plyStatus");
  const moveListEl    = $("#moveList");
  const chartWrap     = $("#chartWrap");
  const moveListWrap  = $("#moveListWrap");
  const chartCanvas   = $("#chart");

  const arrowToggle      = $("#arrowToggle");
  const arrowToggleGame  = $("#arrowToggleGame");
  const arrowLayer       = $("#arrowLayer");
  const topLinesList     = $("#topLinesList");
  const topLinesListGame = $("#topLinesListGame");
  const boardFrameEl     = document.querySelector(".board-frame");
  const SVG_NS           = "http://www.w3.org/2000/svg";

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

  // ── Shared arrow toggle state ─────────────────────────────────────────── //
  function syncArrow(source) {
    const val = source.checked;
    if (arrowToggle && arrowToggle !== source) arrowToggle.checked = val;
    if (arrowToggleGame && arrowToggleGame !== source) arrowToggleGame.checked = val;
    refreshArrow();
  }
  if (arrowToggle)     arrowToggle.addEventListener("change",     () => syncArrow(arrowToggle));
  if (arrowToggleGame) arrowToggleGame.addEventListener("change", () => syncArrow(arrowToggleGame));

  function arrowEnabled() {
    return !!(arrowToggle && arrowToggle.checked);
  }

  // ── Board ─────────────────────────────────────────────────────────────── //
  let suppressSync = false;

  // Hoisted shared state for the analyze/invalidate helpers below. These are
  // consumed by scheduleAutoAnalyze() and analyzeFen() later in the module.
  let autoTimer   = null;
  let inflightFen = null;

  // Single source of truth for "the board changed; any pending or in-flight
  // analysis is now stale". Cancels the debounce timer, aborts the current
  // fetch (so a late response cannot overwrite the cleared UI), and wipes the
  // engine-lines panel + the top-move arrow.
  function invalidateAnalysis() {
    if (autoTimer) {
      clearTimeout(autoTimer);
      autoTimer = null;
    }
    if (inflightFen) {
      try { inflightFen.abort(); } catch (_) { /* ignore */ }
      inflightFen = null;
    }
    setTopMove(null);
    clearTopLinesLists();
  }

  const board = Chessboard("board", {
    draggable:    true,
    sparePieces:  true,
    dropOffBoard: "trash",
    position:     "start",
    pieceTheme:   "/vendor/img/pieces/{piece}.png",
    // NOTE: we intentionally do NOT auto-flip the side-to-move on drop. The
    // explicit White/Black segmented control owns turn state. Auto-flipping
    // conflated "set up pieces" with "play a move" and was a frequent source
    // of the engine being fed a position that disagrees with the board.
    onChange: () => {
      if (suppressSync) return;
      syncFenFromBoard();
      scheduleAutoAnalyze();
    },
  });

  window.addEventListener("resize", () => {
    board.resize();
    refreshArrow();
  });

  // Chess.com-style keyboard navigation through PGN plies. Runs on the game
  // tab only and yields to any text input so PGN/FEN paste & edit still work.
  window.addEventListener("keydown", (e) => {
    if (!loadedPlies.length) return;

    const gameTabActive = document
      .querySelector(".tab[data-tab='game']")
      ?.classList.contains("active");
    if (!gameTabActive) return;

    const t = e.target;
    const tag = t && t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || (t && t.isContentEditable)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const last = loadedPlies.length - 1;
    let next = null;
    switch (e.key) {
      case "ArrowRight": next = Math.min(last, (currentPlyIdx < 0 ? -1 : currentPlyIdx) + 1); break;
      case "ArrowLeft":  next = Math.max(0,    (currentPlyIdx < 0 ?  1 : currentPlyIdx) - 1); break;
      case "Home":       next = 0; break;
      case "End":        next = last; break;
      default: return;
    }
    e.preventDefault();
    if (next !== currentPlyIdx) jumpToPly(next);
  });

  // Expand a FEN rank (e.g. "r3k2r" or "4P3") into an 8-char string where
  // empty squares are "." This lets us index the rank by file (a=0…h=7).
  function expandRank(r) {
    let out = "";
    for (const ch of r) {
      if (ch >= "1" && ch <= "8") out += ".".repeat(ch.charCodeAt(0) - 48);
      else out += ch;
    }
    return out.padEnd(8, ".").slice(0, 8);
  }

  // Recompute castling rights from the current placement. A right survives
  // only if the king and the relevant rook are still on their home squares.
  // Any board edit that moves either piece cancels the matching right.
  function computeCastling(placement) {
    const ranks = placement.split("/");
    if (ranks.length !== 8) return "-";
    const rank1 = expandRank(ranks[7]);
    const rank8 = expandRank(ranks[0]);
    const whiteKingHome = rank1[4] === "K";
    const blackKingHome = rank8[4] === "k";
    let rights = "";
    if (whiteKingHome && rank1[7] === "R") rights += "K";
    if (whiteKingHome && rank1[0] === "R") rights += "Q";
    if (blackKingHome && rank8[7] === "r") rights += "k";
    if (blackKingHome && rank8[0] === "r") rights += "q";
    return rights || "-";
  }

  // Produce a fresh, self-consistent FEN from (placement on the board) +
  // (side-to-move from the previous FEN). We deliberately drop the inherited
  // en-passant square (any edit invalidates it) and reset the halfmove clock
  // to 0. Castling rights are recomputed from placement. This is the only
  // place the UI hands a FEN to the backend.
  function assembleFen() {
    const parts = (fenInput.value || "").trim().split(/\s+/);
    const placement = board.fen();
    const turn = parts[1] === "b" ? "b" : "w";
    const castling = computeCastling(placement);
    const fullmove = Math.max(1, parseInt(parts[5], 10) || 1);
    return `${placement} ${turn} ${castling} - 0 ${fullmove}`;
  }

  function syncFenFromBoard() {
    const fen = assembleFen();
    fenInput.value = fen;
    editorStatus.textContent = validateFen(fen) ? "" : "⚠ Incomplete or illegal position";
    syncTurnToggleFromFen();
    invalidateAnalysis();
  }

  function syncBoardFromFen(fen) {
    const parts = fen.trim().split(/\s+/);
    if (parts.length >= 1) {
      suppressSync = true;
      try { board.position(parts[0], false); } finally { suppressSync = false; }
    }
    // suppressSync swallowed onChange, so the shared cleanup path didn't run.
    // Do it explicitly — otherwise a FEN-paste edit leaves stale engine lines
    // and arrows on the screen until the next re-analysis lands.
    invalidateAnalysis();
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
    const placement = parts[0] || board.fen();
    const castling = computeCastling(placement);
    const fullmove = Math.max(1, parseInt(parts[5], 10) || 1);
    fenInput.value = `${placement} ${next} ${castling} - 0 ${fullmove}`;
    editorStatus.textContent = validateFen(fenInput.value) ? "" : "⚠ Incomplete or illegal position";
    // Any side-to-move change makes a prior analysis stale by definition.
    invalidateAnalysis();
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
  // `autoTimer` is declared near the top of the module alongside
  // `inflightFen` so that `invalidateAnalysis()` can own both.

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
    invalidateAnalysis();
    scheduleAutoAnalyze();
  });

  btnClear.addEventListener("click", () => {
    suppressSync = true;
    board.clear(false);
    suppressSync = false;
    syncFenFromBoard();
    scheduleAutoAnalyze();
  });

  btnFlip.addEventListener("click", () => {
    board.flip();
    setTimeout(refreshArrow, 0);
  });

  if (btnFlipGame) {
    btnFlipGame.addEventListener("click", () => {
      board.flip();
      setTimeout(refreshArrow, 0);
    });
  }

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

  // ── Arrow overlay ─────────────────────────────────────────────────────── //
  let lastTopMoveUci = null;

  function squareCenter(sq) {
    if (!boardFrameEl) return null;
    const el = document.querySelector(`#board .square-${sq}`);
    if (!el) return null;
    const frameRect = boardFrameEl.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    return {
      x: r.left - frameRect.left + r.width / 2,
      y: r.top  - frameRect.top  + r.height / 2,
      size: r.width,
    };
  }

  function clearArrow() {
    if (!arrowLayer) return;
    while (arrowLayer.firstChild) arrowLayer.removeChild(arrowLayer.firstChild);
  }

  function drawArrow(uci) {
    if (!arrowLayer || !uci || uci.length < 4) { clearArrow(); return; }
    const from = uci.slice(0, 2);
    const to   = uci.slice(2, 4);
    const a = squareCenter(from);
    const b = squareCenter(to);
    if (!a || !b) { clearArrow(); return; }

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) { clearArrow(); return; }
    const ux = dx / len;
    const uy = dy / len;

    const sq    = a.size;
    const w     = Math.max(6,  sq * 0.18);
    const head  = Math.max(10, sq * 0.34);
    const inset = sq * 0.22;

    const sx = a.x + ux * inset;
    const sy = a.y + uy * inset;
    const ex = b.x - ux * inset;
    const ey = b.y - uy * inset;

    const shaftEndX = ex - ux * head;
    const shaftEndY = ey - uy * head;

    const px =  uy;
    const py = -ux;

    const hw = w * 0.5;
    const shaft = document.createElementNS(SVG_NS, "polygon");
    shaft.setAttribute(
      "points",
      [
        `${sx + px * hw},${sy + py * hw}`,
        `${shaftEndX + px * hw},${shaftEndY + py * hw}`,
        `${shaftEndX - px * hw},${shaftEndY - py * hw}`,
        `${sx - px * hw},${sy - py * hw}`,
      ].join(" ")
    );
    shaft.setAttribute("class", "arrow-shaft");

    const hhw = w * 1.1;
    const headPoly = document.createElementNS(SVG_NS, "polygon");
    headPoly.setAttribute(
      "points",
      [
        `${ex},${ey}`,
        `${shaftEndX + px * hhw},${shaftEndY + py * hhw}`,
        `${shaftEndX - px * hhw},${shaftEndY - py * hhw}`,
      ].join(" ")
    );
    headPoly.setAttribute("class", "arrow-head");

    clearArrow();
    arrowLayer.appendChild(shaft);
    arrowLayer.appendChild(headPoly);
  }

  function refreshArrow() {
    if (!arrowEnabled() || !lastTopMoveUci) { clearArrow(); return; }
    drawArrow(lastTopMoveUci);
  }

  function setTopMove(uci) {
    lastTopMoveUci = uci || null;
    refreshArrow();
  }

  // ── Engine lines panel ────────────────────────────────────────────────── //
  function formatEvalSigned(cp, turn) {
    if (cp == null) return "—";
    const w = turn === "b" ? -cp : cp;
    if (Math.abs(w) >= 1000) return (w > 0 ? "+" : "−") + "M";
    const abs = (Math.abs(w) / 100).toFixed(2);
    return (w >= 0 ? "+" : "−") + abs;
  }

  function activeTopLinesEl() {
    const gameTabActive = document
      .querySelector(".tab[data-tab='game']")
      ?.classList.contains("active");
    return gameTabActive ? topLinesListGame : topLinesList;
  }

  function clearTopLinesLists() {
    [topLinesList, topLinesListGame].forEach((el) => {
      if (el) el.innerHTML = "";
    });
  }

  function renderTopLines(volJson, turn) {
    clearTopLinesLists();
    const target = activeTopLinesEl();
    if (!target) return;
    const lines = (volJson && volJson.top_lines) || [];
    if (!lines.length) return;

    lines.forEach((line, idx) => {
      const li = document.createElement("li");
      li.className = "top-line" + (idx === 0 ? " best" : "");

      const evalSpan = document.createElement("span");
      evalSpan.className = "top-line-eval";
      evalSpan.textContent = formatEvalSigned(line.eval_cp, turn);
      const w = turn === "b" ? -line.eval_cp : line.eval_cp;
      evalSpan.dataset.sign = w > 30 ? "pos" : w < -30 ? "neg" : "neutral";

      const pvSpan = document.createElement("span");
      pvSpan.className = "top-line-pv";
      const pv = Array.isArray(line.pv_san) ? line.pv_san.slice(0, 6) : [line.san];
      pvSpan.textContent = pv.join(" ");
      pvSpan.title = Array.isArray(line.pv_san) ? line.pv_san.join(" ") : line.san;

      li.appendChild(evalSpan);
      li.appendChild(pvSpan);
      target.appendChild(li);
    });
  }

  // ── Analyze FEN ──────────────────────────────────────────────────────── //
  // `inflightFen` is hoisted near the top so invalidateAnalysis() can abort it.

  async function analyzeFen(fen) {
    if (inflightFen) {
      try { inflightFen.abort(); } catch (_) { /* ignore */ }
    }
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
      // Stale-response guard: if the user edited the board (or a newer
      // analysis started) while this request was in flight, our controller
      // was replaced in inflightFen and/or aborted. Do not let this response
      // overwrite the cleared UI with lines that belong to an older FEN.
      if (ctrl !== inflightFen || ctrl.signal.aborted) return;
      const turn = fen.trim().split(/\s+/)[1] || "w";
      renderEvalBar(data.volatility.best_eval_cp, turn);
      renderVolBar(data.volatility);
      renderTopLines(data.volatility, turn);
      const topUci = data.volatility.top_lines && data.volatility.top_lines[0]
        ? data.volatility.top_lines[0].uci
        : null;
      setTopMove(topUci);
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
  let currentPlyIdx = -1;
  let chart         = null;
  let pgnController = null;

  function resetGame() {
    loadedPlies = [];
    plyResults  = [];
    currentPlyIdx = -1;
    moveListEl.innerHTML = "";
    gameStatus.textContent  = "";
    plyStatus.textContent   = "";
    chartWrap.classList.add("hidden");
    moveListWrap.classList.add("hidden");
    clearTopLinesLists();
    setTopMove(null);
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
    currentPlyIdx = idx;
    const entry = loadedPlies[idx];

    suppressSync = true;
    try { board.position(entry.fen_before.split(/\s+/)[0], true); }
    finally { suppressSync = false; }

    fenInput.value = entry.fen_before;
    syncTurnToggleFromFen();

    const r = plyResults[idx];
    if (r) {
      const turn = entry.fen_before.split(/\s+/)[1] || "w";
      renderEvalBar(r.ply.volatility.best_eval_cp, turn);
      renderVolBar(r.ply.volatility);
      renderTopLines(r.ply.volatility, turn);
      const tl = r.ply.volatility.top_lines;
      setTopMove(tl && tl[0] ? tl[0].uci : null);
    } else {
      clearTopLinesLists();
      setTopMove(null);
    }

    document.querySelectorAll(".move-cell").forEach((c) =>
      c.classList.toggle("active", Number(c.dataset.idx) === idx)
    );

    const active = moveListEl.querySelector(".move-cell.active");
    if (active && moveListWrap && moveListWrap.contains(active)) {
      // Scroll only within the move-list's own container (.move-list-wrap is
      // overflow-y:auto with max-height). Using scrollIntoView() would also
      // scroll the window, which pushes the board off-screen when arrow-keying
      // through a game.
      const cRect = moveListWrap.getBoundingClientRect();
      const aRect = active.getBoundingClientRect();
      if (aRect.top < cRect.top) {
        moveListWrap.scrollTop += aRect.top - cRect.top;
      } else if (aRect.bottom > cRect.bottom) {
        moveListWrap.scrollTop += aRect.bottom - cRect.bottom;
      }
    }

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
