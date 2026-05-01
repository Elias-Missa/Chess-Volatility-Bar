Swap to Lichess Board + Sounds
Goal
Replace the current frontend board (chessboard.js) with Chessground — lichess.org's actual board UI. Add lichess's piece set, board theme, and sound effects. The board should look and feel identical to lichess.org's analysis page.
Backend stays untouched. All Python code, the API, the volatility algorithm, the explain panel, calibration, move classification — none of it changes. This is a frontend-only swap.
Source Repos
Both public, both clonable.

Chessground (board UI): https://github.com/lichess-org/chessground

npm: @lichess-org/chessground
Demo to learn the API from: https://github.com/lichess-org/chessground/blob/master/demo.html
Base CSS: https://github.com/lichess-org/chessground-examples/blob/master/assets/chessground.css


Lila (lichess monorepo — for assets): https://github.com/lichess-org/lila

Pieces: public/piece/cburnett/ (12 SVG files: wK.svg, wQ.svg, wR.svg, wB.svg, wN.svg, wP.svg, and the same six for black)
Sounds: public/sound/standard/ (use Move.mp3, Capture.mp3, Check.mp3, GenericNotify.mp3 plus their .ogg versions)
Board square images (optional, lichess uses CSS gradients for most): public/images/board/



Lila is large — don't clone the whole thing. Use git clone --depth 1 --filter=blob:none --sparse https://github.com/lichess-org/lila.git and then git sparse-checkout set public/piece/cburnett public/sound/standard. Or just fetch the specific files via raw GitHub URLs with curl.
Tasks
1. Add a build step
Current frontend is no-build vanilla JS. Chessground is npm + TypeScript and needs a bundler. Add Vite.

In the project root (or wherever the frontend lives — likely web/), npm init -y.
Install: @lichess-org/chessground, chess.js, vite (dev), typescript (dev).
Configure vite.config.ts: build output goes wherever the FastAPI server currently expects static files (web/ or similar). Add a dev proxy for /analyze/* and /healthz → http://127.0.0.1:8000 so dev mode hits the running backend.
Add scripts: dev, build, preview.
Update FastAPI's static file mount if the build output path changes.

Verify: npm run build produces a bundle, chess-vol serve still loads the (currently old) UI through the new pipeline.
2. Pull in lichess assets

Fetch the 12 Cburnett piece SVGs into web/public/piece/cburnett/ (or wherever Vite serves static assets).
Fetch the 4 sound files (mp3 + ogg pairs) into web/public/sound/.
Fetch chessground.css from the chessground-examples repo and add it to your styles.
For the board, Chessground works with CSS-only theming. Use the green/cream lichess default — see lila/public/styles/board.css for reference, or use the simple background-color approach in chessground-examples.

3. Replace the board

Delete chessboard.js and any related vendored files. Keep chess.js — Chessground has no chess logic, you still need it for move legality and FEN reconstruction.
Convert app.js to TypeScript (main.ts).
Replace the board element with a Chessground instance. Read the demo for the API shape.
Position editor tab: configure draggable.enabled: true, draggable.deleteOnDropOff: true, selectable.enabled: true, movable.free: true, movable.color: 'both'. Two-way bind FEN ↔ board: when the textarea changes, call cg.set({ fen }); when a piece moves on the board, recompute the FEN with chess.js and update the textarea. Wire spare-piece drag-from-tray.
Game (PGN) tab: reconfigure the same Chessground instance: viewOnly: true, movable.free: false. On move-list click or chart scrub, call cg.set({ fen, lastMove: [from, to] }). The lastMove highlight is critical to the lichess feel.
Enable drawable.enabled: true so right-click-drag draws SVG arrows on the board (lichess feature, comes free with Chessground).
Animations: Chessground default is 200ms piece slide. Keep it.
Coordinates: show a–h / 1–8 around the board (Chessground option, default on).

4. Eval bar + volatility bar
Keep your existing eval bar and vol bar components. They're correct and tied to your algorithm. Just make sure their height matches Chessground's .cg-wrap container so they line up visually next to the board. Restyle the surrounding chrome (cards, fonts, colors) to match the lichess aesthetic — clean, muted, system fonts.
5. Sound effects
Small module audio.ts:
tsclass AudioManager {
  play(name: 'move' | 'capture' | 'check' | 'notify'): void;
  setEnabled(on: boolean): void;
  setVolume(v: number): void;  // 0–1
}

Preload all sounds on init. Reuse HTMLAudioElement instances (don't create per play).
Persist enabled state and volume to localStorage.
Default: enabled, volume 0.6.
Editor mode: on every piece placement, play move, or capture if a piece was taken, or check if the resulting position is check (use chess.js to detect).
PGN viewer: play the same sounds when scrubbing forward through moves. Do not play when scrubbing backward — matches lichess behavior.
UI: speaker-icon toggle button + volume slider in the page header.

6. Verify

All existing Python tests still pass with zero modifications.
All existing API behavior unchanged (test by hitting endpoints with curl).
Position editor: drag pieces, edit FEN, click Analyze — vol bar updates correctly.
PGN viewer: paste a game, watch SSE stream populate the chart, click moves to scrub, board updates with lastMove highlight, sounds play on forward scrub.
Explain panel still renders, pattern badges still work, all classifications still display.
Right-click-drag draws arrows. Animations are smooth. Pieces look like lichess.

What Stays Exactly the Same

All of src/chess_vol/ (Python).
All API endpoints, request/response schemas, SSE event sequence.
Volatility algorithm, mate handling, eval-aware scaling, decided flag, recursion.
Calibration pipeline, explain panel logic, move classification logic.
Tests in tests/.