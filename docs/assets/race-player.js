/*
 * Ghost-race player shared by the Race and Championship tabs.
 *
 * CarRacing simulates one car at a time, so a race replays each racer's recorded
 * benchmark drive on the same track, in sync.
 *
 *   var player = RacePlayer({
 *     prefix: "race",                 // element ids: <prefix>Canvas, Clock, ClockOf, Timebar, Board, Scrub, Start, Reset, Speed, Cam
 *     fps: 50,
 *     racer:  function (id) { return { label, color }; },
 *     run:    function (id, seed) { return recording; },   // {every, finish, x, y, a, v, tiles, grass}
 *     result: function (id, seed) { return benchmarkResult; } // {frames, finished, lap_s, reached_pct, ...}
 *   });
 *   player.setTrack(track); player.setRacers(["neat16", "ppo16"]);
 */
(function () {
  "use strict";

  window.RacePlayer = function (opt) {
    var Site = window.Site, css = Site.css;
    function $(name) { return document.getElementById(opt.prefix + name); }
    var S = { track: null, ids: [], frame: 0, playing: false, speed: 1, cam: "whole", last: 0 };

    function run(id, track) { return opt.run(id, (track || S.track).seed); }
    function endFrame(id, track) { return opt.result(id, (track || S.track).seed).frames; }
    function raceLength() { return Math.max.apply(null, [1].concat(S.ids.map(function (id) { return endFrame(id); }))); }
    function fmt(frames) { return Site.clock(frames, opt.fps); }

    /** Tiles a racer never touched on a track (a lap needs every tile). */
    function missed(id, track) { var r = run(id, track); return (track || S.track).tiles - r.tiles[r.tiles.length - 1]; }
    function dnfText(id, track) {
      var m = missed(id, track);
      return m <= 3 ? "DNF · missed " + m + " tile" + (m === 1 ? "" : "s") : "DNF";
    }
    /** Coverage without rounding a near-miss up to 100%. */
    function covered(pct) { return pct >= 99.5 && pct < 100 ? pct.toFixed(1) + "%" : Math.round(pct) + "%"; }

    /** Interpolated state of one racer at (possibly fractional) game frame f. */
    function at(id, f, track) {
      var r = run(id, track), n = r.x.length, i = f / r.every, i0 = Math.min(n - 1, Math.floor(i)), i1 = Math.min(n - 1, i0 + 1), t = Math.min(1, i - i0);
      if (i >= n - 1) { i0 = i1 = n - 1; t = 0; }
      var da = r.a[i1] - r.a[i0]; if (da > Math.PI) da -= 2 * Math.PI; if (da < -Math.PI) da += 2 * Math.PI;
      var done = r.finish != null && f >= r.finish, tiles = (track || S.track).tiles;
      return { x: r.x[i0] + (r.x[i1] - r.x[i0]) * t, y: r.y[i0] + (r.y[i1] - r.y[i0]) * t, a: r.a[i0] + da * t,
        v: done ? 0 : r.v[i0], tiles: done ? tiles : r.tiles[i0], grass: r.grass ? r.grass[i0] : 0, done: done, finish: r.finish };
    }
    /** Order at frame f: finished laps by time, then most tiles reached. */
    function standings(f, track, ids) {
      return (ids || S.ids).map(function (id) { return { id: id, s: at(id, f, track) }; }).sort(function (a, b) {
        var fa = a.s.done ? a.s.finish : Infinity, fb = b.s.done ? b.s.finish : Infinity;
        if (fa !== fb) return fa - fb;
        return b.s.tiles - a.s.tiles;
      });
    }

    function draw() {
      var T = S.track, cv = $("Canvas"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
      var b = T.bounds, cx, cy, scale, f = S.frame, board = standings(f);
      if (S.cam === "leader" && board.length) { cx = board[0].s.x; cy = board[0].s.y; scale = W / 230; }
      else { cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2; scale = Math.min(W / (b[2] - b[0]), H / (b[3] - b[1])) * 0.9; }
      function P(x, y) { return [W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale]; }
      ctx.fillStyle = css("--grass"); ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = css("--road");
      T.polys.forEach(function (p) {
        ctx.beginPath();
        for (var i = 0; i < p.length; i += 2) { var q = P(p[i], p[i + 1]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); }
        ctx.closePath(); ctx.fill();
      });
      var st = P(T.start[0], T.start[1]);
      ctx.save(); ctx.translate(st[0], st[1]); ctx.rotate(-T.start_angle);
      var sq = Math.max(2, 7 * scale / 4);
      for (var k = -4; k < 4; k++) for (var j = 0; j < 2; j++) { ctx.fillStyle = (k + j) % 2 ? "#ffffff" : "#111111"; ctx.fillRect(k * sq, (j - 1) * sq, sq, sq); }
      ctx.restore();
      var order = board.slice().reverse();
      order.forEach(function (e) {
        var r = run(e.id), end = Math.min(r.x.length - 1, Math.floor(f / r.every)), start = Math.max(0, end - 40);
        ctx.strokeStyle = opt.racer(e.id).color; ctx.lineWidth = 3; ctx.globalAlpha = 0.45; ctx.lineJoin = "round";
        ctx.beginPath();
        for (var i = start; i <= end; i++) { var q = P(r.x[i], r.y[i]); if (i > start) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); }
        var c = P(e.s.x, e.s.y); ctx.lineTo(c[0], c[1]); ctx.stroke(); ctx.globalAlpha = 1;
      });
      var placed = [];
      order.forEach(function (e) {
        var info = opt.racer(e.id), c = P(e.s.x, e.s.y), a = e.s.a, size = Math.max(9, 5 * scale);
        var fx = -Math.sin(a), fy = Math.cos(a), rx = Math.cos(a), ry = Math.sin(a);
        var shape = [[0, 1], [0.55, -0.7], [0, -0.35], [-0.55, -0.7]].map(function (p) { return [c[0] + (rx * p[0] + fx * p[1]) * size, c[1] - (ry * p[0] + fy * p[1]) * size]; });
        ctx.fillStyle = info.color; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
        ctx.beginPath(); shape.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.font = "600 15px Inter, system-ui, sans-serif";
        var label = info.label + (e.s.done ? " 🏁" : ""), w = ctx.measureText(label).width + 12, h = 22;
        var lx = c[0] + 12, ly = c[1] - 30;
        for (var tries = 0; tries < 8; tries++) {
          if (!placed.some(function (p) { return lx < p[0] + p[2] && lx + w > p[0] && ly < p[1] + p[3] && ly + h > p[1]; })) break;
          ly += h + 2;
        }
        placed.push([lx, ly, w, h]);
        ctx.fillStyle = css("--surface"); ctx.globalAlpha = 0.92; ctx.beginPath(); ctx.roundRect(lx, ly, w, h, 6); ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = info.color; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = css("--text"); ctx.fillText(label, lx + 6, ly + 16);
      });
      drawBoard(board);
    }

    function drawBoard(board) {
      var f = S.frame, len = raceLength(), T = S.track;
      $("Clock").textContent = fmt(f);
      $("ClockOf").textContent = "of " + fmt(len);
      $("Timebar").style.width = (100 * f / len) + "%";
      var scrub = $("Scrub"); scrub.max = len; scrub.value = Math.round(f);
      $("Board").innerHTML = board.map(function (e, i) {
        var info = opt.racer(e.id), pct = 100 * e.s.tiles / T.tiles, st;
        if (e.s.done) st = '<span class="flag">🏁 ' + fmt(e.s.finish) + "</span>";
        else if (f >= endFrame(e.id)) st = '<span class="warn">' + dnfText(e.id) + "</span>";
        else if (f > 0 && e.s.v < 2) st = '<span class="warn">stopped</span>';
        else if (e.s.grass >= 2) st = '<span class="warn">on grass · ' + Math.round(e.s.v) + "</span>";
        else st = Math.round(e.s.v) + " speed";
        return '<li><span class="pos">' + (i + 1) + '</span><span class="dot" style="background:' + info.color + '"></span>' +
          '<span class="name">' + Site.esc(info.label) + '</span><span class="stat">' + Math.round(pct) + "% · " + st + "</span>" +
          '<span class="prog"><b style="width:' + pct + "%;background:" + info.color + '"></b></span></li>';
      }).join("") || '<li><span></span><span></span><span class="muted">Pick at least one driver</span></li>';
    }

    function done() { return S.frame >= raceLength(); }
    function label() { $("Start").textContent = S.playing ? "❚❚ Pause" : (S.frame > 0 && !done() ? "▶ Resume" : "▶ Start race"); }
    function tick(ts) {
      if (!S.playing) return;
      var dt = S.last ? ts - S.last : 0; S.last = ts;
      S.frame = Math.min(raceLength(), S.frame + dt / 1000 * opt.fps * S.speed);
      draw();
      if (done()) { S.playing = false; label(); if (opt.onFinish) opt.onFinish(); return; }
      requestAnimationFrame(tick);
    }
    function stop() { S.playing = false; label(); }

    $("Start").onclick = function () {
      if (!S.playing && done()) S.frame = 0;
      S.playing = !S.playing; S.last = 0; label(); if (S.playing) requestAnimationFrame(tick);
    };
    $("Reset").onclick = function () { S.frame = 0; stop(); draw(); };
    $("Scrub").oninput = function (e) { S.frame = Number(e.target.value); draw(); label(); };
    [].forEach.call($("Speed").querySelectorAll("button"), function (b) {
      b.onclick = function () { S.speed = Number(b.dataset.s); [].forEach.call(b.parentNode.children, function (c) { c.className = c === b ? "on" : ""; }); };
    });
    [].forEach.call($("Cam").querySelectorAll("button"), function (b) {
      b.onclick = function () { S.cam = b.dataset.c; [].forEach.call(b.parentNode.children, function (c) { c.className = c === b ? "on" : ""; }); draw(); };
    });

    return {
      setTrack: function (track) { S.track = track; S.frame = 0; stop(); draw(); },
      setRacers: function (ids) {
        S.ids = ids.slice();
        if (!S.track) return;  // drawn once a track is set
        S.frame = Math.min(S.frame, raceLength()); label(); draw();
      },
      racers: function () { return S.ids.slice(); },
      redraw: draw,
      standings: standings,
      dnfText: dnfText,
      covered: covered
    };
  };
})();
