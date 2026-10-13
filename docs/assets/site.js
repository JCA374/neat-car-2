/* Shared helpers for the project pages: theme, formatting, colours, the step ladder. */
(function () {
  "use strict";
  var root = document.documentElement;
  try { var saved = localStorage.getItem("theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}

  var redraws = [];
  var Site = {
    css: function (name) { return getComputedStyle(root).getPropertyValue(name).trim(); },
    onTheme: function (fn) { redraws.push(fn); },
    json: function (path) {
      return fetch(path).then(function (r) { if (!r.ok) throw new Error(path + ": " + r.status); return r.json(); });
    },
    esc: function (s) {
      return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
    },
    fmt: function (v, d) { return v == null || isNaN(v) ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d == null ? 1 : d }); },
    /** 3734 -> "1 h 02 min", 600 -> "10 min" */
    dur: function (seconds) {
      if (seconds == null) return "–";
      var m = Math.round(seconds / 60);
      if (m < 60) return m + " min";
      return Math.floor(m / 60) + " h " + String(m % 60).padStart(2, "0") + " min";
    },
    lap: function (s) { return s == null ? "–" : s.toFixed(2) + " s"; },
    clock: function (frames, fps) { var s = frames / fps, m = Math.floor(s / 60); return m + ":" + (s - m * 60).toFixed(2).padStart(5, "0"); },
    modelName: function (m) { return m === "neat" ? "NEAT" : "PPO"; },
    modelColor: function (m) { return Site.css(m === "neat" ? "--neat" : "--ppo"); },
    /** Fixed categorical colour per step, from its id (n00 blue, p00 orange, n01, p01, ...), never from rank. */
    stepColor: function (id) {
      var n = parseInt(id.slice(1, 3), 10) || 0, ppo = id.charAt(0) === "p" ? 1 : 0;
      return Site.css("--c" + (((2 * n + ppo) % 8) + 1));
    },
    short: function (step) { return step.id.split("_")[0].toUpperCase(); },
    label: function (step) { return Site.short(step) + " · " + step.title; },
    chartTheme: function () {
      return {
        grid: { color: Site.css("--grid") }, border: { display: false },
        ticks: { color: Site.css("--text-3") }
      };
    },
    tooltipTheme: function () {
      return {
        backgroundColor: Site.css("--surface"), titleColor: Site.css("--text"), bodyColor: Site.css("--text-2"),
        borderColor: Site.css("--border"), borderWidth: 1, padding: 10, boxPadding: 4
      };
    }
  };

  /** Lap time delta vs parent, as an HTML snippet. Negative = faster = better. */
  function delta(v, ref, unit, lowerIsBetter, digits) {
    if (v == null || ref == null) return "";
    var d = v - ref, eps = Math.pow(10, -(digits || 0)) / 2;
    if (Math.abs(d) < eps) return '<div class="delta">same as parent</div>';
    var better = lowerIsBetter ? d < 0 : d > 0;
    return '<div class="delta ' + (better ? "better" : "worse") + '">' + (d > 0 ? "+" : "−") + Math.abs(d).toFixed(digits || 0) + unit + " vs parent</div>";
  }

  function settingChips(step) {
    var chips = [];
    Object.keys(step.train || {}).forEach(function (k) {
      var v = step.train[k];
      chips.push(v === true ? k : k + "=" + (Array.isArray(v) ? v.join(",") : v));
    });
    Object.keys(step.neat_config || {}).forEach(function (k) { chips.push(k.split(".")[1] + "=" + step.neat_config[k]); });
    if (step.lidar_fov != null) chips.unshift("lidar-fov=±" + step.lidar_fov / 2 + "°");
    if (step.lidar_rays !== 32) chips.unshift("lidar-rays=" + step.lidar_rays);
    return chips.map(function (c) { return "<span>" + Site.esc(c) + "</span>"; }).join("");
  }

  /**
   * Render the ladder of one model's steps into `ol`.
   * `steps`: leaderboard entries for that model; `pending`: step ids defined but not run yet.
   */
  Site.ladder = function (ol, model, board, pending, onPick) {
    var steps = board.steps.filter(function (s) { return s.model === model; }).sort(function (a, b) { return a.id < b.id ? -1 : 1; });
    var byId = {}; board.steps.forEach(function (s) { byId[s.id] = s; });
    ol.style.setProperty("--model", Site.modelColor(model));
    var html = steps.map(function (s) {
      var p = s.parent ? byId[s.parent] : null, sum = s.summary, champ = board.champions[model] === s.id;
      var laps = s.tracks.map(function (t) {
        return t.finished ? '<span title="Track ' + t.seed + '">T' + t.seed + " " + t.lap_s.toFixed(1) + " s</span>"
          : '<span class="dnf" title="Track ' + t.seed + ": " + t.reached_pct + '% of the lap">T' + t.seed + " DNF</span>";
      }).join("");
      return '<li class="' + (champ ? "champion" : "") + '" id="' + s.id + '"><div class="card"><div class="grid"><div>' +
        '<p class="meta"><span>' + Site.short(s) + "</span><span>trained " + Site.esc(s.trained_on) + "</span>" +
        (s.commit ? "<span>" + Site.esc(s.commit) + "</span>" : "") + (p ? "<span>builds on " + Site.short(p) + "</span>" : "") + "</p>" +
        "<h3>" + Site.esc(s.title) + (champ ? '<span class="tag champ-tag">' + Site.modelName(model) + " champion</span>" : "") +
        (s.imported ? '<span class="tag">imported</span>' : "") + "</h3>" +
        '<p class="muted small" style="margin:6px 0 0">' + Site.esc(s.why) + "</p>" +
        '<div class="tiles">' +
        '<div class="tile"><div class="num">' + sum.finished + '/5</div><div class="lbl">tracks finished</div>' + delta(sum.finished, p && p.summary.finished, "", false) + "</div>" +
        '<div class="tile"><div class="num">' + Site.lap(sum.lap_s) + '</div><div class="lbl">mean lap</div>' + delta(sum.lap_s, p && p.summary.lap_s, " s", true, 2) + "</div>" +
        '<div class="tile"><div class="num">' + Site.dur(s.train_seconds) + '</div><div class="lbl">training, no GPU</div>' + delta(s.train_seconds / 60, p && p.train_seconds / 60, " min", true) + "</div>" +
        '<div class="tile"><div class="num">' + Site.fmt(s.size.params, 0) + '</div><div class="lbl">parameters · ' + Site.esc(s.size.label) + "</div>" + delta(s.size.params, p && p.size.params, "", true) + "</div>" +
        "</div>" +
        '<div class="laps">' + laps + "</div>" +
        (s.outcome ? '<div class="outcome"><strong>Outcome:</strong> ' + Site.esc(s.outcome) + "</div>" : "") +
        '<div class="kv">' + settingChips(s) + "</div>" +
        (s.note ? '<p class="small faint" style="margin:8px 0 0">' + Site.esc(s.note) + "</p>" : "") +
        "</div><figure style=\"margin:0\"><img src=\"assets/gifs/" + s.id + '.gif" alt="' + Site.esc(s.id) + ' driving benchmark track 3" loading="lazy">' +
        '<div class="row" style="margin:8px 0 0">' +
        (onPick ? '<button class="btn ghost sm" type="button" data-pick="' + s.id + '">Inspect ↓</button>' : "") +
        '<a class="btn ghost sm" style="text-decoration:none" href="race.html?drivers=' + s.id + "," + (board.champions[model === "neat" ? "ppo" : "neat"] || "") + '">Race it</a></div>' +
        "</figure></div></div></li>";
    }).join("");
    html += (pending || []).map(function (p) {
      return '<li><div class="card pending"><p class="meta"><span>' + Site.esc(p.id.split("_")[0].toUpperCase()) + "</span><span>not trained yet</span></p>" +
        "<h3>" + Site.esc(p.title) + "</h3><p class=\"small\" style=\"margin:4px 0 0\">Run <code>python scripts/run_step.py experiments/" + model + "/" + p.id + ".toml</code></p></div></li>";
    }).join("");
    ol.innerHTML = html;
    if (onPick) [].forEach.call(ol.querySelectorAll("[data-pick]"), function (b) { b.onclick = function () { onPick(b.dataset.pick); }; });
  };

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var dark = getComputedStyle(root).colorScheme.indexOf("dark") !== -1, next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      redraws.forEach(function (fn) { fn(); });
    });
  });
  window.Site = Site;
})();
