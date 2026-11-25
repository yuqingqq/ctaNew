// DOM check for flow.html without a browser (Chromium cannot start under this box's 12 GiB address-space limit, which is not raised).
// Needs jsdom and @resvg/resvg-js; pass their node_modules directory as TOOLS. Usage: TOOLS=<dir>/node_modules node check_dom.cjs <page> [pngdir]
// Runs the page's own script at two widths, clicks every market / take / zoom / row, hovers the chart, and fails (exit 1) on a script error,
// an empty chart or table, a missing tooltip, NaN / undefined / Infinity anywhere in the rendered text or SVG, or a FAIL in the footer.
// With pngdir it rasterises the chart SVG (light and dark CSS resolved) for visual inspection.
// Falsifiers (run them): a copy that selects a market that does not exist, and a copy whose table reads a missing field, must both fail.
const path = require('path'), fs = require('fs');
const TOOLS = process.env.TOOLS;
const {JSDOM, VirtualConsole} = require(path.join(TOOLS, 'jsdom'));
const {Resvg} = require(path.join(TOOLS, '@resvg/resvg-js'));
const file = path.resolve(process.argv[2]), pngdir = process.argv[3];
const html = fs.readFileSync(file, 'utf8');
const errors = [], checks = [];
const BAD = /NaN|undefined|Infinity|\[object Object\]/;   // no \b: textContent joins table cells without spaces

function themeCss(dark) {
  const block = html.match(dark ? /:root\[data-theme="dark"\] \.viz-root\{([^}]*)\}/ : /\.viz-root\{(color-scheme:light;[^}]*)\}/)[1];
  const vars = {}; for (const m of block.matchAll(/--([\w-]+):([^;]+)/g)) vars[m[1]] = m[2].trim();
  const css = [...html.matchAll(/^\.(grid|axis|ln|ln-bn|ln-bid|ln-ask|ln-ours|thr|tk|halo|ttl|mk-fill|mk-hollow|ord|ref|trd|cross|selln)\{[^}]*\}/gm)].map(m => m[0]).join('\n');
  return {bg: vars['surface-1'], css: css.replace(/var\(--([\w-]+)\)/g, (_, k) => vars[k])};
}
function png(svg, name, dark) {
  const t = themeCss(dark);
  const s = svg.replace(/^<svg([^>]*)>/, (m, a) => `<svg xmlns="http://www.w3.org/2000/svg"${a}><style>${t.css}\ntext{font-family:DejaVu Sans,sans-serif}</style><rect width="100%" height="100%" fill="${t.bg}"/>`);
  const r = new Resvg(s, {font: {loadSystemFonts: true, defaultFontFamily: 'DejaVu Sans'}, fitTo: {mode: 'zoom', value: 1.5}});
  fs.writeFileSync(path.join(pngdir, name), r.render().asPng());
}

for (const [view, W] of [['desktop', 1100], ['mobile', 360]]) {
  const vc = new VirtualConsole();
  vc.on('jsdomError', e => errors.push(`${view}: ${e.message}`));
  vc.on('error', e => errors.push(`${view} console: ${e}`));
  const dom = new JSDOM(html, {runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole: vc, beforeParse(w) {
    Object.defineProperty(w.HTMLElement.prototype, 'clientWidth', {get() { return this.id === 'chart' ? W : 0; }});
    w.HTMLElement.prototype.getBoundingClientRect = function () { const svg = this.querySelector && this.querySelector('svg'); const h = svg ? +svg.getAttribute('height') : 0; return {left: 0, top: 0, width: W, height: h, right: W, bottom: h, x: 0, y: 0}; };
    w.addEventListener('error', e => errors.push(`${view}: ${e.message}`));
  }});
  const w = dom.window, d = w.document;
  const svgOf = () => d.querySelector('#chart svg');
  const snap = label => {
    const svg = svgOf(), text = d.querySelector('main').textContent;   // rendered text only: the page's own script source sits outside <main>
    const c = {view, label, steps: d.querySelectorAll('.step').length, tiles: d.querySelectorAll('.tile').length, rows: d.querySelectorAll('#events tbody tr').length,
      svgH: svg ? +svg.getAttribute('height') : 0, bid: !!(svg && /class="ln ln-bid" d="M/.test(svg.outerHTML)), ask: !!(svg && /class="ln ln-ask" d="M/.test(svg.outerHTML)),
      bn: !!(svg && /class="ln ln-bn" d="M/.test(svg.outerHTML)), pos: !!(svg && (svg.outerHTML.match(/class="ln ln-ours" d="M/g) || []).length === 2),
      bad_text: (text.match(BAD) || [null])[0], bad_svg: svg ? (svg.outerHTML.match(BAD) || [null])[0] : 'no svg', fail: /FAIL|DO NOT ADD UP/.test(text),
      info: d.getElementById('epinfo').textContent.slice(0, 110)};
    c.ok = !c.bad_text && !c.bad_svg && !c.fail && c.steps === 10 && c.tiles === 4 && c.rows > 0 && c.svgH > 300 && c.bid && c.ask && c.bn && c.pos;
    checks.push(c); return c;
  };
  const click = el => el.dispatchEvent(new w.MouseEvent('click', {bubbles: true}));
  const hover = (fx, fy) => { const svg = svgOf(); const h = +svg.getAttribute('height'); d.getElementById('chart').dispatchEvent(new w.MouseEvent('pointermove', {bubbles: true, clientX: W * fx, clientY: h * fy})); const tip = d.getElementById('tip'); return tip.style.display === 'block' ? tip.textContent : ''; };
  if (!svgOf()) { errors.push(`${view}: no chart after load`); continue; }
  snap('initial');
  const tipText = hover(0.4, 0.3);
  checks.push({view, label: 'hover', ok: tipText.length > 40 && !BAD.test(tipText), tip: tipText.slice(0, 160)});
  if (pngdir && view === 'desktop') { png(svgOf().outerHTML, 'chart_typical_take.png', false); png(svgOf().outerHTML, 'chart_typical_take_dark.png', true); }
  const mk = () => d.querySelectorAll('#mk-controls button');
  for (const [i, lab] of [[1, 'weak'], [2, 'strong'], [0, 'typical']]) {
    click(mk()[i]); snap(`${lab}`);
    if (pngdir && view === 'desktop' && lab !== 'typical') png(svgOf().outerHTML, `chart_${lab}_take.png`, false);
    click(d.getElementById('next')); snap(`${lab} next`);
    click(d.getElementById('prev')); snap(`${lab} prev`);
    const sel = d.getElementById('episode'); for (let j = 0; j < sel.options.length; j++) { sel.value = String(j); sel.dispatchEvent(new w.Event('change', {bubbles: true})); const c = snap(`${lab} take ${j + 1}`); if (!c.ok) break; checks.push({view, label: `${lab} take ${j + 1} hover`, ok: (t => t.length > 40 && !BAD.test(t))(hover(0.5, 0.45))}); }
    click(d.getElementById('whole')); snap(`${lab} whole`);
    if (pngdir && view === 'desktop') png(svgOf().outerHTML, `chart_${lab}_whole.png`, false);
    click(d.getElementById('zin')); snap(`${lab} zoom in`);
    click(d.getElementById('zout')); snap(`${lab} zoom out`);
    const rows = d.querySelectorAll('#events tbody tr'); click(rows[Math.floor(rows.length / 2)]); snap(`${lab} row click`);
  }
  if (pngdir && view === 'mobile') { click(mk()[0]); png(svgOf().outerHTML, 'chart_mobile.png', false); }
  if (view === 'desktop' && pngdir) fs.writeFileSync(path.join(pngdir, 'page_text.txt'), d.querySelector('main').textContent.replace(/\s+\n/g, '\n'));
  w.close();
}
const bad = checks.filter(c => !c.ok);
console.log(JSON.stringify({errors: errors.slice(0, 8), n_errors: errors.length, n_checks: checks.length, n_bad: bad.length, bad: bad.slice(0, 6),
  hover: checks.filter(c => c.label === 'hover')}, null, 1));
if (errors.length || bad.length || checks.length < 20) process.exitCode = 1;
