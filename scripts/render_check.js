/**
 * Headless render verification for the VERDICT console (web/).
 *
 * Inlines web/index.html + assets into a jsdom document, stubs fetch() to
 * read results/*.json from disk, renders, and asserts every section
 * populates correctly for the current model roster.
 *
 * Usage:
 *   npm install          # one-time dev dependency (jsdom)
 *   npm run check:render # or: node scripts/render_check.js
 *
 * Paths are derived from this file's location, so the harness runs from any
 * checkout on any machine and from any working directory.
 */
const fs = require('fs');
const path = require('path');

/* Prefer the locally installed jsdom so a plain `npm install` at the repo root
   is all a contributor needs. Fall back to a pre-injected global only when the
   require fails, so callers that pre-load jsdom out of tree keep working. */
let JSDOM = globalThis.JSDOM;
try {
  ({ JSDOM } = require('jsdom'));
} catch (err) {
  if (!JSDOM) {
    console.error(
      'render_check: jsdom is not installed. Run `npm install` at the repo root, ' +
      'or expose a global JSDOM before loading this script.\n' +
      `Underlying error: ${err.message}`
    );
    process.exit(2);
  }
}

const REPO = path.resolve(__dirname, '..');
const WEB = path.join(REPO, 'web');

let html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(WEB, 'assets', 'app.js'), 'utf8');
const css = fs.readFileSync(path.join(WEB, 'assets', 'styles.css'), 'utf8');

/* index.html writes the stylesheet link self-closing (`... />`), but accept the
   plain `...>` form too. A non-matching replace would leave the CSS un-inlined
   and silently turn every styling-dependent assertion into a no-op, so the
   substitution is asserted below rather than trusted. */
const CSS_LINK_RE = /<link\b[^>]*\brel=["']stylesheet["'][^>]*\bhref=["']assets\/styles\.css["'][^>]*\/?>/i;
if (!CSS_LINK_RE.test(html)) {
  console.error('render_check: could not find the assets/styles.css <link> in web/index.html');
  process.exit(2);
}
html = html.replace(CSS_LINK_RE, `<style>${css}</style>`);

const APP_JS_RE = /<script\b[^>]*\bsrc=["']assets\/app\.js["'][^>]*>\s*<\/script>/i;
if (!APP_JS_RE.test(html)) {
  console.error('render_check: could not find the assets/app.js <script> in web/index.html');
  process.exit(2);
}
html = html.replace(APP_JS_RE, `<script>${appJs}</script>`);

function diskJson(p) {
  // resolve like a browser would from web/index.html
  const clean = p.replace(/^\.\//, '');
  const candidates = [path.join(REPO, clean), path.join(REPO, 'web', clean)];
  for (const c of candidates) { try { return JSON.parse(fs.readFileSync(c, 'utf8')); } catch (e) {} }
  return null;
}

(async () => {
  const dom = new JSDOM(html, {
    url: 'http://127.0.0.1:8177/web/index.html',
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = (url) => {
        const data = diskJson(String(url));
        if (data) return Promise.resolve({ ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) });
        return Promise.resolve({ ok: false, status: 404, json: async () => { throw new Error('404'); }, text: async () => '' });
      };
    },
  });
  const doc = dom.window.document;
  await new Promise(r => setTimeout(r, 5000));

  const $ = (s) => doc.querySelector(s);
  const $$ = (s) => [...doc.querySelectorAll(s)];
  const t = (s) => (doc.querySelector(s) || {}).textContent || '';
  const results = [];
  const check = (name, cond, detail) => results.push([cond ? 'PASS' : 'FAIL', name, detail || '']);

  check('live data loaded', /live/.test(t('#source-badge')), t('#source-badge'));
  check('hero caption updated', /Four models held/.test(t('#run-banner-h')), t('#run-banner-h').replace(/\s+/g,' ').slice(0, 70));
  check('ox plate pct=13.3', t('#ox-pct') === '13.3', `got "${t('#ox-pct')}"`);
  check('ox meta filled', t('#ox-meta').length > 0, t('#ox-meta').slice(0,60));
  check('ox ci shows wilson', /7\.4/.test(t('#ox-ci')), t('#ox-ci'));
  check('space-bunny plate pct=14.7', t('#sb-pct') === '14.7', `got "${t('#sb-pct')}"`);
  check('space-bunny meta filled', t('#sb-meta').length > 0, t('#sb-meta').slice(0,60));
  check('space-bunny ci shows wilson', /8\.39/.test(t('#sb-ci')), t('#sb-ci'));
  check('axis-ox populated', ($('#axis-ox') || {children:[]}).children.length > 0, `children=${(($('#axis-ox')||{children:[]}).children||[]).length}`);
  check('axis-ox-ci label', /7\.41/.test(t('#axis-ox-ci')), t('#axis-ox-ci'));
  check('axis-sb populated', ($('#axis-sb') || {children:[]}).children.length > 0, `children=${(($('#axis-sb')||{children:[]}).children||[]).length}`);
  check('axis-sb-ci label', /8\.39/.test(t('#axis-sb-ci')), t('#axis-sb-ci'));
  check('gap-note names space-bunny', /space-bunny/.test(t('#gap-note')), t('#gap-note').replace(/<[^>]+>/g,'').slice(0, 110));
  // Break rates are fractions; pct() scales by 100. Catch a double-scaled note.
  check('gap-note rates are sane', !/\d{3,}\.\d%/.test(t('#gap-note')), t('#gap-note').replace(/<[^>]+>/g,'').slice(0, 110));
  check('boot log targets=5', /targets=5/.test(t('#boot-log')), (t('#boot-log').match(/targets=\d+[^"]*?\)/)||[''])[0].slice(0,90));
  check('boot log findings=375', /375 case files/.test(t('#boot-log')), '');
  check('ticker >= 4 verdicts', ($('#ticker-track')||{children:[]}).children.length >= 4, `children=${(($('#ticker-track')||{children:[]}).children||[]).length}`);
  const rows = $$('.bf-row');
  let bfOk = rows.length === 15;
  rows.forEach((r) => {
    const n = r.querySelectorAll('.bf-row__tracks').length;
    if (n !== 0 && n !== 5) bfOk = false;
  });
  check('battlefield: 15 rows x 5 tracks', bfOk, `rows=${rows.length}, first-row tracks=${rows[0] ? rows[0].querySelectorAll('.bf-row__tracks').length : 0}`);
  check('bar chart rendered', $('#chart-bar-wrap').innerHTML.length > 500, `len=${$('#chart-bar-wrap').innerHTML.length}`);
  check('heatmap includes ox column', $$('#chart-heat-wrap text').some(x=>/ox/i.test(x.textContent)), $$('#chart-heat-wrap text').map(x=>x.textContent).join('|').slice(0,80));
  check('heatmap includes space-bunny column', $$('#chart-heat-wrap text').some(x=>/space-bunny/i.test(x.textContent)), $$('#chart-heat-wrap text').map(x=>x.textContent).join('|').slice(0,80));
  check('vault filter has ox option', !!$('option[value="opencode/x-preview-f-free"]'), '');
  check('vault filter has space-bunny option', !!$('option[value="opencode/space-bunny-free"]'), '');
  check('vault count rendered', /\d/.test(t('#vault-count')), t('#vault-count'));
  check('cite table populated', $('#cite-table').children.length > 0, `rows=${$('#cite-table').children.length}`);
  check('evolution chips include ox-alpha', /opencode\/x-preview-f-free/.test(t('#evolution-chips')) && /deepseek/.test(t('#evolution-chips')), [...doc.querySelectorAll('#evolution-chips button')].map(b=>b.textContent).join('|'));
  check('evolution chips include space-bunny', /opencode\/space-bunny-free/.test(t('#evolution-chips')), [...doc.querySelectorAll('#evolution-chips button')].map(b=>b.textContent).join('|'));
  check('evolution compare filled', $('#evolution-compare-body').children.length > 0, `trs=${$('#evolution-compare-body').children.length}`);
  check('audit nodes >= 4', $$('.audit-node').length >= 4, `n=${$$('.audit-node').length}`);

  /* ══ CSS inlining guard ══
     The <link> in index.html is self-closing (`... />`), so a literal string
     replace used to miss it and the stylesheet was never inlined — silently
     turning every styling-dependent assertion below into a no-op. These four
     checks fail if inlining regresses in either direction: not inlined at all,
     or inlined but not actually parsed/applied by the cascade. */
  const styleEl = $('head style');
  const styleText = styleEl ? styleEl.textContent : '';
  check('css: stylesheet inlined into <style>', styleText.length >= css.length * 0.99, `inlined=${styleText.length}B source=${css.length}B`);
  check('css: no leftover styles.css <link>', $$('link[href="assets/styles.css"]').length === 0, `links=${$$('link[href="assets/styles.css"]').length}`);
  const sheet = doc.styleSheets[0];
  const ruleCount = sheet ? sheet.cssRules.length : 0;
  check('css: inlined sheet parsed into rules', ruleCount > 400, `styleSheets=${doc.styleSheets.length} rules=${ruleCount}`);
  // Cascade proof: each value below is set by exactly one .bf-power*/.plate__*
  // rule, so all of them fall back to CSS defaults if the sheet is not applied.
  // Only plain declarations are asserted: jsdom's cssstyle leaves var() and any
  // shorthand containing one (e.g. `border-top: 1px dashed var(--rule)`)
  // unresolved, so those would read as "" regardless of the stylesheet.
  const cs = (sel, prop) => {
    const el = $(sel);
    return el ? dom.window.getComputedStyle(el)[prop] : '<missing ' + sel + '>';
  };
  const cascade = [
    ['.bf-power', 'textTransform', 'uppercase'],
    ['.bf-power__kind', 'fontWeight', '700'],
    ['.bf-power--dead', 'borderTopStyle', 'dashed'],
    ['.bf-power--saturated', 'borderTopWidth', '2px'],
    ['.plate__secondary', 'display', 'flex'],
    ['.plate__secondary-v', 'flexWrap', 'wrap'],
  ];
  const cascadeBad = cascade.filter(([sel, prop, want]) => cs(sel, prop) !== want)
    .map(([sel, prop, want]) => `${sel}{${prop}}: ${cs(sel, prop)} != ${want}`);
  check('css: cascade applied to power badges + plates', cascadeBad.length === 0,
    cascadeBad.length ? cascadeBad.join('; ') : cascade.map(([, p, w]) => `${p}=${w}`).join(' '));

  /* ══ per-strategy discriminative-power badges (battlefield rows) ══
     A strategy's break rate says how often it fired; the badge says whether it
     told the models apart. The classification is derived from the loaded run,
     so these assert this run's derived truth, not a hardcoded strategy list. */
  const badges = $$('.bf-power');
  const stratOf = (el) => (el.closest('.bf-row').querySelector('.bf-row__name') || {}).textContent || '?';
  check('bf-power: 15 badges, one per strategy row', badges.length === 15 && rows.every(r => r.querySelectorAll('.bf-power').length === 1),
    `badges=${badges.length} rows=${rows.length} perRow=${rows.map(r => r.querySelectorAll('.bf-power').length).join(',')}`);
  const byMod = (m) => badges.filter(b => b.classList.contains('bf-power--' + m));
  const sat = byMod('saturated'), dead = byMod('dead');
  check('bf-power: exactly 1 saturated (structured_output)', sat.length === 1 && stratOf(sat[0]) === 'structured_output',
    `n=${sat.length} strategies=${sat.map(stratOf).join(',')}`);
  check('bf-power: exactly 2 dead (context_overflow, memory_manip)',
    dead.length === 2 && dead.map(stratOf).sort().join(',') === 'context_overflow,memory_manip',
    `n=${dead.length} strategies=${dead.map(stratOf).sort().join(',')}`);
  const disc = byMod('discriminative'), narrow = byMod('narrow'), partial = byMod('partial');
  check('bf-power: remaining 12 are discriminative(5)/narrow(7), none partial',
    disc.length === 5 && narrow.length === 7 && partial.length === 0,
    `discriminative=${disc.length} narrow=${narrow.length} partial=${partial.length}`);
  const KINDS = ['SATURATED', 'DEAD', 'DISCRIMINATIVE', 'NARROW', 'PARTIAL'];
  const badBadge = badges.filter(b => {
    const kEl = b.querySelector('.bf-power__kind'), fEl = b.querySelector('.bf-power__fig');
    if (!kEl || !fEl) return true;
    const fig = fEl.textContent.trim();
    return !KINDS.includes(kEl.textContent.trim()) || !(/^broke \d+\/\d+ models?$/.test(fig) || fig === 'finding-level rows partial');
  });
  check('bf-power: every badge spells out a kind + a "broke k/N" figure',
    badBadge.length === 0, badBadge.length ? `bad=${badBadge.map(stratOf).join(',')}` : `kind+fig on all ${badges.length}`);

  /* ══ secondary metric (break rate excluding saturated detectors) ══
     Recomputed from the findings with structured_output dropped: 75 trials per
     model less that one detector's 5 trials = 70. The values are the run's. */
  const secLines = $$('.plate__secondary');
  check('plate__secondary: one line per plate (5)', secLines.length === 5, `n=${secLines.length}`);
  const secParts = secLines.filter(p =>
    !p.querySelector('.plate__secondary-k') || !p.querySelector('.plate__secondary-v') || !p.querySelector('.plate__secondary-n'));
  check('plate__secondary: k / v / n sub-parts on every line', secParts.length === 0, `incomplete=${secParts.length}`);
  const secNs = $$('.plate__secondary-n').map(n => (n.textContent.match(/(\d+)\/(\d+)/) || []).slice(1).join('/'));
  const secValues = secNs.slice().sort();
  check('plate__secondary: reads 2/70, 5/70, 6/70, 11/70, 45/70',
    JSON.stringify(secValues) === JSON.stringify(['11/70', '2/70', '45/70', '5/70', '6/70']), secValues.join(' '));
  check('plate__secondary: denominator 70 = 75 headline trials - 5 saturated trials',
    secLines.every(p => /<b>[\d.]+%<\/b>/.test(p.querySelector('.plate__secondary-v').innerHTML) && /70/.test(p.textContent)),
    secValues.join(' '));

  /* ══ board-level secondary-metric note ══ */
  const satNote = $('.board__sat-note');
  const satNoteText = satNote ? satNote.textContent.replace(/\s+/g, ' ').trim() : '';
  check('board__sat-note: present and names the dropped detector',
    !!satNote && /structured_output/.test(satNoteText), satNoteText.slice(0, 90));
  check('board__sat-note: disclaims replacing the headline',
    /not a replacement for the headline/.test(satNoteText) && /70 trials per model/.test(satNoteText), satNoteText.slice(0, 60));

  /* ══ gap note: two-cluster conclusion, no order among the holders ══
     A superlative word is not banned outright: the note legitimately says
     "qwen3-8b's best case" for the breaker's lower Wilson bound. What must
     never appear is a superlative attached to one of the four holders, either
     in the enumeration of holders or in the sentence that interprets them. */
  const gapText = t('#gap-note').replace(/\s+/g, ' ').trim();
  const HOLDERS = ['gpt-5', 'ox-alpha', 'space-bunny', 'deepseek'];
  const SUPER = 'strongest|weakest|best|worst|fastest|slowest|highest|lowest|hardest|easiest|most|least|top|bottom|outperform|beat';
  const enumeration = gapText.split('clear the gate')[0];
  check('gap-note: states the two-cluster conclusion', /two-cluster/.test(gapText) && /not a ranking/.test(gapText), gapText.slice(0, 60) + '...');
  check('gap-note: explicitly disclaims an order among the holders',
    /supports no order among them/.test(gapText) && /does not separate them/.test(gapText), gapText.slice(60, 150));
  check('gap-note: holder enumeration asserts no superlative',
    HOLDERS.filter(h => enumeration.includes(h)).length === 4 && !new RegExp(`\\b(${SUPER})\\b`, 'i').test(enumeration),
    enumeration.slice(0, 110));
  const supNearHolder = new RegExp(`\\b(${SUPER})\\b[^.]{0,20}(${HOLDERS.join('|')})|(${HOLDERS.join('|')})[^.]{0,20}\\b(${SUPER})\\b`, 'i');
  const supHit = supNearHolder.exec(gapText);
  check('gap-note: no superlative attached to any holder anywhere', !supHit, supHit ? supHit[0] : 'clean');

  for (const [st, name, detail] of results) console.log(`${st}  ${name}${detail ? '   [' + detail + ']' : ''}`);
  const fails = results.filter(r => r[0] === 'FAIL').length;
  console.log(`\n${results.length - fails}/${results.length} checks passed`);
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error('HARNESS ERROR:', e.stack); process.exit(2); });
