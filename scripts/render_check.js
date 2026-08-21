/**
 * Headless render verification for the VERDICT console (web/).
 *
 * Inlines web/index.html + assets into a jsdom document, stubs fetch() to
 * read results/*.json from disk, renders, and asserts every section
 * populates correctly for the current model roster.
 *
 * Usage:
 *   cd /tmp/jsdomtest && npm i jsdom   # one-time dev dependency
 *   node scripts/render_check.js       # expects jsdom resolvable + repo checkout
 */
const fs = require('fs');
const path = require('path');
const WEB = '/home/ashwin/Projects/ai-risk-evaluation-workbench/web';
const REPO = path.dirname(WEB);

let html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(WEB, 'assets', 'app.js'), 'utf8');
const css = fs.readFileSync(path.join(WEB, 'assets', 'styles.css'), 'utf8');
html = html.replace('<link rel="stylesheet" href="assets/styles.css">', `<style>${css}</style>`);
html = html.replace('<script src="assets/app.js"></script>', `<script>${appJs}</script>`);

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
  check('hero caption updated', /Three models held/.test(t('#run-banner-h')), t('#run-banner-h').replace(/\s+/g,' ').slice(0, 70));
  check('ox plate pct=13.3', t('#ox-pct') === '13.3', `got "${t('#ox-pct')}"`);
  check('ox meta filled', t('#ox-meta').length > 0, t('#ox-meta').slice(0,60));
  check('ox ci shows wilson', /7\.4/.test(t('#ox-ci')), t('#ox-ci'));
  check('axis-ox populated', ($('#axis-ox') || {children:[]}).children.length > 0, `children=${(($('#axis-ox')||{children:[]}).children||[]).length}`);
  check('axis-ox-ci label', /7\.41/.test(t('#axis-ox-ci')), t('#axis-ox-ci'));
  check('gap-note names ox-alpha', /Ox Alpha Free/.test(t('#gap-note')), t('#gap-note').replace(/<[^>]+>/g,'').slice(0, 110));
  check('boot log targets=4', /targets=4/.test(t('#boot-log')), (t('#boot-log').match(/targets=\d+[^"]*?\)/)||[''])[0].slice(0,90));
  check('boot log findings=300', /300 case files/.test(t('#boot-log')), '');
  check('ticker >= 4 verdicts', ($('#ticker-track')||{children:[]}).children.length >= 4, `children=${(($('#ticker-track')||{children:[]}).children||[]).length}`);
  const rows = $$('.bf-row');
  let bfOk = rows.length === 15;
  rows.forEach((r) => {
    const n = r.querySelectorAll('.bf-row__tracks').length;
    if (n !== 0 && n !== 4) bfOk = false;
  });
  check('battlefield: 15 rows x 4 tracks', bfOk, `rows=${rows.length}, first-row tracks=${rows[0] ? rows[0].querySelectorAll('.bf-row__tracks').length : 0}`);
  check('bar chart rendered', $('#chart-bar-wrap').innerHTML.length > 500, `len=${$('#chart-bar-wrap').innerHTML.length}`);
  check('heatmap includes ox column', $$('#chart-heat-wrap text').some(x=>/ox/i.test(x.textContent)), $$('#chart-heat-wrap text').map(x=>x.textContent).join('|').slice(0,80));
  check('vault filter has ox option', !!$('option[value="opencode/x-preview-f-free"]'), '');
  check('vault count rendered', /\d/.test(t('#vault-count')), t('#vault-count'));
  check('cite table populated', $('#cite-table').children.length > 0, `rows=${$('#cite-table').children.length}`);
  check('evolution chips include ox-alpha', /opencode\/x-preview-f-free/.test(t('#evolution-chips')) && /deepseek/.test(t('#evolution-chips')), [...doc.querySelectorAll('#evolution-chips button')].map(b=>b.textContent).join('|'));
  check('evolution compare filled', $('#evolution-compare-body').children.length > 0, `trs=${$('#evolution-compare-body').children.length}`);
  check('audit nodes >= 4', $$('.audit-node').length >= 4, `n=${$$('.audit-node').length}`);

  for (const [st, name, detail] of results) console.log(`${st}  ${name}${detail ? '   [' + detail + ']' : ''}`);
  const fails = results.filter(r => r[0] === 'FAIL').length;
  console.log(`\n${results.length - fails}/${results.length} checks passed`);
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error('HARNESS ERROR:', e.stack); process.exit(2); });
