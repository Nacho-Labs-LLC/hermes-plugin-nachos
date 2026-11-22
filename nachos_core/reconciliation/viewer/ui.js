/* Nachos reconciliation viewer UI. Imported text only ever reaches the DOM via
   textContent / textarea.value. No innerHTML, no eval, no network. */
(function () {
  'use strict';
  var Core = globalThis.NachosCore;
  var MEMORY_FIELDS = ['key', 'title', 'summary', 'category', 'body'];
  var DECISIONS = ['unreviewed', 'accept', 'request_revision', 'reject', 'defer'];
  var KINDS = ['memory_record', 'memory_file', 'skill_file', 'reference_file'];
  var MAX_ROWS = 500;

  var S = {
    session: null, fileName: '', selected: null, gen: 0, busy: false,
    filters: { kind: '', finding: '', decision: '' }, showAll: false,
    edit: null, ledger: {}, ledgerGen: 0, view: 'unified', note: ''
  };

  function $(id) { return document.getElementById(id); }
  function el(tag, props, kids) {
    var n = document.createElement(tag);
    var k;
    if (props) {
      for (k in props) {
        if (k === 'text') { n.textContent = props[k]; }
        else if (k === 'class') { n.className = props[k]; }
        else if (k === 'data') { var d; for (d in props[k]) { n.setAttribute('data-' + d, props[k][d]); } }
        else if (k === 'on') { var e; for (e in props[k]) { n.addEventListener(e, props[k][e]); } }
        else if (props[k] === true) { n.setAttribute(k, ''); }
        else if (props[k] !== false && props[k] !== null && props[k] !== undefined) { n.setAttribute(k, String(props[k])); }
      }
    }
    (kids || []).forEach(function (c) {
      if (c === null || c === undefined) { return; }
      n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  function clear(n) { while (n.firstChild) { n.removeChild(n.firstChild); } }
  function say(msg) { $('status').textContent = msg; }
  function showError(msg) { $('error').textContent = msg || ''; }
  function clone(v) { return JSON.parse(JSON.stringify(v)); }
  function errText(e) {
    return (e && e.nachos ? 'Rejected: ' : 'Error: ') + (e && e.message ? e.message : String(e));
  }
  function dl(pairs) {
    var d = el('dl');
    pairs.forEach(function (p) { d.appendChild(el('dt', { text: p[0] })); d.appendChild(el('dd', { text: p[1] })); });
    return d;
  }
  function yn(v) { return v === true ? 'yes' : v === false ? 'no' : 'unknown'; }

  /* ----------------------------------------------------------- data access */
  function packet() { return S.session ? S.session.packet : null; }
  function record(id) {
    var rs = S.session.reviews;
    for (var i = 0; i < rs.length; i++) { if (rs[i].proposal_id === id) { return rs[i]; } }
    return null;
  }
  function current(rec) { return rec.revisions[rec.revisions.length - 1]; }
  function sourceMap() {
    var m = {};
    packet().inventory.sources.forEach(function (s) { m[s.id] = s; });
    return m;
  }
  function findingMap() {
    var m = {};
    packet().findings.forEach(function (f) { m[f.id] = f; });
    return m;
  }
  function proposalKinds(p) {
    var sm = sourceMap(); var out = [];
    p.changes.forEach(function (c) {
      var s = sm[c.source_id];
      if (s && out.indexOf(s.kind) < 0) { out.push(s.kind); }
    });
    return out;
  }
  function proposalSourceIds(p) {
    var fm = findingMap(); var ids = {};
    p.finding_ids.forEach(function (f) { (fm[f] ? fm[f].source_ids : []).forEach(function (s) { ids[s] = true; }); });
    p.changes.forEach(function (c) { ids[c.source_id] = true; });
    return packet().inventory.sources.filter(function (s) { return ids[s.id]; });
  }

  /* ------------------------------------------------------------- dialogs */
  function confirmDialog(title, bodyNodes, okLabel) {
    return new Promise(function (resolve) {
      var m = $('modal'); var body = $('modal-body');
      if (!m.hidden) { resolve(false); return; } /* one dialog at a time; never clobber the open one */
      var prev = document.activeElement;
      $('modal-title').textContent = title;
      clear(body);
      bodyNodes.forEach(function (n) { body.appendChild(n); });
      $('modal-ok').textContent = okLabel || 'Continue';
      m.hidden = false;
      $('app').inert = true; $('app').setAttribute('aria-hidden', 'true');
      $('modal-cancel').focus();
      function done(v) {
        m.hidden = true;
        $('app').inert = false; $('app').removeAttribute('aria-hidden');
        $('modal-ok').onclick = null; $('modal-cancel').onclick = null; document.removeEventListener('keydown', onKey, true);
        if (prev && prev.focus) { prev.focus(); }
        resolve(v);
      }
      $('modal-ok').onclick = function () { done(true); };
      $('modal-cancel').onclick = function () { done(false); };
      function onKey(ev) {
        if (ev.key === 'Escape') { ev.preventDefault(); done(false); return; }
        if (ev.key === 'Tab') {
          var f = Array.prototype.filter.call(m.querySelectorAll('button,[href],input,select,textarea,[tabindex]:not([tabindex="-1"])'), function (n) {
            return !n.disabled && !n.hidden && !n.closest('[hidden]');
          });
          ev.preventDefault();
          if (!f.length) { return; }
          var i = f.indexOf(document.activeElement);
          var last = f.length - 1;
          if (i < 0) { f[ev.shiftKey ? last : 0].focus(); }
          else if (ev.shiftKey) { f[i <= 0 ? last : i - 1].focus(); }
          else { f[i >= last ? 0 : i + 1].focus(); }
        }
      }
      document.addEventListener('keydown', onKey, true); /* works even if a backdrop click blurred the dialog */
    });
  }

  /* --------------------------------------------------------------- edits */
  function isDirty() {
    if (!S.edit) { return false; }
    var k;
    for (k in S.edit.fields) { if (S.edit.fields[k].touched && S.edit.fields[k].value !== S.edit.fields[k].norm) { return true; } }
    return vSerial() !== S.edit.vInit;
  }
  var EV_KINDS = ['schedule', 'execution', 'delivery', 'manual', 'source_content', 'usage'];
  var EV_STATUSES = ['verified', 'unverified', 'missing', 'conflicting'];
  function vSerial() {
    return JSON.stringify(S.edit.verify.map(function (v) {
      return [v.fid, v.items.map(function (i) { return [i.item, i.preserved, i.explanation]; }),
        v.ev.map(function (e) { return [e.claim, e.observation, e.status, e.kind, e.sources]; })];
    }));
  }
  function nowUtc() { return new Date().toISOString().replace(/\.\d+Z$/, 'Z'); }
  function buildVerify(p, fm) {
    return p.finding_ids.map(function (fid) {
      var a = null;
      p.assessments.forEach(function (x) { if (x.finding_id === fid) { a = x; } });
      var v = { fid: fid, items: [], ev: [] };
      if (a) {
        a.preservation_checklist.forEach(function (c) { v.items.push({ item: c.item, preserved: c.preserved, explanation: c.explanation }); });
        a.evidence.forEach(function (o) { v.ev.push({ claim: o.claim, observation: o.observation, status: o.status, kind: o.kind, sources: o.source_ids.slice() }); });
      }
      if (!v.items.length) { v.items.push({ item: '', preserved: true, explanation: '' }); }
      if (!v.ev.length) { v.ev.push({ claim: '', observation: '', status: 'verified', kind: 'manual', sources: [] }); }
      return v;
    });
  }
  function verifyPanel(fm) {
    var panel = el('div', { class: 'card', data: { testid: 'verify-panel' }, role: 'group', 'aria-label': 'Reverification' });
    function focusAt(vi, kind, idx, tid, fallback) {
      var blk = panel.querySelectorAll('[data-testid="verify-finding"]')[vi];
      if (!blk) { return; }
      var rows = blk.querySelectorAll('[data-testid="' + kind + '"]');
      var row = rows[Math.min(idx, rows.length - 1)];
      var t = row ? row.querySelector('[data-testid="' + tid + '"]') : null;
      if (!t || t.disabled) { t = blk.querySelector('[data-testid="' + fallback + '"]'); }
      if (t) { t.focus(); }
    }
    function render() {
      panel.textContent = '';
      panel.appendChild(el('h3', { text: 'Reverify preservation (new revision)' }));
      panel.appendChild(el('p', { class: 'note', text: 'Fill every checklist and evidence row for each finding, then use Reverify and save revision. Unchecked preserved items keep Accept disabled.' }));
      S.edit.verify.forEach(function (v, vi) {
        var pre = 'v' + vi + '-';
        var blk = el('div', { class: 'card', data: { testid: 'verify-finding', finding: v.fid }, role: 'group', 'aria-label': 'Verification for finding ' + v.fid });
        blk.appendChild(el('h4', { text: 'Finding ' + v.fid }));
        v.items.forEach(function (it, ii) {
          var id = pre + 'i' + ii;
          var row = el('div', { class: 'row vrow', data: { testid: 'verify-item', index: String(ii) } });
          var t = el('input', { type: 'text', id: id + 't', data: { testid: 'verify-item-text' } }); t.value = it.item;
          t.addEventListener('input', function () { it.item = t.value; updateActions(); });
          var c = el('input', { type: 'checkbox', id: id + 'c', data: { testid: 'verify-preserved' } }); c.checked = it.preserved;
          c.addEventListener('change', function () { it.preserved = c.checked; updateActions(); });
          var x = el('input', { type: 'text', id: id + 'x', data: { testid: 'verify-explanation' } }); x.value = it.explanation;
          x.addEventListener('input', function () { it.explanation = x.value; updateActions(); });
          var rm = el('button', { type: 'button', text: 'Remove item', data: { testid: 'verify-remove-item' }, 'aria-label': 'Remove checklist item ' + (ii + 1) + ' of finding ' + v.fid });
          if (v.items.length < 2) { rm.disabled = true; }
          rm.addEventListener('click', function () { if (v.items.length > 1) { v.items.splice(ii, 1); render(); updateActions(); focusAt(vi, 'verify-item', ii, 'verify-remove-item', 'verify-add-item'); } });
          row.appendChild(el('label', { for: id + 't', text: 'Item' })); row.appendChild(t);
          row.appendChild(el('label', { for: id + 'c', text: 'Preserved' })); row.appendChild(c);
          row.appendChild(el('label', { for: id + 'x', text: 'Explanation' })); row.appendChild(x);
          row.appendChild(rm); blk.appendChild(row);
        });
        var add = el('button', { type: 'button', text: 'Add checklist item', data: { testid: 'verify-add-item' } });
        add.addEventListener('click', function () { v.items.push({ item: '', preserved: true, explanation: '' }); render(); updateActions(); focusAt(vi, 'verify-item', v.items.length - 1, 'verify-item-text', 'verify-add-item'); });
        blk.appendChild(add);
        var fsrc = fm[v.fid] ? fm[v.fid].source_ids : [];
        v.ev.forEach(function (e, ei) {
          var id = pre + 'e' + ei;
          var row = el('div', { class: 'row vrow', data: { testid: 'verify-evidence', index: String(ei) } });
          var cl = el('input', { type: 'text', id: id + 'c', data: { testid: 'verify-claim' } }); cl.value = e.claim;
          cl.addEventListener('input', function () { e.claim = cl.value; updateActions(); });
          var ob = el('input', { type: 'text', id: id + 'o', data: { testid: 'verify-observation' } }); ob.value = e.observation;
          ob.addEventListener('input', function () { e.observation = ob.value; updateActions(); });
          var st = el('select', { id: id + 's', data: { testid: 'verify-evidence-status' } });
          EV_STATUSES.forEach(function (o) { var op = el('option', { text: o }); op.value = o; st.appendChild(op); });
          st.value = e.status; st.addEventListener('change', function () { e.status = st.value; updateActions(); });
          var kd = el('select', { id: id + 'k', data: { testid: 'verify-evidence-kind' } });
          EV_KINDS.forEach(function (o) { var op = el('option', { text: o }); op.value = o; kd.appendChild(op); });
          kd.value = e.kind; kd.addEventListener('change', function () { e.kind = kd.value; updateActions(); });
          row.appendChild(el('label', { for: id + 'c', text: 'Claim' })); row.appendChild(cl);
          row.appendChild(el('label', { for: id + 'o', text: 'Observation' })); row.appendChild(ob);
          row.appendChild(el('label', { for: id + 's', text: 'Status' })); row.appendChild(st);
          row.appendChild(el('label', { for: id + 'k', text: 'Kind' })); row.appendChild(kd);
          var fs = el('fieldset', { class: 'vsrc' }, [el('legend', { text: 'Sources' })]);
          var allSrc = fsrc.concat(e.sources.filter(function (z) { return fsrc.indexOf(z) < 0; }));
          allSrc.forEach(function (sid, si) {
            var cid = id + 'src' + si;
            var cb = el('input', { type: 'checkbox', id: cid, data: { testid: 'verify-source' } });
            cb.value = sid; cb.checked = e.sources.indexOf(sid) >= 0;
            cb.addEventListener('change', function () {
              e.sources = e.sources.filter(function (z) { return z !== sid; });
              if (cb.checked) { e.sources.push(sid); }
              e.sources.sort(function (a, b) { return allSrc.indexOf(a) - allSrc.indexOf(b); });
              updateActions();
            });
            fs.appendChild(cb); fs.appendChild(el('label', { for: cid, text: sid }));
          });
          row.appendChild(fs);
          var rm = el('button', { type: 'button', text: 'Remove evidence', data: { testid: 'verify-remove-evidence' }, 'aria-label': 'Remove evidence row ' + (ei + 1) + ' of finding ' + v.fid });
          if (v.ev.length < 2) { rm.disabled = true; }
          rm.addEventListener('click', function () { if (v.ev.length > 1) { v.ev.splice(ei, 1); render(); updateActions(); focusAt(vi, 'verify-evidence', ei, 'verify-remove-evidence', 'verify-add-evidence'); } });
          row.appendChild(rm); blk.appendChild(row);
        });
        var adde = el('button', { type: 'button', text: 'Add evidence', data: { testid: 'verify-add-evidence' } });
        adde.addEventListener('click', function () { v.ev.push({ claim: '', observation: '', status: 'verified', kind: 'manual', sources: [] }); render(); updateActions(); focusAt(vi, 'verify-evidence', v.ev.length - 1, 'verify-claim', 'verify-add-evidence'); });
        blk.appendChild(adde);
        panel.appendChild(blk);
      });
    }
    render();
    return panel;
  }
  function verifiedAssessments(p, revision) {
    var at = nowUtc();
    return S.edit.verify.map(function (v) {
      var tag = 'Finding ' + v.fid + ': ';
      if (!v.items.length) { throw new Error(tag + 'add at least one checklist item.'); }
      v.items.forEach(function (i, n) {
        if (i.item.trim() === '') { throw new Error(tag + 'checklist item ' + (n + 1) + ' needs an item text.'); }
        if (i.explanation.trim() === '') { throw new Error(tag + 'checklist item ' + (n + 1) + ' needs an explanation.'); }
      });
      if (!v.ev.length) { throw new Error(tag + 'add at least one evidence row.'); }
      v.ev.forEach(function (e, n) {
        if (e.claim.trim() === '') { throw new Error(tag + 'evidence row ' + (n + 1) + ' needs a claim.'); }
        if (e.observation.trim() === '') { throw new Error(tag + 'evidence row ' + (n + 1) + ' needs an observation.'); }
        if (!e.sources.length) { throw new Error(tag + 'evidence row ' + (n + 1) + ' needs at least one source.'); }
      });
      return {
        finding_id: v.fid, revision: revision,
        evidence: v.ev.map(function (e) { return { kind: e.kind, status: e.status, source_ids: e.sources.slice(), claim: e.claim, observation: e.observation, observed_at: at }; }),
        preservation_checklist: v.items.map(function (i) { return { item: i.item, preserved: i.preserved, explanation: i.explanation }; })
      };
    });
  }

  async function guardDirty() {
    if (!isDirty()) { return true; }
    return confirmDialog('Discard unsaved edits?', [el('p', { text: 'You have unsaved edits. Continuing discards them.' })], 'Discard edits');
  }

  /* ------------------------------------------------------------- import */
  async function importFile(file) {
    showError('');
    var buf;
    try {
      buf = await file.arrayBuffer();
      var value = Core.parseStrict(new Uint8Array(buf));
      var schema = value && value.schema;
      if (schema === Core.SCHEMAS.plan || schema === Core.SCHEMAS.inventory) {
        throw new Error('This is a ' + (schema === Core.SCHEMAS.plan ? 'plan' : 'inventory') +
          ' document. The viewer reviews packets and sessions only; a plan is never converted to a session.');
      }
      if (schema !== Core.SCHEMAS.packet && schema !== Core.SCHEMAS.session) {
        throw new Error('Unsupported document schema; expected a packet or session.');
      }
      var doc = await Core.validateDocument(value);
      var session = schema === Core.SCHEMAS.packet ? await Core.newSession(doc) : doc;
      if (schema === Core.SCHEMAS.session) { session = await Core.validateDocument(session); }
      var prof = session.profile;
      var ok = await confirmDialog('Confirm profile binding', [
        el('p', { text: 'Loading ' + (schema === Core.SCHEMAS.packet ? 'packet' : 'session') + ' "' + file.name + '". This replaces the current review context.' }),
        dl([['profile.id', prof.id], ['profile.root', prof.root], ['proposals', String(session.reviews.length)]]),
        el('p', { class: 'note', text: 'Profile values are inert labels; nothing is opened or written.' })
      ], 'Load and replace');
      if (!ok) { say('Import cancelled; previous state kept.'); return; }
      if (!(await guardDirty())) { say('Import cancelled; unsaved edits kept.'); return; }
      S.session = session; S.fileName = file.name; S.gen++; S.edit = null; S.selected = null;
      S.filters = { kind: '', finding: '', decision: '' }; S.showAll = false; S.ledger = {};
      afterSessionChange();
      say('Loaded ' + file.name + ' (' + session.reviews.length + ' proposals).');
    } catch (e) {
      showError('Import failed; previous state kept. ' + errText(e));
      say('Import failed.');
    }
  }

  function afterSessionChange() {
    $('context').textContent = S.session
      ? 'Loaded: ' + S.fileName + ' | profile ' + S.session.profile.id + ' @ ' + S.session.profile.root + ' | session digest ' + S.session.digest
      : 'No document loaded.';
    document.querySelector('[data-action="save-session"]').disabled = !S.session;
    document.querySelector('[data-action="export-plan"]').disabled = !S.session;
    buildFilterOptions();
    renderQueue();
    renderDetail();
    refreshLedger();
  }

  async function refreshLedger() {
    var my = ++S.ledgerGen; var gen = S.gen;
    if (!S.session) { return; }
    try {
      var plan = await Core.exportPlan(S.session);
      if (my !== S.ledgerGen || gen !== S.gen) { return; }
      var m = {};
      plan.ledger.forEach(function (l) { m[l.proposal_id] = l; });
      S.ledger = m;
      renderQueue();
    } catch (e) {
      if (my === S.ledgerGen) { S.ledger = {}; showError('Export preview unavailable. ' + errText(e)); }
    }
  }

  /* ------------------------------------------------------------- filters */
  function fillSelect(sel, opts, value) {
    clear(sel);
    opts.forEach(function (o) { sel.appendChild(el('option', { value: o[0], text: o[1] })); });
    sel.value = value;
  }
  function buildFilterOptions() {
    fillSelect($('filter-kind'), [['', 'All kinds']].concat(KINDS.map(function (k) { return [k, k]; })), S.filters.kind);
    var f = [['', 'All findings']];
    if (S.session) {
      var types = {};
      packet().findings.forEach(function (x) { types[x.type] = true; });
      Object.keys(types).sort().forEach(function (t) { f.push(['type:' + t, 'Type: ' + t]); });
      packet().findings.forEach(function (x) { f.push(['id:' + x.id, 'ID: ' + x.id]); });
    }
    fillSelect($('filter-finding'), f, S.filters.finding);
    fillSelect($('filter-decision'), [['', 'All decisions']].concat(DECISIONS.map(function (d) { return [d, d]; })), S.filters.decision);
  }
  function matches(rec) {
    var p = current(rec); var fl = S.filters;
    if (fl.decision && rec.decision !== fl.decision) { return false; }
    if (fl.kind && proposalKinds(p).indexOf(fl.kind) < 0) { return false; }
    if (fl.finding) {
      var fm = findingMap();
      var ok = p.finding_ids.some(function (id) {
        if (fl.finding.indexOf('id:') === 0) { return id === fl.finding.slice(3); }
        return fm[id] && fm[id].type === fl.finding.slice(5);
      });
      if (!ok) { return false; }
    }
    return true;
  }

  /* Focus management after a re-render. Only called for activations that came from inside the app. */
  function focusSelectedQueueItem() {
    var b = $('queue').querySelector('button[aria-current="true"]');
    if (b) { b.focus(); }
  }
  function focusAction(action) {
    var r = S.refs; var order = ['accept', 'edit', 'save-edit', 'reverify', 'cancel-edit', 'request-revision', 'reject', 'defer'];
    var usable = function (b) { return b && !b.disabled && !b.hidden; };
    if (r && r.acts) {
      var t = r.acts[action];
      if (!usable(t)) {
        var at = order.indexOf(action); var best = null; var d;
        for (d = 1; d < order.length && !best; d++) {
          [order[at - d], order[at + d]].forEach(function (o) { if (!best && o && usable(r.acts[o])) { best = r.acts[o]; } });
        }
        t = best;
      }
      if (t) { t.focus(); return; }
    }
    var h = $('detail').querySelector('h2');
    if (h) { h.focus(); }
  }
  function inApp() {
    var a = document.activeElement;
    return !a || a === document.body || !!a.closest('#app');
  }

  async function selectProposal(id) {
    if (id === S.selected) { return; }
    if (!(await guardDirty())) { return; }
    S.gen++; S.selected = id; S.edit = null; S.note = '';
    renderQueue(); renderDetail();
    focusSelectedQueueItem();
    say('Selected ' + id);
  }

  function renderQueue() {
    var q = $('queue');
    var ae = document.activeElement; var keep = ae && ae.dataset && ae.dataset.testid === 'queue-item' && q.contains(ae) ? ae.dataset.proposal : null;
    renderQueueRows();
    if (keep !== null) { /* re-render must not drop focus to body (e.g. async ledger refresh) */
      var rows = q.querySelectorAll('button[data-testid="queue-item"]'); var j;
      for (j = 0; j < rows.length; j++) { if (rows[j].dataset.proposal === keep) { rows[j].focus(); break; } }
    }
  }
  function renderQueueRows() {
    var q = $('queue'); clear(q);
    var counts = $('counts');
    if (!S.session) { counts.textContent = ''; return; }
    var all = S.session.reviews;
    if (all.length === 0) { counts.textContent = 'Total 0 | Showing 0'; q.appendChild(el('li', { text: 'This packet has no proposals.', data: { testid: 'empty-packet' } })); return; }
    var shown = all.filter(matches);
    counts.textContent = 'Total ' + all.length + ' | Filtered ' + shown.length;
    if (shown.length === 0) { q.appendChild(el('li', { text: 'No proposals match the current filters.', data: { testid: 'no-results' } })); return; }
    var limit = S.showAll ? shown.length : Math.min(shown.length, MAX_ROWS);
    shown.slice(0, limit).forEach(function (rec) {
      var p = current(rec); var led = S.ledger[rec.proposal_id];
      var b = el('button', { type: 'button', data: { proposal: rec.proposal_id, testid: 'queue-item' }, 'aria-current': rec.proposal_id === S.selected ? 'true' : 'false',
        on: { click: function () { selectProposal(rec.proposal_id); } } }, [
        el('div', { text: rec.proposal_id }),
        el('span', { class: 'tag', text: 'rev ' + p.revision }),
        el('span', { class: 'tag', text: rec.decision }),
        led && led.status === 'blocked' ? el('span', { class: 'tag blocked', text: 'export blocked' }) : null
      ]);
      q.appendChild(el('li', null, [b]));
    });
    if (limit < shown.length) {
      q.appendChild(el('li', null, [el('button', { type: 'button', text: 'Show all ' + shown.length, on: { click: function () { S.showAll = true; renderQueue(); var nr = $('queue').querySelectorAll('button[data-testid="queue-item"]')[limit]; if (nr) { nr.focus(); } } } })]));
    }
  }

  /* ---------------------------------------------------------------- diff */
  function diffNode(a, b, mode) {
    var d = Core.lineDiff(a, b);
    if (d.fallback) {
      return el('div', { class: 'card', data: { testid: 'diff-fallback' } }, [
        el('p', { class: 'warn', text: 'Diff too large; showing full before and after text without comparison.' }),
        el('h4', { text: 'Before (full)' }), el('pre', { text: a }),
        el('h4', { text: 'After (full)' }), el('pre', { text: b })
      ]);
    }
    if (mode === 'unified') {
      var u = el('div', { class: 'diff', data: { testid: 'diff-unified' } });
      d.rows.forEach(function (r) {
        var pre = r.type === 'add' ? '+ ' : r.type === 'del' ? '- ' : '  ';
        u.appendChild(el('div', { class: 'line ' + r.type, text: pre + (r.type === 'add' ? r.b : r.a) }));
      });
      return u;
    }
    var left = el('div', null, [el('div', { class: 'pane-h', text: 'Before' })]);
    var right = el('div', null, [el('div', { class: 'pane-h', text: 'After' })]);
    var i = 0; var rows = d.rows;
    function pad(n, c) { var k; for (k = 0; k < c; k++) { n.appendChild(el('div', { class: 'line', text: '' })); } }
    while (i < rows.length) {
      if (rows[i].type === 'eq') {
        left.appendChild(el('div', { class: 'line eq', text: rows[i].a }));
        right.appendChild(el('div', { class: 'line eq', text: rows[i].b }));
        i++;
      } else {
        var dels = []; var adds = [];
        while (i < rows.length && rows[i].type !== 'eq') { (rows[i].type === 'del' ? dels : adds).push(rows[i]); i++; }
        dels.forEach(function (r) { left.appendChild(el('div', { class: 'line del', text: r.a })); });
        adds.forEach(function (r) { right.appendChild(el('div', { class: 'line add', text: r.b })); });
        pad(left, Math.max(0, adds.length - dels.length));
        pad(right, Math.max(0, dels.length - adds.length));
      }
    }
    return el('div', { class: 'diff side', data: { testid: 'diff-side' } }, [left, right]);
  }

  function diffTabs(a, b, label) {
    var wrap = el('div');
    var box = el('div');
    var tabs = el('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Diff view for ' + label });
    var tU, tS;
    function show(mode) {
      S.view = mode;
      tU.setAttribute('aria-selected', mode === 'unified' ? 'true' : 'false'); tU.tabIndex = mode === 'unified' ? 0 : -1;
      tS.setAttribute('aria-selected', mode === 'side' ? 'true' : 'false'); tS.tabIndex = mode === 'side' ? 0 : -1;
      clear(box); box.appendChild(diffNode(a, b, mode));
    }
    function key(ev) {
      if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight' || ev.key === 'Home' || ev.key === 'End') {
        ev.preventDefault();
        var m = ev.key === 'ArrowLeft' || ev.key === 'Home' ? 'unified' : 'side';
        show(m); (m === 'unified' ? tU : tS).focus();
      }
    }
    tU = el('button', { type: 'button', role: 'tab', text: 'Unified', data: { testid: 'tab-unified' }, on: { click: function () { show('unified'); }, keydown: key } });
    tS = el('button', { type: 'button', role: 'tab', text: 'Side by side', data: { testid: 'tab-side' }, on: { click: function () { show('side'); }, keydown: key } });
    tabs.appendChild(tU); tabs.appendChild(tS);
    wrap.appendChild(tabs); wrap.appendChild(box);
    show(S.view);
    return wrap;
  }

  /* -------------------------------------------------------------- detail */
  function snapshotNodes(src) {
    var out = el('div');
    if (typeof src.before === 'string') {
      out.appendChild(el('pre', { text: src.before }));
    } else {
      MEMORY_FIELDS.forEach(function (f) { out.appendChild(el('h4', { text: f })); out.appendChild(el('pre', { text: src.before[f] })); });
    }
    return out;
  }
  function coverageDl(c) {
    return dl([['discovered', yn(c.discovered)], ['hashed', yn(c.hashed)], ['content searched', yn(c.content_searched)],
      ['section reviewed', yn(c.section_reviewed)], ['body reviewed', yn(c.body_reviewed)],
      ['limitations', c.limitations.length ? c.limitations.join(' | ') : '(none)']]);
  }
  function obsNode(o) {
    return el('div', { class: 'card' }, [dl([['kind', o.kind], ['status', o.status], ['claim', o.claim], ['observation', o.observation],
      ['observed_at', o.observed_at], ['sources', o.source_ids.join(', ')]])]);
  }
  function sourceCard(s, sec) {
    var m = s.metadata; var u = m.usage;
    var d = dl([['id', s.id], ['kind', s.kind], ['role', s.role], ['identity', s.identity], ['locator', s.locator], ['sha256', s.sha256],
      ['enabled', m.enabled === null ? 'unknown' : m.enabled ? 'true' : 'false'], ['bytes', String(m.byte_size)], ['chars', String(m.char_size)],
      ['provenance', m.provenance.length ? m.provenance.join(' | ') : '(none)'],
      ['usage status', u.status + (u.status === 'available' ? '' : ' (not proof of unused)')],
      ['usage references', u.references && u.references.length ? u.references.join(' | ') : '(none)']]);
    var det = el('details', { class: 'card' }, [el('summary', { text: sec + ' ' + s.id }), d, el('h4', { text: 'Coverage' }), coverageDl(s.coverage),
      el('h4', { text: 'Full snapshot' }), snapshotNodes(s)]);
    return det;
  }

  function changeNode(c, idx, rec, p) {
    var sm = sourceMap(); var s = sm[c.source_id];
    var card = el('div', { class: 'card', data: { testid: 'change' } });
    card.appendChild(el('h3', { text: 'Change ' + (idx + 1) + ': ' + c.operation + ' ' + c.source_id }));
    card.appendChild(dl([['source id', c.source_id], ['kind', s ? s.kind : '?'], ['identity', s ? s.identity : '?'], ['locator', s ? s.locator : '?'],
      ['source sha256 (precondition)', c.source_sha256], ['operation', c.operation]]));
    var fields = typeof c.before === 'string' ? [null] : MEMORY_FIELDS;
    if (c.operation === 'disable') {
      card.appendChild(el('p', { class: 'warn', text: 'Proposed disable - no writes' }));
      fields.forEach(function (f) {
        card.appendChild(el('h4', { text: 'Before' + (f ? ' - ' + f : '') }));
        card.appendChild(el('pre', { text: f ? c.before[f] : c.before }));
      });
      return card;
    }
    fields.forEach(function (f) {
      var a = f ? c.before[f] : c.before; var b = f ? c.after[f] : c.after;
      card.appendChild(el('h4', { text: f ? 'Memory field: ' + f : 'File text' }));
      card.appendChild(diffTabs(a, b, f || 'file'));
      if (S.edit && f !== 'key') {
        var name = idx + ':' + (f || '');
        var fld = S.edit.fields[name];
        var ta = el('textarea', { id: 'edit-' + idx + '-' + (f || 'text'), data: { testid: 'edit-' + idx + '-' + (f || 'text') }, 'aria-label': 'Edit after ' + (f || 'text') + ' of change ' + (idx + 1) });
        ta.value = fld.touched ? fld.value : b;
        ta.addEventListener('input', function () {
          fld.touched = true; fld.value = ta.value; updateActions();
        });
        card.appendChild(el('label', { for: ta.id, text: 'Edit after: ' + (f || 'text') }));
        card.appendChild(ta);
      }
    });
    return card;
  }

  function renderDetail() {
    var box = $('detail'); clear(box);
    if (!S.session) { box.appendChild(el('p', { text: 'Open a packet or session to begin.' })); return; }
    if (!S.selected || !record(S.selected)) { box.appendChild(el('p', { text: 'Select a proposal from the queue.' })); return; }
    var rec = record(S.selected); var p = current(rec); var fm = findingMap();
    box.appendChild(el('h2', { tabindex: '-1', text: 'Proposal ' + p.id + ' (revision ' + p.revision + ')' }));
    var led = S.ledger[rec.proposal_id];
    box.appendChild(dl([['decision', rec.decision], ['accepted digest', rec.accepted_digest === null ? '(none)' : rec.accepted_digest],
      ['depends on', p.depends_on.length ? p.depends_on.join(', ') : '(none)'],
      ['export', led ? led.status + (led.reason ? ' - ' + led.reason : '') : '(pending)']]));

    /* actions */
    var safe = Core.approvalSafe(p);
    var bar = el('div', { class: 'row', role: 'group', 'aria-label': 'Review actions' });
    function btn(action, label, extra) {
      return el('button', Object.assign({ type: 'button', text: label, data: { action: action, testid: action } }, extra || {}));
    }
    var acts = {};
    ['accept', 'edit', 'save-edit', 'reverify', 'cancel-edit', 'request-revision', 'reject', 'defer'].forEach(function (a) {
      acts[a] = btn(a, { accept: 'Accept', edit: 'Edit', 'save-edit': 'Save edit', reverify: 'Reverify and save revision', 'cancel-edit': 'Cancel edit', 'request-revision': 'Request revision', reject: 'Reject', defer: 'Defer' }[a]);
      acts[a].addEventListener('click', function () { act(a); });
      if ((a === 'save-edit' || a === 'reverify' || a === 'cancel-edit') && !S.edit) { acts[a].hidden = true; }
      if (a === 'edit' && S.edit) { acts[a].hidden = true; }
      bar.appendChild(acts[a]);
    });
    var noteIn = el('input', { type: 'text', id: 'note', data: { testid: 'note' } });
    noteIn.value = S.note;
    noteIn.addEventListener('input', function () { S.note = noteIn.value; });
    bar.appendChild(el('label', { for: 'note', text: 'Note (optional)' }));
    bar.appendChild(noteIn);
    box.appendChild(bar);
    var warn = el('p', { class: 'warn', id: 'dirty-warning', data: { testid: 'dirty-warning' } });
    box.appendChild(warn);
    var why = el('p', { class: 'note', data: { testid: 'accept-reason' } });
    box.appendChild(why);
    S.refs = { acts: acts, warn: warn, why: why, safe: safe };

    if (S.edit) {
      box.appendChild(verifyPanel(fm));
    }

    /* changes */
    box.appendChild(el('h3', { text: 'Changes (' + p.changes.length + ')' }));
    p.changes.forEach(function (c, i) { box.appendChild(changeNode(c, i, rec, p)); });

    /* evidence */
    box.appendChild(el('h3', { text: 'Evidence' }));
    var ev = el('div', { data: { testid: 'evidence' } });
    p.finding_ids.forEach(function (id) {
      var f = fm[id]; var card = el('div', { class: 'card' });
      card.appendChild(el('h4', { text: 'Finding ' + id + (f ? ' (' + f.type + ')' : '') }));
      if (f) {
        card.appendChild(dl([['rationale', f.rationale], ['sources', f.source_ids.join(', ')], ['limitations', f.limitations.length ? f.limitations.join(' | ') : '(none)']]));
        f.excerpts.forEach(function (x) { card.appendChild(el('div', null, [el('div', { class: 'note', text: 'Excerpt from ' + x.source_id }), el('pre', { text: x.text })])); });
        card.appendChild(el('h4', { text: 'Typed evidence' }));
        f.evidence.forEach(function (o) { card.appendChild(obsNode(o)); });
      }
      ev.appendChild(card);
    });
    box.appendChild(ev);
    box.appendChild(el('h3', { text: 'Preservation assessments (revision ' + p.revision + ')' }));
    var as = el('div', { data: { testid: 'assessments' } });
    if (p.assessments.length === 0) { as.appendChild(el('p', { class: 'warn', text: 'No assessments on this revision; acceptance is blocked.' })); }
    p.assessments.forEach(function (a) {
      var card = el('div', { class: 'card' }, [el('h4', { text: 'Finding ' + a.finding_id + ' (assessment revision ' + a.revision + ')' })]);
      a.preservation_checklist.forEach(function (c) {
        card.appendChild(dl([['item', c.item], ['preserved', String(c.preserved)], ['explanation', c.explanation]]));
      });
      a.evidence.forEach(function (o) { card.appendChild(obsNode(o)); });
      as.appendChild(card);
    });
    box.appendChild(as);

    box.appendChild(el('h3', { text: 'Sources and coverage' }));
    var pk = packet();
    box.appendChild(el('details', { class: 'card' }, [el('summary', { text: 'Inventory coverage' }), coverageDl(pk.inventory.coverage)]));
    proposalSourceIds(p).forEach(function (s) { box.appendChild(sourceCard(s, 'Source')); });

    box.appendChild(el('h3', { text: 'History' }));
    var h = el('ol', { data: { testid: 'history' } });
    rec.history.forEach(function (e) { h.appendChild(el('li', { text: 'rev ' + e.revision + ' ' + e.action + (e.note ? ' - ' + e.note : '') + ' [' + e.digest + ']' })); });
    if (!rec.history.length) { h.appendChild(el('li', { text: '(no events)' })); }
    box.appendChild(h);
    updateActions();
  }

  function updateActions() {
    if (!S.refs || !S.session || !S.selected) { return; }
    var r = S.refs; var dirty = isDirty(); var rec = record(S.selected); var p = current(rec);
    r.acts.accept.disabled = !(Core.approvalSafe(p)) || dirty || !!S.edit || S.busy;
    r.acts['save-edit'].disabled = !dirty || S.busy;
    r.acts.reverify.disabled = S.busy;
    ['request-revision', 'reject', 'defer', 'edit'].forEach(function (a) { r.acts[a].disabled = S.busy || (a !== 'edit' && dirty); });
    r.warn.textContent = dirty ? 'Unsaved edits; prior approval will be revoked on save.' : '';
    r.why.textContent = !Core.approvalSafe(p) ? 'Accept disabled: preservation assessments are missing, incomplete or not all preserved.' :
      (dirty || S.edit) ? 'Accept disabled while editing.' : '';
  }

  async function act(action) {
    if (S.busy || !S.session || !S.selected) { return; }
    var gen = S.gen; var id = S.selected; var rec = record(id); var p = current(rec);
    showError('');
    if (action === 'edit') {
      S.edit = { fields: {}, verify: buildVerify(p, findingMap()), vInit: '' };
      S.edit.vInit = vSerial();
      p.changes.forEach(function (c, i) {
        if (c.operation !== 'edit' || c.after === null) { return; }
        if (typeof c.after === 'string') { S.edit.fields[i + ':'] = { touched: false, value: '', norm: c.after.replace(/\r\n?/g, '\n') }; }
        else { MEMORY_FIELDS.forEach(function (f) { if (f !== 'key') { S.edit.fields[i + ':' + f] = { touched: false, value: '', norm: c.after[f].replace(/\r\n?/g, '\n') }; } }); }
      });
      renderDetail();
      var vp0 = $('detail').querySelector('[data-testid="verify-panel"]');
      var fld = vp0 && vp0.querySelector('input:not([disabled]):not([type="hidden"]),textarea:not([disabled]),select:not([disabled])');
      if (fld) { fld.focus(); } else { focusAction('save-edit'); }
      say('Editing ' + id + '. Save edit or cancel.'); return;
    }
    if (action === 'cancel-edit') { S.edit = null; renderDetail(); focusAction('edit'); say('Edit cancelled; nothing changed.'); return; }
    var wasInApp = inApp();
    S.busy = true; updateActions();
    try {
      var next; var opts = { note: S.note };
      var wire = action.replace(/-/g, '_');
      if (action === 'save-edit' || action === 'reverify') {
        var ep = clone(p); ep.revision = p.revision + 1; ep.assessments = [];
        if (action === 'reverify') { ep.assessments = verifiedAssessments(p, ep.revision); }
        ep.changes.forEach(function (c, i) {
          if (c.operation !== 'edit' || c.after === null) { return; }
          if (typeof c.after === 'string') {
            var f0 = S.edit.fields[i + ':'];
            if (f0 && f0.touched && f0.value !== f0.norm) { c.after = f0.value; }
          } else {
            MEMORY_FIELDS.forEach(function (f) {
              var fl = S.edit.fields[i + ':' + f];
              if (fl && fl.touched && fl.value !== fl.norm) { c.after[f] = fl.value; }
            });
          }
        });
        opts.editedProposal = ep; wire = 'edit';
      }
      next = await Core.reviewProposal(S.session, id, wire, opts);
      if (gen !== S.gen || id !== S.selected) { say(action + ' for ' + id + ' was discarded because the selection changed; repeat it.'); return; }
      S.session = next; S.edit = null; S.gen++;
      afterSessionChange();
      if (wasInApp) { S.busy = false; updateActions(); focusAction(action); }
      say(action + ' recorded for ' + id + '. Save the session to keep it.');
    } catch (e) {
      if (gen === S.gen) { showError(errText(e)); say(action + ' failed.'); }
      else { say(action + ' for ' + id + ' failed after the context changed: ' + errText(e)); }
    } finally {
      S.busy = false; updateActions();
    }
  }

  /* ------------------------------------------------------------ downloads */
  function download(bytes, name) {
    var blob = new Blob([bytes], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = name; a.rel = 'noopener';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  async function saveSession() {
    try {
      var sealed = await Core.sealDocument(S.session);
      download(Core.dumps(sealed), 'reconciliation-session.json');
      say('Session downloaded' + (isDirty() ? ' (unsaved edit buffers not included).' : '.'));
    } catch (e) { showError(errText(e)); }
  }
  async function exportPlan() {
    try {
      var plan = await Core.exportPlan(S.session);
      await Core.validatePlan(plan);
      download(Core.dumps(plan), 'reconciliation-plan.json');
      say('Plan downloaded (' + plan.approved.length + ' approved). Nothing executed or validated against sources.');
    } catch (e) { showError(errText(e)); }
  }

  /* ----------------------------------------------------------------- init */
  function init() {
    $('file-input').addEventListener('change', function (ev) {
      var f = ev.target.files && ev.target.files[0];
      if (f) { importFile(f).then(function () { ev.target.value = ''; }); }
    });
    ['kind', 'finding', 'decision'].forEach(function (k) {
      $('filter-' + k).addEventListener('change', function (ev) { S.filters[k] = ev.target.value; S.showAll = false; renderQueue(); });
    });
    document.querySelector('[data-action="reset-filters"]').addEventListener('click', function () {
      S.filters = { kind: '', finding: '', decision: '' }; buildFilterOptions(); renderQueue(); say('Filters reset.');
    });
    document.querySelector('[data-action="save-session"]').addEventListener('click', saveSession);
    document.querySelector('[data-action="export-plan"]').addEventListener('click', exportPlan);
    window.addEventListener('beforeunload', function (ev) { if (isDirty()) { ev.preventDefault(); ev.returnValue = ''; } });
    buildFilterOptions();
  }
  init();
}());
