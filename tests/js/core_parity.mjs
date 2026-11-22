// Standalone (dependency-free) parity checks for nachos_core/reconciliation/viewer/core.js.
// usage: node core_parity.mjs <repo-root>
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';

const root = path.resolve(process.argv[2] || '.');
const require = createRequire(import.meta.url);
const coreFile = path.join(root, 'nachos_core/reconciliation/viewer/core.js');
const fixDir = path.join(root, 'tests/fixtures/reconciliation');
const C = require(coreFile);
const enc = new TextEncoder();
const bytesOf = (s) => enc.encode(s);
const read = (name) => fs.readFileSync(path.join(fixDir, name));
const dec = (u8) => new TextDecoder().decode(u8);
const hex = (u8) => Buffer.from(u8).toString('hex');

let passed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed++;
  } catch (e) {
    failures.push(name + ': ' + (e && e.stack ? e.stack : e));
  }
}
function eq(a, b, msg) {
  if (a !== b) throw new Error((msg || 'not equal') + ': ' + String(a).slice(0, 200) + ' !== ' + String(b).slice(0, 200));
}
function ok(c, msg) {
  if (!c) throw new Error(msg || 'assertion failed');
}
async function rejects(fn, msg) {
  try {
    await fn();
  } catch (e) {
    ok(e instanceof Error, 'must throw Error');
    return e;
  }
  throw new Error('expected rejection: ' + (msg || ''));
}

// ---------------------------------------------------------------- surface
await test('api surface and version', () => {
  const names = ['parseStrict', 'canonicalBytes', 'digest', 'sha256Hex', 'fileSha256', 'memorySha256',
    'validateDocument', 'dumps', 'sealDocument', 'newSession', 'reviewProposal', 'exportPlan',
    'validatePlan', 'approvalSafe', 'proposalDigest', 'reviewContext', 'validateValidationResult',
    'lineDiff'];
  for (const n of names) eq(typeof C[n], 'function', n);
  eq(C.version, 'v1');
  eq(C.LIMITS.MAX_BYTES, 8388608);
  eq(C.LIMITS.MAX_DEPTH, 32);
  eq(C.LIMITS.MAX_NODES, 100000);
  eq(C.LIMITS.MAX_ITEMS, 10000);
});

await test('source is ASCII, inline-script safe and free of dynamic code or DOM', () => {
  const src = fs.readFileSync(coreFile, 'utf8');
  ok(/^[\x00-\x7f]*$/.test(src), 'non-ASCII byte');
  ok(!/<\/script/i.test(src) && !/<!--/.test(src) && !/-->/.test(src), 'inline script hazard');
  for (const bad of [/\beval\s*\(/, /\bnew\s+Function\b/, /\bFunction\s*\(/, /\bimport\s*\(/, /\bdocument\./,
    /\bwindow\b/, /\blocalStorage\b/, /\bsessionStorage\b/, /\bfetch\s*\(/, /XMLHttpRequest/, /\brequire\s*\(/,
    /\bimport\s+[\w{*]/, /\bsetTimeout\b/, /\bWebSocket\b/]) {
    ok(!bad.test(src), 'forbidden construct ' + bad);
  }
});

await test('loads in a module-less global context and sets NachosCore', async () => {
  const sandbox = { TextEncoder, TextDecoder, crypto: globalThis.crypto, Uint8Array, ArrayBuffer };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(coreFile, 'utf8'), sandbox);
  ok(sandbox.NachosCore && typeof sandbox.NachosCore.parseStrict === 'function');
  const v = sandbox.NachosCore.parseStrict(bytesOf('{"b":1,"a":[true,null]}'));
  eq(dec(sandbox.NachosCore.canonicalBytes(v)), '{"a":[true,null],"b":1}');
  eq(await sandbox.NachosCore.fileSha256('a'),
    'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb');
});

// ------------------------------------------------------------ hash vectors
const vectors = JSON.parse(read('hash-vectors.json').toString('utf8'));

await test('canonical vectors: bytes and proposal-domain digests', async () => {
  for (const v of vectors.canonical) {
    const value = C.parseStrict(bytesOf(JSON.stringify(v.value)));
    eq(hex(C.canonicalBytes(value)), v.utf8_hex);
    eq(await C.digest('proposal', value), v.proposal_domain_digest);
  }
});

await test('canonical vector: key order is by code point not UTF-16 or integer-like', () => {
  const value = C.parseStrict(bytesOf('{"\\ud83d\\ude00":"astral","\\ue000":"BMP","2":"two","10":"ten"}'));
  eq(dec(C.canonicalBytes(value)), '{"10":"ten","2":"two","":"BMP","\u{1f600}":"astral"}');
  // Also works for plain JSON.parse objects (integer-like key enumeration order differs).
  eq(dec(C.canonicalBytes(JSON.parse('{"b":1,"2":1,"1":1,"a":1}'))), '{"1":1,"2":1,"a":1,"b":1}');
});

await test('domains are separated and unknown domains are rejected', async () => {
  const a = await C.digest('proposal', { x: 1 });
  const b = await C.digest('packet', { x: 1 });
  ok(a !== b);
  await rejects(() => C.digest('other', {}));
  await rejects(() => C.digest('__proto__', {}));
});

await test('memorySha256 requires exactly five string fields', async () => {
  const rec = { key: 'k', title: 't', summary: 's', category: 'c', body: 'b' };
  eq((await C.memorySha256(rec)).length, 64);
  await rejects(() => C.memorySha256({ ...rec, extra: 'x' }));
  await rejects(() => C.memorySha256({ key: 'k' }));
  await rejects(() => C.memorySha256({ ...rec, body: 1 }));
  await rejects(() => C.memorySha256([]));
});

await test('fileSha256 rejects lone surrogates instead of hashing replacement text', async () => {
  await rejects(() => C.fileSha256('\ud800'));
});

// ---------------------------------------------------------------- fixtures
const names = fs.readdirSync(fixDir).filter((n) => n.endsWith('.json') && n !== 'hash-vectors.json');
const docs = {};
await test('every fixture parses strictly, validates and re-serializes byte-exact', async () => {
  for (const n of names) {
    const raw = read(n);
    const doc = await C.validateDocument(C.parseStrict(raw));
    eq(dec(C.dumps(doc)) + '\n', raw.toString('utf8'), n);
    docs[n] = doc;
  }
});

await test('fixture proposal digests match Python vectors', async () => {
  const packet = docs['packet.json'];
  for (const p of packet.proposals) eq(await C.proposalDigest(packet, p), vectors.proposal_digests[p.id]);
});

await test('lifecycle reproduces the Python fixture session snapshots and plans byte-for-byte', async () => {
  const same = (doc, name) => eq(dec(C.dumps(doc)) + '\n', read(name).toString('utf8'), name);
  const packet = docs['packet.json'];
  let s = await C.newSession(packet);
  same(s, 'session-initial.json');
  s = await C.reviewProposal(s, 'p-memory', 'accept', { note: 'Checked all five fields and authority clauses.' });
  const edited = docs['session-edit-invalidated.json'].reviews[0].revisions[1];
  s = await C.reviewProposal(s, 'p-memory', 'edit', { editedProposal: edited, note: 'Summary clarified; approval revoked.' });
  same(s, 'session-edit-invalidated.json');
  s = await C.reviewProposal(s, 'p-memory', 'accept', { note: 'Explicitly reaccepted revision 2.' });
  s = await C.reviewProposal(s, 'p-skill', 'request_revision', { note: 'Recheck preservation before acceptance.' });
  same(s, 'session-revision-requested.json');
  s = await C.reviewProposal(s, 'p-skill', 'reject', { note: 'Retain original skill for this review.' });
  s = await C.reviewProposal(s, 'p-disable', 'defer', { note: 'Proposal only; no skill state changed.' });
  same(s, 'session-reviewed.json');
  const plan = await C.exportPlan(s);
  same(plan, 'plan.json');
  eq(plan.execution, 'not_performed');
  eq(plan.source_validation, 'not_performed');
  const d = await C.reviewProposal(await C.newSession(packet), 'p-disable', 'accept',
    { note: 'Accept proposal only, not permission to execute.' });
  same(await C.exportPlan(d), 'plan-disable-proposal-only.json');
});

await test('empty packet makes an empty session and plan', async () => {
  const s = await C.newSession(docs['packet-empty.json']);
  eq(s.reviews.length, 0);
  const plan = await C.exportPlan(s);
  eq(plan.approved.length, 0);
  eq(plan.ledger.length, 0);
});

await test('lifecycle rules: accept gate, revocation, immutability of inputs', async () => {
  const packet = docs['packet.json'];
  const s0 = await C.newSession(packet);
  const before = dec(C.dumps(s0));
  const s1 = await C.reviewProposal(s0, 'p-memory', 'accept');
  eq(dec(C.dumps(s0)), before, 'input mutated');
  const rec = s1.reviews[0];
  eq(rec.decision, 'accept');
  ok(/^[0-9a-f]{64}$/.test(rec.accepted_digest));
  const s2 = await C.reviewProposal(s1, 'p-memory', 'request_revision');
  eq(s2.reviews[0].revisions.length, 2);
  eq(s2.reviews[0].revisions[1].assessments.length, 0);
  eq(s2.reviews[0].accepted_digest, null);
  await rejects(() => C.reviewProposal(s2, 'p-memory', 'accept'), 'accept without assessments');
  const s3 = await C.reviewProposal(s2, 'p-memory', 'reject');
  eq(s3.reviews[0].decision, 'reject');
  await rejects(() => C.reviewProposal(s0, 'nope', 'accept'));
  await rejects(() => C.reviewProposal(s0, 'p-memory', 'approve'));
  await rejects(() => C.reviewProposal(s0, 'p-memory', 'defer', { note: 5 }));
  await rejects(() => C.reviewProposal(s0, 'p-memory', 'edit'));
  await rejects(() => C.reviewProposal(s0, 'p-memory', 'defer', { editedProposal: s0.reviews[0].revisions[0] }));
  // False preservation blocks accept.
  const bad = JSON.parse(JSON.stringify(packet));
  ok(C.approvalSafe(bad.proposals[0]));
  bad.proposals[0].assessments[0].preservation_checklist[0].preserved = false;
  ok(!C.approvalSafe(bad.proposals[0]));
  bad.proposals[0].assessments = [];
  ok(!C.approvalSafe(bad.proposals[0]));
});

await test('validatePlan accepts exported plans only', async () => {
  const plan = await C.validatePlan(docs['plan.json']);
  eq(plan.approved.length, 1);
  await rejects(() => C.validatePlan(docs['packet.json']));
  await rejects(() => C.exportPlan(docs['packet.json']));
});

// -------------------------------------------------------------- negatives
function cleanText() {
  return read('packet.json').toString('utf8').trimEnd();
}

await test('lexical negatives from the fixture vector file are rejected', () => {
  for (const t of vectors.invalid_json) {
    let threw = false;
    try { C.parseStrict(bytesOf(t)); } catch (e) { threw = e instanceof Error; }
    ok(threw, t);
  }
});

await test('strict parser negatives', () => {
  const bad = [
    '{"a":1,"a":2}', '{"a":1,"b":{"c":1,"c":1}}', '{"__proto__":1,"__proto__":2}',
    '1.0', '1e0', '1E2', '-1.5', '0.0', '1e-2', '{"n":1.}', '[01]', '[-]', '[+1]', '.5', '[1,]', '{"a":1,}', '{,}',
    'NaN', '[NaN]', 'Infinity', '-Infinity', '[Infinity]', '9007199254740992', '-9007199254740992',
    '12345678901234567890', '[1 2]', '{"a" 1}', "{'a':1}", '"\\ud800"', '"\\udc00"', '"\\ud800\\ud800"',
    '"\\ud83dx"', '"\\u12"', '"\\x41"', '"\\"', '"a\nb"', '"a\tb"', '"\u0001"', 'tru', 'nul', 'truex',
    '', '   ', '{}{}', '[]]', '{"a":1', '[1', '"abc', '// c\n1', '{"a":/*x*/1}', '  1', '[1] ', ' 1',
  ];
  for (const t of bad) {
    let threw = false;
    try { C.parseStrict(bytesOf(t)); } catch (e) { threw = e instanceof Error; }
    ok(threw, 'should reject ' + JSON.stringify(t));
  }
  const good = ['0', '-0', '[ ]', ' \t\r\n{ "a" : [ 1 , -2 , 9007199254740991 , -9007199254740991 ] } \n',
    '"\\ud83d\\ude00"', '"\\u00e9\\/"', 'true', 'null', '{"":{}}'];
  for (const t of good) C.parseStrict(bytesOf(t));
  eq(Object.is(C.parseStrict(bytesOf('-0')), 0), true);
  eq(C.parseStrict(bytesOf('"\\ud83d\\ude00"')), '\u{1f600}');
});

await test('invalid UTF-8 and BOM', () => {
  const badBytes = [[0xff], [0xc3], [0xc0, 0x80], [0xed, 0xa0, 0x80], [0xf4, 0x90, 0x80, 0x80], [0x22, 0xe9, 0x22]];
  for (const b of badBytes) {
    let threw = false;
    try { C.parseStrict(Uint8Array.from(b)); } catch (e) { threw = e instanceof Error; }
    ok(threw, 'bad utf8 ' + b);
  }
  let threw = false;
  try { C.parseStrict(Uint8Array.from([0xef, 0xbb, 0xbf, 0x7b, 0x7d])); } catch (e) { threw = true; }
  ok(threw, 'BOM must be rejected like Python');
  let t2 = false;
  try { C.parseStrict('{}'); } catch (e) { t2 = true; }
  ok(t2, 'strings are not accepted');
});

await test('limits: depth, nodes, entries, bytes', () => {
  const nest = (n) => bytesOf('['.repeat(n) + ']'.repeat(n));
  C.parseStrict(nest(33)); // deepest value at depth 32
  C.canonicalBytes(C.parseStrict(nest(33)));
  let t = false;
  try { C.parseStrict(nest(34)); } catch (e) { t = true; }
  ok(t, 'depth 33');
  const zeros = (n) => bytesOf('[' + new Array(n).fill('0').join(',') + ']');
  C.parseStrict(zeros(10000));
  t = false;
  try { C.parseStrict(zeros(10001)); } catch (e) { t = true; }
  ok(t, '10001 items');
  const rows = (n, m) => bytesOf('[' + new Array(n).fill('[' + new Array(m).fill('0').join(',') + ']').join(',') + ']');
  C.parseStrict(rows(9, 10000)); // 1 + 9 + 90000 = 90010 nodes
  t = false;
  try { C.parseStrict(rows(10, 10000)); } catch (e) { t = true; } // 100011 nodes
  ok(t, 'node limit');
  const keys = (n) => { const p = []; for (let i = 0; i < n; i++) p.push('"k' + i + '":0'); return bytesOf('{' + p.join(',') + '}'); };
  C.parseStrict(keys(10000));
  t = false;
  try { C.parseStrict(keys(10001)); } catch (e) { t = true; }
  ok(t, '10001 keys');
  const big = (n) => bytesOf('"' + 'a'.repeat(n) + '"');
  C.canonicalBytes(C.parseStrict(big(8388606)));
  t = false;
  try { C.parseStrict(big(8388607)); } catch (e) { t = true; }
  ok(t, 'raw byte limit');
  t = false;
  try { C.canonicalBytes('a'.repeat(8388607)); } catch (e) { t = true; }
  ok(t, 'canonical byte limit');
  // canonicalBytes enforces limits on in-memory values as well.
  let deep = [];
  for (let i = 0; i < 40; i++) deep = [deep];
  t = false;
  try { C.canonicalBytes(deep); } catch (e) { t = true; }
  ok(t, 'canonical depth');
  const cyc = []; cyc.push(cyc);
  t = false;
  try { C.canonicalBytes(cyc); } catch (e) { t = true; }
  ok(t, 'cycle');
  for (const bad of [1.5, NaN, Infinity, undefined, () => 1, 10n, Symbol('x'), new Date(), new Map(), 2 ** 53]) {
    t = false;
    try { C.canonicalBytes({ a: bad }); } catch (e) { t = true; }
    ok(t, 'non-JSON value ' + String(typeof bad));
  }
  t = false;
  try { C.canonicalBytes('\ud800'); } catch (e) { t = true; }
  ok(t, 'lone surrogate');
});

await test('prototype pollution is impossible', async () => {
  const text = '{"__proto__":{"polluted":"yes"},"constructor":{"prototype":{"polluted2":"yes"}},"toString":1}';
  const v = C.parseStrict(bytesOf(text));
  eq(Object.getPrototypeOf(v), null);
  ok(Object.prototype.hasOwnProperty.call(v, '__proto__'));
  eq(v.__proto__.polluted, 'yes');
  eq({}.polluted, undefined);
  eq({}.polluted2, undefined);
  eq(dec(C.canonicalBytes(v)), '{"__proto__":{"polluted":"yes"},"constructor":{"prototype":{"polluted2":"yes"}},"toString":1}');
  // A pre-existing plain object (e.g. from JSON.parse) is also safe.
  const plain = JSON.parse(text);
  eq(dec(C.canonicalBytes(plain)), dec(C.canonicalBytes(v)));
  // Pollution via document validation.
  const packet = C.parseStrict(read('packet.json'));
  packet.__proto__ = { polluted3: 'yes' }; // sets the (null) prototype slot as an own key on a null-proto object
  await rejects(() => C.validateDocument(packet));
  const evil = C.parseStrict(bytesOf(cleanText().replace('{', '{"__proto__":{"polluted4":1},')));
  await rejects(() => C.validateDocument(evil));
  const evil2 = C.parseStrict(bytesOf(cleanText().replace('{', '{"constructor":{"prototype":{"polluted5":1}},')));
  await rejects(() => C.validateDocument(evil2));
  for (const k of ['polluted', 'polluted2', 'polluted3', 'polluted4', 'polluted5']) eq({}[k], undefined);
  // Validated output is detached and null-prototype.
  const ok1 = await C.validateDocument(C.parseStrict(read('packet.json')));
  eq(Object.getPrototypeOf(ok1), null);
  eq(Object.getPrototypeOf(ok1.inventory.sources[0]), null);
});

await test('validateDocument returns a detached deep copy', async () => {
  const raw = C.parseStrict(read('session-reviewed.json'));
  const checked = await C.validateDocument(raw);
  ok(checked !== raw && checked.packet !== raw.packet);
  checked.reviews[0].decision = 'reject';
  eq(raw.reviews[0].decision, 'accept');
});

function mutated(fn, name = 'packet.json') {
  const d = JSON.parse(read(name).toString('utf8'));
  fn(d);
  return C.parseStrict(bytesOf(JSON.stringify(d)));
}

await test('tampered artifacts without resealing are rejected', async () => {
  const cases = [
    ['packet.json', (d) => { d.findings[0].rationale += 'x'; }],
    ['packet.json', (d) => { d.digest = 'f'.repeat(64); }],
    ['packet.json', (d) => { d.digest = d.digest.toUpperCase(); }],
    ['packet.json', (d) => { d.inventory.digest = '0'.repeat(64); }],
    ['packet.json', (d) => { d.inventory.sources[0].before.body += 'x'; }],
    ['packet.json', (d) => { d.extra = 1; }],
    ['packet.json', (d) => { delete d.profile; }],
    ['packet.json', (d) => { d.schema = 'nachos.reconciliation.packet.v2'; }],
    ['session-reviewed.json', (d) => { d.reviews[0].history[0].note = 'changed'; }],
    ['session-reviewed.json', (d) => { d.reviews[0].history[0].digest = '0'.repeat(64); }],
    ['session-reviewed.json', (d) => { d.packet.digest = '0'.repeat(64); }],
    ['session-reviewed.json', (d) => { d.reviews[0].accepted_digest = null; }],
    ['plan.json', (d) => { d.approved[0].review_digest = '0'.repeat(64); }],
    ['plan.json', (d) => { d.execution = 'performed'; }],
    ['plan.json', (d) => { d.source_validation = 'valid'; }],
    ['plan.json', (d) => { d.ledger[0].status = 'accepted'; }],
  ];
  for (const [name, fn] of cases) await rejects(() => C.validateDocument(mutated(fn, name)), name);
});

await test('wrong profile binding is rejected, correct one accepted', async () => {
  const raw = C.parseStrict(read('packet.json'));
  await C.validateDocument(raw, { id: 'synthetic-review', root: '/inert/synthetic-profile' });
  await rejects(() => C.validateDocument(raw, { id: 'synthetic-review', root: '/other' }));
  await rejects(() => C.validateDocument(raw, { id: 'x', root: '/inert/synthetic-profile' }));
  await rejects(() => C.validateDocument(raw, { id: 'synthetic-review', root: '/inert/synthetic-profile', z: 1 }));
});

await test('errors never echo document content', async () => {
  const secret = 'SECRET-TOKEN-12345';
  const e1 = await rejects(() => C.validateDocument({ schema: secret }));
  ok(!e1.message.includes(secret));
  const e2 = await rejects(() => C.validateDocument(mutated((d) => { d.findings[0].rationale = secret; })));
  ok(!e2.message.includes(secret));
  for (const t of ['{"' + secret + '":1,"' + secret + '":2}', '[' + secret + ']']) {
    try { C.parseStrict(bytesOf(t)); } catch (e) { ok(!e.message.includes(secret) && e.message.length < 80); }
  }
});

// ------------------------------------------------------------ line diff
function lcsLen(x, y) {
  const dp = Array.from({ length: x.length + 1 }, () => new Array(y.length + 1).fill(0));
  for (let i = 1; i <= x.length; i++)
    for (let j = 1; j <= y.length; j++)
      dp[i][j] = x[i - 1] === y[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  return dp[x.length][y.length];
}

function checkDiff(a, b, opts) {
  const r = C.lineDiff(a, b, opts);
  eq(r.fallback, false, 'unexpected fallback');
  const left = r.rows.filter((x) => x.type !== 'add').map((x) => x.a).join('\n');
  const right = r.rows.filter((x) => x.type !== 'del').map((x) => x.b).join('\n');
  eq(left, a, 'left reconstruction');
  eq(right, b, 'right reconstruction');
  for (const row of r.rows) {
    if (row.type === 'eq') eq(row.a, row.b);
    if (row.type === 'eq') ok('a' in row && 'b' in row);
    if (row.type === 'del') ok('a' in row && !('b' in row));
    if (row.type === 'add') ok('b' in row && !('a' in row));
  }
  const eqs = r.rows.filter((x) => x.type === 'eq').length;
  eq(eqs, lcsLen(a.split('\n'), b.split('\n')), 'not a minimal diff');
  return r;
}

await test('lineDiff basics, CRLF and bounds', () => {
  checkDiff('', '');
  checkDiff('a', 'a');
  checkDiff('a\nb\nc', 'a\nc');
  checkDiff('a\r\nb\r\n', 'a\r\nc\r\n');
  const r = checkDiff('x\r\ny', 'x\ny');
  eq(r.rows[0].type, 'del');
  eq(r.rows[0].a, 'x\r');
  checkDiff('', 'a\nb');
  checkDiff('a\nb', '');
  checkDiff('é\né', 'é\né');
  const lines = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join('\n');
  const f1 = C.lineDiff(lines(2001, (i) => 'a' + i), 'x', {});
  eq(f1.fallback, true);
  eq(f1.rows.length, 0);
  eq(C.lineDiff(lines(2000, (i) => 'a' + i), lines(2000, (i) => (i % 100 ? 'a' + i : 'x' + i))).fallback, false);
  const f2 = C.lineDiff(lines(900, (i) => 'a' + i), lines(900, (i) => 'b' + i), { maxOps: 1000 });
  eq(f2.fallback, true);
  eq(C.lineDiff('a\nb', 'a\nc', { maxLines: 1 }).fallback, true);
  // Default op budget: two totally different 2000 line texts still resolve or fall back, quickly.
  const started = Date.now();
  const big = C.lineDiff(lines(2000, (i) => 'p' + i), lines(2000, (i) => 'q' + i));
  ok(Date.now() - started < 5000);
  ok(big.fallback === true || big.rows.length === 4000);
});

await test('lineDiff random differential against LCS', () => {
  let seed = 12345;
  const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let t = 0; t < 300; t++) {
    const mk = () => Array.from({ length: rnd(14) }, () => 'ab\r'.charAt(rnd(3)) + 'xyz'.charAt(rnd(2))).join('\n');
    checkDiff(mk(), mk());
  }
});

// ---------------------------------------------------- validation results
await test('validateValidationResult structural rules', async () => {
  const plan = docs['plan.json'];
  const src = plan.approved[0].sources[0];
  const good = {
    schema: 'nachos.reconciliation.validation.v1',
    plan_digest: plan.digest,
    profile: plan.profile,
    checked_at: '2026-11-16T13:00:00+00:00',
    status: 'valid_at_check',
    sources: [{ identity: src.identity, proposal_ids: ['p-memory'], expected_sha256: src.sha256,
      actual_sha256: src.sha256, status: 'unchanged', reason: 'sha256_match' }],
    limitations: ['a', 'b'],
  };
  const checked = C.validateValidationResult(good, plan);
  ok(checked !== good);
  const variants = [
    (r) => { r.extra = 1; }, (r) => { r.schema = 'x'; }, (r) => { r.plan_digest = '0'.repeat(64); },
    (r) => { r.checked_at = '2026-11-16T13:00:00'; }, (r) => { r.checked_at = 'now'; },
    (r) => { r.status = 'stale'; }, (r) => { r.status = 'ok'; }, (r) => { r.sources = []; },
    (r) => { r.sources[0].status = 'changed'; }, (r) => { r.sources[0].extra = 1; },
    (r) => { r.sources[0].identity = 'other'; }, (r) => { r.sources[0].proposal_ids = []; },
    (r) => { r.sources[0].expected_sha256 = '1'.repeat(64); }, (r) => { r.limitations = [1]; },
    (r) => { r.profile = { id: 'a', root: 'b' }; }, (r) => { r.sources[0].reason = 'Bad Reason!'; },
  ];
  for (const fn of variants) {
    const bad = JSON.parse(JSON.stringify(good));
    fn(bad);
    let threw = false;
    try { C.validateValidationResult(bad, plan); } catch (e) { threw = e instanceof Error; }
    ok(threw, String(fn));
  }
});

if (failures.length) {
  console.error(failures.join('\n---\n'));
  console.error('FAILED ' + failures.length + ', passed ' + passed);
  process.exit(1);
}
console.log('core_parity ok: ' + passed + ' checks');
