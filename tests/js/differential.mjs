// Node half of the Python-driven differential test (tests/test_viewer_core_node.py).
// usage: node differential.mjs <core.js> <corpus.json> <out.json>
// Reads cases produced by Python, computes the JS result of each, writes them back.
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const [coreFile, corpusFile, outFile] = process.argv.slice(2);
const C = require(coreFile);
const corpus = JSON.parse(fs.readFileSync(corpusFile, 'utf8'));
const enc = new TextEncoder();

const fromHex = (h) => Uint8Array.from(h.match(/../g) || [], (x) => parseInt(x, 16));

async function summary(bytes) {
  const sha = await C.sha256Hex(bytes);
  return { sha, len: bytes.length, hex: bytes.length <= 4096 ? Buffer.from(bytes).toString('hex') : null };
}

function generate(gen) {
  const { kind, n } = gen;
  if (kind === 'str') return enc.encode('"' + 'a'.repeat(n) + '"');
  if (kind === 'zeros') return enc.encode('[' + new Array(n).fill('0').join(',') + ']');
  if (kind === 'nest') return enc.encode('['.repeat(n) + ']'.repeat(n));
  if (kind === 'keys') {
    const parts = [];
    for (let i = 0; i < n; i++) parts.push('"k' + i + '":0');
    return enc.encode('{' + parts.join(',') + '}');
  }
  if (kind === 'objnest') return enc.encode('{"a":'.repeat(n) + '0' + '}'.repeat(n));
  throw new Error('bad generator');
}

const rawOf = (c) => (c.gen ? generate(c.gen) : fromHex(c.raw_hex));
const fail = (e) => ({ ok: false, message: e && e.message });

const out = {};

out.canonical = [];
for (const c of corpus.canonical) {
  try {
    const value = C.parseStrict(rawOf(c));
    const bytes = C.canonicalBytes(value);
    const digests = {};
    for (const d of Object.keys(c.digests || {})) digests[d] = await C.digest(d, value);
    out.canonical.push({ ok: true, ...(await summary(bytes)), digests });
  } catch (e) {
    out.canonical.push(fail(e));
  }
}

out.docs = [];
for (const c of corpus.docs) {
  try {
    const value = C.parseStrict(fromHex(c.raw_hex));
    const checked = await C.validateDocument(value, c.expected_profile === null ? undefined : c.expected_profile);
    out.docs.push({ ok: true, ...(await summary(C.dumps(checked))) });
  } catch (e) {
    out.docs.push(fail(e));
  }
}

out.lifecycle = [];
for (const c of corpus.lifecycle) {
  const session = C.parseStrict(fromHex(c.session_hex));
  const steps = [];
  for (const s of c.steps) {
    try {
      const edited = s.edited_hex === null ? undefined : C.parseStrict(fromHex(s.edited_hex));
      const next = await C.reviewProposal(session, s.pid, s.action, { editedProposal: edited, note: s.note });
      steps.push({ ok: true, ...(await summary(C.dumps(next))) });
    } catch (e) {
      steps.push(fail(e));
    }
  }
  out.lifecycle.push(steps);
}

out.new_session = [];
for (const c of corpus.new_session) {
  try {
    const next = await C.newSession(C.parseStrict(fromHex(c.packet_hex)));
    out.new_session.push({ ok: true, ...(await summary(C.dumps(next))) });
  } catch (e) {
    out.new_session.push(fail(e));
  }
}

out.export = [];
for (const c of corpus.export) {
  try {
    const plan = await C.exportPlan(C.parseStrict(fromHex(c.session_hex)));
    const again = await C.validatePlan(plan);
    out.export.push({ ok: true, ...(await summary(C.dumps(again))) });
  } catch (e) {
    out.export.push(fail(e));
  }
}

out.proposal_digest = [];
for (const c of corpus.proposal_digest) {
  try {
    const packet = C.parseStrict(fromHex(c.packet_hex));
    const item = C.parseStrict(fromHex(c.proposal_hex));
    out.proposal_digest.push({ ok: true, digest: await C.proposalDigest(packet, item) });
  } catch (e) {
    out.proposal_digest.push(fail(e));
  }
}

out.review_context = [];
for (const c of corpus.review_context) {
  const packet = C.parseStrict(fromHex(c.packet_hex));
  const item = C.parseStrict(fromHex(c.proposal_hex));
  const ctx = C.reviewContext(packet, item);
  out.review_context.push({ digest: await C.digest('proposal', ctx), safe: C.approvalSafe(item) });
}

out.validation = [];
for (const c of corpus.validation) {
  try {
    const plan = C.parseStrict(fromHex(c.plan_hex));
    const result = C.parseStrict(fromHex(c.result_hex));
    const checked = C.validateValidationResult(result, plan);
    out.validation.push({ ok: true, same: Buffer.from(C.dumps(checked)).equals(Buffer.from(C.dumps(result))) });
  } catch (e) {
    out.validation.push(fail(e));
  }
}

fs.writeFileSync(outFile, JSON.stringify(out));
