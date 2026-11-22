/*
 * NachosCore: dependency-free browser/Node port of the Python reconciliation
 * contracts (hashing, codec, models, review, export, validation result shape).
 * Python (nachos_core/reconciliation) is authoritative; this file mirrors it.
 *
 * No modules, no eval, no DOM, no network. All errors are Error objects with
 * short fixed messages and never echo document content. Parsed and copied
 * documents use null-prototype objects (no __proto__ pollution).
 * Integrity hashes only: not signatures, not authentication.
 */
(function (root) {
  'use strict';

  var LIMITS = Object.freeze({
    MAX_BYTES: 8388608,
    MAX_DEPTH: 32,
    MAX_NODES: 100000,
    MAX_ITEMS: 10000,
    MAX_SAFE_INT: 9007199254740991,
    DIFF_MAX_LINES: 2000,
    DIFF_MAX_OPS: 1000000
  });

  var SCHEMAS = Object.freeze({
    inventory: 'nachos.reconciliation.inventory.v1',
    packet: 'nachos.reconciliation.packet.v1',
    session: 'nachos.reconciliation.session.v1',
    plan: 'nachos.reconciliation.plan.v1'
  });
  var VALIDATION_SCHEMA = 'nachos.reconciliation.validation.v1';
  var KIND_BY_SCHEMA = new Map([
    [SCHEMAS.inventory, 'inventory'],
    [SCHEMAS.packet, 'packet'],
    [SCHEMAS.session, 'session'],
    [SCHEMAS.plan, 'plan']
  ]);
  var DOMAINS = ['memory-record', 'proposal', 'packet', 'session', 'plan', 'inventory'];
  var MEMORY_FIELDS = ['key', 'title', 'summary', 'category', 'body'];
  var ACTIONS = ['accept', 'edit', 'request_revision', 'reject', 'defer'];

  var hasOwn = Object.prototype.hasOwnProperty;
  var encoder = new TextEncoder();

  function fail(message) {
    var error = new Error(message);
    error.nachos = true;
    throw error;
  }

  /* ---------------------------------------------------------------- basics */

  function isList(v) {
    return Array.isArray(v);
  }

  function isDict(v) {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) {
      return false;
    }
    var proto = Object.getPrototypeOf(v);
    return proto === null || proto === Object.prototype;
  }

  function isInt(v) {
    return typeof v === 'number' && Number.isSafeInteger(v);
  }

  function get(obj, key) {
    return hasOwn.call(obj, key) ? obj[key] : undefined;
  }

  function newObject() {
    return Object.create(null);
  }

  var LONE_SURROGATE = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/;

  function wellFormed(s) {
    return !LONE_SURROGATE.test(s);
  }

  function codePointLength(s) {
    var n = 0;
    for (var i = 0; i < s.length; i++) {
      var c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
        var d = s.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) {
          i++;
        }
      }
      n++;
    }
    return n;
  }

  /* Order strings by Unicode code points (Python str ordering). */
  function foldUnit(u) {
    if (u >= 0xe000) {
      return u - 0x800;
    }
    if (u >= 0xd800) {
      return u + 0x2000;
    }
    return u;
  }

  function cpCompare(a, b) {
    var n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) {
      var x = a.charCodeAt(i);
      var y = b.charCodeAt(i);
      if (x !== y) {
        return foldUnit(x) - foldUnit(y);
      }
    }
    return a.length - b.length;
  }

  function sortedStrings(values) {
    return values.slice().sort(cpCompare);
  }

  function deepEqual(a, b) {
    if (a === b) {
      return true;
    }
    if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') {
      return false;
    }
    var al = Array.isArray(a);
    if (al !== Array.isArray(b)) {
      return false;
    }
    if (al) {
      if (a.length !== b.length) {
        return false;
      }
      for (var i = 0; i < a.length; i++) {
        if (!deepEqual(a[i], b[i])) {
          return false;
        }
      }
      return true;
    }
    var ak = Object.keys(a);
    var bk = Object.keys(b);
    if (ak.length !== bk.length) {
      return false;
    }
    for (var j = 0; j < ak.length; j++) {
      if (!hasOwn.call(b, ak[j]) || !deepEqual(a[ak[j]], b[ak[j]])) {
        return false;
      }
    }
    return true;
  }

  /* ------------------------------------------------------- canonical JSON */

  function quote(s) {
    if (!wellFormed(s)) {
      fail('invalid canonical JSON');
    }
    return JSON.stringify(s);
  }

  function canonWalk(v, depth, state, out) {
    state.nodes -= 1;
    if (depth > LIMITS.MAX_DEPTH || state.nodes < 0) {
      fail('JSON depth/node limit exceeded');
    }
    if (v === null) {
      out.push('null');
      return;
    }
    var t = typeof v;
    if (t === 'string') {
      out.push(quote(v));
    } else if (t === 'boolean') {
      out.push(v ? 'true' : 'false');
    } else if (t === 'number') {
      if (!Number.isSafeInteger(v)) {
        fail('integer out of range');
      }
      out.push(String(v));
    } else if (t !== 'object') {
      fail('unsupported JSON type or collection limit exceeded');
    } else if (Array.isArray(v)) {
      if (v.length > LIMITS.MAX_ITEMS) {
        fail('unsupported JSON type or collection limit exceeded');
      }
      out.push('[');
      for (var i = 0; i < v.length; i++) {
        if (i > 0) {
          out.push(',');
        }
        if (!(i in v)) {
          fail('unsupported JSON type or collection limit exceeded');
        }
        canonWalk(v[i], depth + 1, state, out);
      }
      out.push(']');
    } else if (isDict(v)) {
      var keys = Object.keys(v);
      if (keys.length > LIMITS.MAX_ITEMS) {
        fail('unsupported JSON type or collection limit exceeded');
      }
      keys.sort(cpCompare);
      out.push('{');
      for (var j = 0; j < keys.length; j++) {
        if (j > 0) {
          out.push(',');
        }
        out.push(quote(keys[j]), ':');
        canonWalk(v[keys[j]], depth + 1, state, out);
      }
      out.push('}');
    } else {
      fail('unsupported JSON type or collection limit exceeded');
    }
  }

  function canonicalString(value) {
    var out = [];
    canonWalk(value, 0, { nodes: LIMITS.MAX_NODES }, out);
    return out.join('');
  }

  function canonicalBytes(value) {
    var raw = encoder.encode(canonicalString(value));
    if (raw.length > LIMITS.MAX_BYTES) {
      fail('JSON byte limit exceeded');
    }
    return raw;
  }

  /* ---------------------------------------------------------------- hashes */

  function subtle() {
    var c = root.crypto;
    if (!c || !c.subtle || typeof c.subtle.digest !== 'function') {
      fail('WebCrypto unavailable');
    }
    return c.subtle;
  }

  var HEX = [];
  for (var hexIndex = 0; hexIndex < 256; hexIndex++) {
    HEX.push((hexIndex < 16 ? '0' : '') + hexIndex.toString(16));
  }

  async function sha256Hex(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      fail('raw file content must be bytes');
    }
    var buf = await subtle().digest('SHA-256', bytes);
    var view = new Uint8Array(buf);
    var out = new Array(view.length);
    for (var i = 0; i < view.length; i++) {
      out[i] = HEX[view[i]];
    }
    return out.join('');
  }

  async function fileSha256(textValue) {
    if (typeof textValue !== 'string' || !wellFormed(textValue)) {
      fail('raw file content must be text');
    }
    return sha256Hex(encoder.encode(textValue));
  }

  async function digest(domain, value) {
    if (DOMAINS.indexOf(domain) < 0) {
      fail('unknown hash domain');
    }
    var prefix = encoder.encode('nachos.reconciliation/' + domain + '/v1\u0000');
    var body = canonicalBytes(value);
    var joined = new Uint8Array(prefix.length + body.length);
    joined.set(prefix, 0);
    joined.set(body, prefix.length);
    return sha256Hex(joined);
  }

  function checkMemoryRecord(record) {
    if (!isDict(record)) {
      fail('memory record requires key/title/summary/category/body');
    }
    var keys = Object.keys(record);
    if (keys.length !== MEMORY_FIELDS.length) {
      fail('memory record requires key/title/summary/category/body');
    }
    for (var i = 0; i < MEMORY_FIELDS.length; i++) {
      if (!hasOwn.call(record, MEMORY_FIELDS[i])) {
        fail('memory record requires key/title/summary/category/body');
      }
    }
    for (var j = 0; j < keys.length; j++) {
      if (typeof record[keys[j]] !== 'string') {
        fail('memory fields must be strings');
      }
    }
  }

  async function memorySha256(record) {
    checkMemoryRecord(record);
    return digest('memory-record', record);
  }

  /* ----------------------------------------------------------- strict parse */

  var CH_QUOTE = 0x22;
  var CH_BACKSLASH = 0x5c;

  function isHexDigit(c) {
    return (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x46) || (c >= 0x61 && c <= 0x66);
  }

  function parseStrict(bytes) {
    if (bytes instanceof ArrayBuffer) {
      bytes = new Uint8Array(bytes);
    }
    if (!(bytes instanceof Uint8Array)) {
      fail('expected JSON text or bytes');
    }
    if (bytes.length > LIMITS.MAX_BYTES) {
      fail('JSON byte limit exceeded');
    }
    var src;
    try {
      src = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
    } catch (e) {
      fail('invalid UTF-8');
    }
    if (src.length > 0 && src.charCodeAt(0) === 0xfeff) {
      fail('unexpected UTF-8 BOM');
    }
    var pos = 0;
    var nodes = 0;
    var len = src.length;

    function skip() {
      while (pos < len) {
        var c = src.charCodeAt(pos);
        if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) {
          pos++;
        } else {
          break;
        }
      }
    }

    function count(depth) {
      nodes++;
      if (nodes > LIMITS.MAX_NODES || depth > LIMITS.MAX_DEPTH) {
        fail('JSON depth/node limit exceeded');
      }
    }

    function parseString() {
      pos++; // opening quote
      var parts = null;
      var start = pos;
      var escaped = false;
      for (;;) {
        if (pos >= len) {
          fail('invalid JSON');
        }
        var c = src.charCodeAt(pos);
        if (c === CH_QUOTE) {
          break;
        }
        if (c < 0x20) {
          fail('invalid JSON');
        }
        if (c !== CH_BACKSLASH) {
          pos++;
          continue;
        }
        escaped = true;
        if (parts === null) {
          parts = [];
        }
        parts.push(src.slice(start, pos));
        pos++;
        if (pos >= len) {
          fail('invalid JSON');
        }
        var e = src.charCodeAt(pos);
        var rep = null;
        if (e === 0x22) { rep = '"'; }
        else if (e === 0x5c) { rep = '\\'; }
        else if (e === 0x2f) { rep = '/'; }
        else if (e === 0x62) { rep = '\b'; }
        else if (e === 0x66) { rep = '\f'; }
        else if (e === 0x6e) { rep = '\n'; }
        else if (e === 0x72) { rep = '\r'; }
        else if (e === 0x74) { rep = '\t'; }
        if (rep !== null) {
          parts.push(rep);
          pos++;
        } else if (e === 0x75) {
          if (pos + 4 >= len) {
            fail('invalid JSON');
          }
          for (var k = 1; k <= 4; k++) {
            if (!isHexDigit(src.charCodeAt(pos + k))) {
              fail('invalid JSON');
            }
          }
          parts.push(String.fromCharCode(parseInt(src.slice(pos + 1, pos + 5), 16)));
          pos += 5;
        } else {
          fail('invalid JSON');
        }
        start = pos;
      }
      var result;
      if (parts === null) {
        result = src.slice(start, pos);
      } else {
        parts.push(src.slice(start, pos));
        result = parts.join('');
      }
      pos++; // closing quote
      if (escaped && !wellFormed(result)) {
        fail('lone surrogate in JSON string');
      }
      return result;
    }

    function parseNumber() {
      var start = pos;
      if (src.charCodeAt(pos) === 0x2d) {
        pos++;
      }
      var c = pos < len ? src.charCodeAt(pos) : -1;
      if (c === 0x30) {
        pos++;
      } else if (c >= 0x31 && c <= 0x39) {
        while (pos < len) {
          var d = src.charCodeAt(pos);
          if (d >= 0x30 && d <= 0x39) {
            pos++;
          } else {
            break;
          }
        }
      } else {
        fail('unsupported JSON number');
      }
      var next = pos < len ? src.charCodeAt(pos) : -1;
      if (next === 0x2e || next === 0x65 || next === 0x45) {
        fail('unsupported JSON number');
      }
      var lexeme = src.slice(start, pos);
      var digits = lexeme.charCodeAt(0) === 0x2d ? lexeme.length - 1 : lexeme.length;
      if (digits > 16) {
        fail('integer out of range');
      }
      var value = Number(lexeme);
      if (!Number.isSafeInteger(value)) {
        fail('integer out of range');
      }
      return value === 0 ? 0 : value;
    }

    function literal(word, value) {
      if (src.startsWith(word, pos)) {
        pos += word.length;
        return value;
      }
      return fail('invalid JSON');
    }

    function parseValue(depth) {
      skip();
      if (pos >= len) {
        fail('invalid JSON');
      }
      count(depth);
      var c = src.charCodeAt(pos);
      if (c === 0x7b) {
        return parseObject(depth);
      }
      if (c === 0x5b) {
        return parseArray(depth);
      }
      if (c === CH_QUOTE) {
        return parseString();
      }
      if (c === 0x74) { return literal('true', true); }
      if (c === 0x66) { return literal('false', false); }
      if (c === 0x6e) { return literal('null', null); }
      if (c === 0x2d || (c >= 0x30 && c <= 0x39)) {
        return parseNumber();
      }
      return fail('invalid JSON');
    }

    function parseArray(depth) {
      pos++;
      var arr = [];
      skip();
      if (src.charCodeAt(pos) === 0x5d) {
        pos++;
        return arr;
      }
      for (;;) {
        if (arr.length >= LIMITS.MAX_ITEMS) {
          fail('collection limit exceeded');
        }
        arr.push(parseValue(depth + 1));
        skip();
        var c = pos < len ? src.charCodeAt(pos) : -1;
        pos++;
        if (c === 0x5d) {
          return arr;
        }
        if (c !== 0x2c) {
          fail('invalid JSON');
        }
      }
    }

    function parseObject(depth) {
      pos++;
      var obj = newObject();
      var size = 0;
      skip();
      if (src.charCodeAt(pos) === 0x7d) {
        pos++;
        return obj;
      }
      for (;;) {
        skip();
        if (pos >= len || src.charCodeAt(pos) !== CH_QUOTE) {
          fail('invalid JSON');
        }
        var key = parseString();
        if (hasOwn.call(obj, key)) {
          fail('duplicate JSON object key');
        }
        size++;
        if (size > LIMITS.MAX_ITEMS) {
          fail('collection limit exceeded');
        }
        skip();
        if (src.charCodeAt(pos) !== 0x3a) {
          fail('invalid JSON');
        }
        pos++;
        var value = parseValue(depth + 1);
        Object.defineProperty(obj, key, {
          value: value,
          enumerable: true,
          writable: true,
          configurable: true
        });
        skip();
        var c = pos < len ? src.charCodeAt(pos) : -1;
        pos++;
        if (c === 0x7d) {
          return obj;
        }
        if (c !== 0x2c) {
          fail('invalid JSON');
        }
      }
    }

    var result = parseValue(0);
    skip();
    if (pos !== len) {
      fail('invalid JSON');
    }
    return result;
  }

  /* Detached deep copy; null-prototype objects; applies canonical JSON limits. */
  function cloneJson(value) {
    canonWalkCheckOnly(value);
    return copyValue(value);
  }

  function canonWalkCheckOnly(value) {
    canonicalString(value);
  }

  function copyValue(v) {
    if (v === null || typeof v !== 'object') {
      return v;
    }
    var i;
    if (Array.isArray(v)) {
      var arr = new Array(v.length);
      for (i = 0; i < v.length; i++) {
        arr[i] = copyValue(v[i]);
      }
      return arr;
    }
    var keys = Object.keys(v);
    var out = newObject();
    for (i = 0; i < keys.length; i++) {
      Object.defineProperty(out, keys[i], {
        value: copyValue(v[keys[i]]),
        enumerable: true,
        writable: true,
        configurable: true
      });
    }
    return out;
  }

  /* ------------------------------------------------------ model validators */

  function fields(value, names) {
    var list = names.split(' ');
    var message = 'expected object fields: ' + names;
    if (!isDict(value)) {
      fail(message);
    }
    var keys = Object.keys(value);
    if (keys.length !== list.length) {
      fail(message);
    }
    for (var i = 0; i < list.length; i++) {
      if (!hasOwn.call(value, list[i])) {
        fail(message);
      }
    }
  }

  function text(value, empty) {
    if (typeof value !== 'string' || (!empty && value === '')) {
      fail('expected string');
    }
  }

  function sequence(value, nonempty) {
    if (!isList(value) || (nonempty && value.length === 0)) {
      fail('expected list');
    }
  }

  function strings(value, nonempty) {
    sequence(value, nonempty);
    var seen = new Set();
    for (var i = 0; i < value.length; i++) {
      text(value[i]);
      seen.add(value[i]);
    }
    if (seen.size !== value.length) {
      fail('duplicate string/ID');
    }
  }

  var DIGEST_RE = /^[0-9a-f]{64}$/;

  function digestText(value) {
    if (typeof value !== 'string' || !DIGEST_RE.test(value)) {
      fail('expected full lowercase SHA256');
    }
  }

  function indexed(values) {
    sequence(values);
    var result = new Map();
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      if (!isDict(v) || !hasOwn.call(v, 'id')) {
        fail('expected identified object');
      }
      text(v.id);
      if (result.has(v.id)) {
        fail('duplicate ID');
      }
      result.set(v.id, v);
    }
    return result;
  }

  function profileCheck(value) {
    fields(value, 'id root');
    text(value.id);
    text(value.root);
    if (value.root.indexOf('\u0000') >= 0) {
      fail('profile root contains NUL');
    }
  }

  function coverageCheck(value) {
    fields(value, 'discovered hashed content_searched section_reviewed body_reviewed limitations');
    var flags = ['discovered', 'hashed', 'content_searched', 'section_reviewed', 'body_reviewed'];
    for (var i = 0; i < flags.length; i++) {
      if (typeof value[flags[i]] !== 'boolean') {
        fail('coverage flags must be booleans');
      }
    }
    strings(value.limitations);
  }

  function locatorCheck(locator) {
    if (locator.indexOf('\u0000') >= 0) {
      fail('source locator contains NUL');
    }
    if (locator.indexOf('\\') >= 0 || locator.indexOf(':') >= 0) {
      fail('locator must be a relative POSIX label without traversal');
    }
    var parts = locator.split('/');
    for (var i = 0; i < parts.length; i++) {
      if (parts[i] === '' || parts[i] === '.' || parts[i] === '..') {
        fail('locator must be a relative POSIX label without traversal');
      }
    }
  }

  function metadataCheck(value) {
    var metadata = value.metadata;
    fields(metadata, 'byte_size char_size enabled provenance usage');
    var snapshot = value.kind === 'memory_record' ? canonicalString(value.before) : value.before;
    var sizes = [
      ['byte_size', encoder.encode(snapshot).length],
      ['char_size', codePointLength(snapshot)]
    ];
    for (var i = 0; i < sizes.length; i++) {
      if (!isInt(metadata[sizes[i][0]]) || metadata[sizes[i][0]] !== sizes[i][1]) {
        fail('snapshot size mismatch');
      }
    }
    if (metadata.enabled !== null && typeof metadata.enabled !== 'boolean') {
      fail('enabled must be boolean or null (unknown)');
    }
    strings(metadata.provenance);
    fields(metadata.usage, 'status references');
    var status = metadata.usage.status;
    if (status !== 'available' && status !== 'unavailable' && status !== 'unknown') {
      fail('unsupported usage availability');
    }
    strings(metadata.usage.references);
    if (status !== 'available' && metadata.usage.references.length > 0) {
      fail('unavailable usage cannot claim references');
    }
  }

  var SOURCE_KINDS = ['memory_record', 'memory_file', 'skill_file', 'reference_file'];

  async function sourceCheck(value) {
    fields(value, 'id kind role identity locator sha256 before coverage metadata');
    text(value.id);
    text(value.identity);
    text(value.locator);
    locatorCheck(value.locator);
    if (SOURCE_KINDS.indexOf(value.kind) < 0) {
      fail('unsupported source kind');
    }
    if (value.role !== 'primary' && value.role !== 'reference') {
      fail('unsupported source role');
    }
    digestText(value.sha256);
    var actual;
    if (value.kind === 'memory_record') {
      actual = await memorySha256(value.before);
    } else {
      text(value.before, true);
      actual = await fileSha256(value.before);
    }
    if (actual !== value.sha256) {
      fail('source snapshot hash mismatch');
    }
    coverageCheck(value.coverage);
    if (!value.coverage.hashed) {
      fail('snapshot must be hashed');
    }
    metadataCheck(value);
  }

  async function inventoryCheck(value) {
    fields(value, 'schema profile digest sources coverage');
    indexed(value.sources);
    var identities = new Set();
    for (var i = 0; i < value.sources.length; i++) {
      var item = value.sources[i];
      await sourceCheck(item);
      var identity = canonicalString([item.kind, item.identity, item.locator]);
      if (identities.has(identity)) {
        fail('duplicate source identity');
      }
      identities.add(identity);
    }
    coverageCheck(value.coverage);
  }

  var FINDING_TYPES = ['stale', 'duplicate', 'conflict', 'unsupported', 'clarification'];
  var EVIDENCE_KINDS = ['schedule', 'execution', 'delivery', 'manual', 'source_content', 'usage'];
  var EVIDENCE_STATUSES = ['verified', 'unverified', 'missing', 'conflicting'];

  function findingCheck(value, sources) {
    fields(value, 'id type source_ids excerpts rationale evidence limitations');
    text(value.id);
    if (FINDING_TYPES.indexOf(value.type) < 0) {
      fail('unsupported finding type');
    }
    strings(value.source_ids, true);
    for (var i = 0; i < value.source_ids.length; i++) {
      if (!sources.has(value.source_ids[i])) {
        fail('unknown finding source');
      }
    }
    text(value.rationale);
    strings(value.limitations);
    sequence(value.excerpts, true);
    for (var j = 0; j < value.excerpts.length; j++) {
      var excerpt = value.excerpts[j];
      fields(excerpt, 'source_id text');
      text(excerpt.text);
      if (value.source_ids.indexOf(excerpt.source_id) < 0) {
        fail('excerpt source mismatch');
      }
      var before = sources.get(excerpt.source_id).before;
      var found = false;
      if (typeof before === 'string') {
        found = before.indexOf(excerpt.text) >= 0;
      } else {
        var bkeys = Object.keys(before);
        for (var k = 0; k < bkeys.length && !found; k++) {
          found = before[bkeys[k]].indexOf(excerpt.text) >= 0;
        }
      }
      if (!found) {
        fail('excerpt is not exact source text');
      }
    }
    observations(value.evidence, value.source_ids);
  }

  /* ---- Python 3.13 datetime.fromisoformat parity for timezone-aware stamps */

  function isDigits(s) {
    return /^[0-9]+$/.test(s);
  }

  function isLeap(y) {
    return y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
  }

  var DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  var DAYS_BEFORE_MONTH = [0];
  (function () {
    var total = 0;
    for (var m = 1; m <= 12; m++) {
      DAYS_BEFORE_MONTH.push(total);
      total += DAYS_IN_MONTH[m];
    }
  })();

  function daysInMonth(y, m) {
    return m === 2 && isLeap(y) ? 29 : DAYS_IN_MONTH[m];
  }

  function daysBeforeYear(y) {
    var p = y - 1;
    return p * 365 + Math.floor(p / 4) - Math.floor(p / 100) + Math.floor(p / 400);
  }

  function ymd2ord(y, m, d) {
    return daysBeforeYear(y) + DAYS_BEFORE_MONTH[m] + (m > 2 && isLeap(y) ? 1 : 0) + d;
  }

  function isoWeekOk(year, week, day) {
    if (year < 1 || year > 9999) {
      return false;
    }
    if (!(week > 0 && week < 53)) {
      var ok = false;
      if (week === 53) {
        var firstWeekday = ymd2ord(year, 1, 1) % 7;
        ok = firstWeekday === 4 || (firstWeekday === 3 && isLeap(year));
      }
      if (!ok) {
        return false;
      }
    }
    if (!(day > 0 && day < 8)) {
      return false;
    }
    var first = ymd2ord(year, 1, 1);
    var firstWeekday2 = (first + 6) % 7;
    var week1Monday = first - firstWeekday2;
    if (firstWeekday2 > 3) {
      week1Monday += 7;
    }
    var ord = week1Monday + (week - 1) * 7 + (day - 1);
    return ord >= 1 && ord <= ymd2ord(9999, 12, 31);
  }

  function isoSeparator(s) {
    var n = s.length;
    if (n === 7) {
      return 7;
    }
    if (s[4] === '-') {
      if (s[5] === 'W') {
        if (n > 8 && s[8] === '-') {
          if (n === 9) {
            return -1;
          }
          if (n > 10 && isDigits(s[10])) {
            return 8;
          }
          return 10;
        }
        return 8;
      }
      return 10;
    }
    if (s[4] === 'W') {
      var idx = 7;
      while (idx < n && isDigits(s[idx])) {
        idx++;
      }
      if (idx < 9) {
        return idx;
      }
      return idx % 2 === 0 ? 7 : 8;
    }
    return 8;
  }

  function digitsAt(s, pos, count) {
    var part = s.slice(pos, pos + count);
    if (part.length !== count || !isDigits(part)) {
      return -1;
    }
    return parseInt(part, 10);
  }

  function dateOk(d) {
    var len = d.length;
    if (len !== 7 && len !== 8 && len !== 10) {
      return false;
    }
    var year = digitsAt(d, 0, 4);
    if (year < 0) {
      return false;
    }
    var hasSep = d[4] === '-';
    var pos = 4 + (hasSep ? 1 : 0);
    if (d.slice(pos, pos + 1) === 'W') {
      pos += 1;
      var week = digitsAt(d, pos, 2);
      if (week < 0) {
        return false;
      }
      pos += 2;
      var day = 1;
      if (len > pos) {
        if ((d.slice(pos, pos + 1) === '-') !== hasSep) {
          return false;
        }
        pos += hasSep ? 1 : 0;
        day = digitsAt(d, pos, 1);
        if (day < 0) {
          return false;
        }
      }
      return isoWeekOk(year, week, day);
    }
    var month = digitsAt(d, pos, 2);
    if (month < 0) {
      return false;
    }
    pos += 2;
    if ((d.slice(pos, pos + 1) === '-') !== hasSep) {
      return false;
    }
    pos += hasSep ? 1 : 0;
    var dom = digitsAt(d, pos, 2);
    if (dom < 0) {
      return false;
    }
    return year >= 1 && month >= 1 && month <= 12 && dom >= 1 && dom <= daysInMonth(year, month);
  }

  var FRACTION_CORRECTION = [100000, 10000, 1000, 100, 10];

  function hhmmssff(t) {
    var n = t.length;
    var comps = [0, 0, 0, 0];
    var pos = 0;
    var hasSep = false;
    for (var comp = 0; comp < 3; comp++) {
      if (n - pos < 2) {
        return null;
      }
      var v = digitsAt(t, pos, 2);
      if (v < 0) {
        return null;
      }
      comps[comp] = v;
      pos += 2;
      var next = t.slice(pos, pos + 1);
      if (comp === 0) {
        hasSep = next === ':';
      }
      if (next === '' || comp >= 2) {
        break;
      }
      if (hasSep && next !== ':') {
        return null;
      }
      pos += hasSep ? 1 : 0;
    }
    if (pos < n) {
      if (t[pos] !== '.' && t[pos] !== ',') {
        return null;
      }
      pos += 1;
      var frac = t.slice(pos);
      if (frac !== '' && !isDigits(frac)) {
        return null;
      }
      var remainder = n - pos;
      var take = remainder >= 6 ? 6 : remainder;
      if (take === 0) {
        return null;
      }
      comps[3] = parseInt(frac.slice(0, take), 10);
      if (take < 6) {
        comps[3] *= FRACTION_CORRECTION[take - 1];
      }
    }
    return comps;
  }

  function timeAwareOk(t) {
    if (t.length < 2) {
      return false;
    }
    var tzPos = (t.indexOf('-') + 1) || (t.indexOf('+') + 1) || (t.indexOf('Z') + 1);
    var timestr = tzPos > 0 ? t.slice(0, tzPos - 1) : t;
    var comps = hhmmssff(timestr);
    if (comps === null) {
      return false;
    }
    if (comps[0] > 23 || comps[1] > 59 || comps[2] > 59) {
      return false;
    }
    if (tzPos === t.length && t[t.length - 1] === 'Z') {
      return true;
    }
    if (tzPos <= 0) {
      return 'naive';
    }
    var tzstr = t.slice(tzPos);
    if (tzstr.length === 0 || tzstr.length === 1 || tzstr.length === 3) {
      return false;
    }
    var tz = hhmmssff(tzstr);
    if (tz === null) {
      return false;
    }
    var seconds = tz[0] * 3600 + tz[1] * 60 + tz[2] + tz[3] / 1e6;
    return seconds < 86400;
  }

  function timestamp(value) {
    text(value);
    var s = value.split('Z').join('+00:00');
    // Python's C implementation is byte oriented and its exotic leniencies differ between
    // builds; JS accepts the documented ISO subset only (fail closed: never looser than Python).
    if (!/^[\x00-\x7f]*$/.test(s) || s.length < 7) {
      fail('invalid timestamp');
    }
    var sep = isoSeparator(s);
    if (sep < 0 || sep >= s.length) {
      fail('invalid timestamp');
    }
    if (!dateOk(s.slice(0, sep))) {
      fail('invalid timestamp');
    }
    var tstr = s.slice(sep + 1);
    if (tstr === '') {
      fail('timestamp must include timezone');
    }
    var verdict = timeAwareOk(tstr);
    if (verdict === 'naive') {
      fail('timestamp must include timezone');
    }
    if (verdict !== true) {
      fail('invalid timestamp');
    }
  }

  function observations(value, sourceIds) {
    sequence(value, true);
    for (var i = 0; i < value.length; i++) {
      var evidence = value[i];
      fields(evidence, 'kind status source_ids claim observation observed_at');
      if (EVIDENCE_KINDS.indexOf(evidence.kind) < 0) {
        fail('unsupported evidence kind');
      }
      if (EVIDENCE_STATUSES.indexOf(evidence.status) < 0) {
        fail('unsupported evidence status');
      }
      strings(evidence.source_ids, true);
      for (var j = 0; j < evidence.source_ids.length; j++) {
        if (sourceIds.indexOf(evidence.source_ids[j]) < 0) {
          fail('unknown evidence source');
        }
      }
      text(evidence.claim);
      text(evidence.observation);
      timestamp(evidence.observed_at);
    }
  }

  function preservation(value) {
    sequence(value, true);
    for (var i = 0; i < value.length; i++) {
      var check = value[i];
      fields(check, 'item preserved explanation');
      text(check.item);
      text(check.explanation);
      if (typeof check.preserved !== 'boolean') {
        fail('preserved must be boolean');
      }
    }
  }

  function assessmentsCheck(value, findings) {
    sequence(value.assessments);
    var assessed = new Set();
    for (var i = 0; i < value.assessments.length; i++) {
      var a = value.assessments[i];
      fields(a, 'finding_id revision evidence preservation_checklist');
      var fid = a.finding_id;
      text(fid);
      if (value.finding_ids.indexOf(fid) < 0 || assessed.has(fid)) {
        fail('unknown or duplicate assessment finding');
      }
      assessed.add(fid);
      if (!isInt(a.revision) || a.revision !== value.revision) {
        fail('preservation assessment must bind exact revision');
      }
      observations(a.evidence, findings.get(fid).source_ids);
      preservation(a.preservation_checklist);
    }
  }

  var PROPOSAL_KINDS = ['merge', 'supersede', 'unlink', 'rewrite', 'delete', 'choose-one', 'other'];
  var CONFIDENCE = ['low', 'medium', 'high'];

  function linkChange(change, src, memoryKeys) {
    if (src.kind !== 'memory_record') {
      fail('supersede/unlink apply to memory records only');
    }
    fields(change.after, 'superseded_by');
    var target = change.after.superseded_by;
    text(target);
    if (target === src.before.key || !memoryKeys.has(target)) {
      fail('link target must be another inventory memory record');
    }
  }

  function changeCheck(change, src, memoryKeys) {
    fields(change, 'source_id source_sha256 operation before after');
    if (change.source_sha256 !== src.sha256 || !deepEqual(change.before, src.before)) {
      fail('source precondition mismatch');
    }
    if (change.operation === 'disable') {
      if (change.after !== null) {
        fail('disable requires null after');
      }
    } else if (change.operation === 'supersede' || change.operation === 'unlink') {
      linkChange(change, src, memoryKeys);
    } else if (change.operation === 'edit') {
      if (src.kind === 'memory_record') {
        checkMemoryRecord(change.after);
        if (change.after.key !== change.before.key) {
          fail('memory key identity cannot change');
        }
      } else {
        text(change.after, true);
      }
    } else {
      fail('unsupported operation');
    }
  }

  function optionalProposalFields(value) {
    if (!isDict(value)) {
      fail('proposal must be an object');
    }
    var base = {};
    Object.keys(value).forEach(function (k) {
      if (k !== 'kind' && k !== 'confidence') {
        base[k] = value[k];
      }
    });
    fields(base, 'id revision finding_ids depends_on changes assessments');
    if (hasOwn.call(value, 'kind') && PROPOSAL_KINDS.indexOf(value.kind) < 0) {
      fail('unsupported proposal kind');
    }
    if (hasOwn.call(value, 'confidence') && CONFIDENCE.indexOf(value.confidence) < 0) {
      fail('unsupported confidence');
    }
  }

  function proposalCheck(value, packet) {
    optionalProposalFields(value);
    text(value.id);
    if (!isInt(value.revision) || value.revision < 1) {
      fail('revision must be positive integer');
    }
    strings(value.finding_ids, true);
    strings(value.depends_on);
    var findings = indexed(packet.findings);
    for (var i = 0; i < value.finding_ids.length; i++) {
      if (!findings.has(value.finding_ids[i])) {
        fail('unknown proposal finding');
      }
    }
    assessmentsCheck(value, findings);
    var sources = indexed(packet.inventory.sources);
    var covered = new Set();
    var excerpted = new Set();
    for (var a = 0; a < value.finding_ids.length; a++) {
      var f = findings.get(value.finding_ids[a]);
      for (var b = 0; b < f.source_ids.length; b++) {
        covered.add(f.source_ids[b]);
      }
      for (var c = 0; c < f.excerpts.length; c++) {
        excerpted.add(f.excerpts[c].source_id);
      }
    }
    var memoryKeys = new Set();
    sources.forEach(function (s) {
      if (s.kind === 'memory_record') {
        memoryKeys.add(s.before.key);
      }
    });
    sequence(value.changes, true);
    var changed = new Set();
    for (var k = 0; k < value.changes.length; k++) {
      var change = value.changes[k];
      if (!isDict(change)) {
        fail('change must be an object');
      }
      var sid = get(change, 'source_id');
      text(sid);
      if (!covered.has(sid) || changed.has(sid)) {
        fail('unknown, uncovered or duplicate change source');
      }
      changed.add(sid);
      if (!excerpted.has(sid)) {
        fail('each changed source requires an exact supported excerpt');
      }
      changeCheck(change, sources.get(sid), memoryKeys);
    }
  }

  function approvalSafe(item) {
    var assessed = new Set();
    var all = true;
    for (var i = 0; i < item.assessments.length; i++) {
      var a = item.assessments[i];
      assessed.add(a.finding_id);
      for (var j = 0; j < a.preservation_checklist.length; j++) {
        if (!a.preservation_checklist[j].preserved) {
          all = false;
        }
      }
    }
    var wanted = new Set(item.finding_ids);
    if (assessed.size !== wanted.size) {
      return false;
    }
    var same = true;
    assessed.forEach(function (fid) {
      if (!wanted.has(fid)) {
        same = false;
      }
    });
    return same && all;
  }

  async function packetCheck(value) {
    fields(value, 'schema profile digest inventory findings proposals');
    await validateInner(value.inventory, value.profile);
    if (value.inventory.schema !== SCHEMAS.inventory) {
      fail('packet requires inventory schema');
    }
    var sources = indexed(value.inventory.sources);
    var findings = indexed(value.findings);
    findings.forEach(function (item) {
      findingCheck(item, sources);
    });
    indexed(value.proposals).forEach(function (item) {
      proposalCheck(item, value);
      if (item.revision !== 1) {
        fail('packet proposals must start at revision 1');
      }
    });
  }

  async function validateInner(value, expectedProfile) {
    canonicalBytes(value);
    if (!isDict(value) || !KIND_BY_SCHEMA.has(get(value, 'schema'))) {
      fail('unsupported schema/version');
    }
    if (!hasOwn.call(value, 'profile') || !hasOwn.call(value, 'digest')) {
      fail('missing profile/digest');
    }
    profileCheck(value.profile);
    if (expectedProfile !== undefined && expectedProfile !== null &&
        !deepEqual(value.profile, expectedProfile)) {
      fail('profile binding mismatch');
    }
    var kind = KIND_BY_SCHEMA.get(value.schema);
    digestText(value.digest);
    var payload = newObject();
    var keys = Object.keys(value);
    for (var i = 0; i < keys.length; i++) {
      if (keys[i] !== 'digest') {
        payload[keys[i]] = value[keys[i]];
      }
    }
    if (value.digest !== await digest(kind, payload)) {
      fail('artifact digest mismatch');
    }
    if (kind === 'inventory') {
      await inventoryCheck(value);
    } else if (kind === 'packet') {
      await packetCheck(value);
    } else if (kind === 'session') {
      await sessionCheck(value);
    } else {
      await planCheck(value);
    }
  }

  async function validateDocument(value, expectedProfile) {
    try {
      await validateInner(value, expectedProfile);
      return cloneJson(value);
    } catch (e) {
      if (e && e.nachos === true) {
        throw e;
      }
      return fail('malformed reconciliation document');
    }
  }

  async function sealDocument(value) {
    try {
      canonicalBytes(value);
      var result = cloneJson(value);
      if (!isDict(result) || !KIND_BY_SCHEMA.has(get(result, 'schema'))) {
        fail('unsupported schema/version');
      }
      var kind = KIND_BY_SCHEMA.get(result.schema);
      delete result.digest;
      result.digest = await digest(kind, result);
      return await validateDocument(result);
    } catch (e) {
      if (e && e.nachos === true) {
        throw e;
      }
      return fail('malformed reconciliation document');
    }
  }

  function dumps(value) {
    return canonicalBytes(value);
  }

  /* ------------------------------------------------------- review lifecycle */

  function reviewContext(packet, item) {
    var findings = indexed(packet.findings);
    var selected = [];
    var ids = new Set();
    for (var i = 0; i < item.finding_ids.length; i++) {
      var f = findings.get(item.finding_ids[i]);
      if (f === undefined) {
        fail('unknown proposal finding');
      }
      selected.push(f);
      for (var j = 0; j < f.source_ids.length; j++) {
        ids.add(f.source_ids[j]);
      }
    }
    var sources = packet.inventory.sources.filter(function (src) {
      return ids.has(src.id);
    });
    return {
      profile: packet.profile,
      packet_digest: packet.digest,
      proposal: item,
      findings: selected,
      sources: sources,
      inventory_coverage: packet.inventory.coverage
    };
  }

  async function proposalDigest(packet, item) {
    var checked = await validateDocument(packet);
    proposalCheck(item, checked);
    return digest('proposal', reviewContext(checked, item));
  }

  function eventRevision(event, revisions, current) {
    fields(event, 'revision action digest note');
    text(event.note, true);
    var action = event.action;
    if (ACTIONS.indexOf(action) < 0) {
      fail('unknown review action');
    }
    if (action === 'edit' || action === 'request_revision') {
      current += 1;
    }
    if (!isInt(event.revision) || event.revision !== current || current > revisions.length) {
      fail('history revision mismatch');
    }
    if (action === 'request_revision') {
      var expected = Object.assign(newObject(), revisions[current - 2]);
      expected.revision = current;
      expected.assessments = [];
      if (!deepEqual(revisions[current - 1], expected)) {
        fail('revision request cannot edit content');
      }
    }
    return current;
  }

  async function replay(record, packet, original) {
    fields(record, 'proposal_id revisions history current_revision decision accepted_digest');
    sequence(record.revisions, true);
    sequence(record.history);
    var revisions = record.revisions;
    if (!deepEqual(revisions[0], original)) {
      fail('initial revision differs from packet');
    }
    for (var n = 0; n < revisions.length; n++) {
      proposalCheck(revisions[n], packet);
      if (revisions[n].id !== original.id || revisions[n].revision !== n + 1) {
        fail('noncontiguous or mismatched revision identity');
      }
    }
    var current = 1;
    var decision = 'unreviewed';
    var accepted = null;
    for (var i = 0; i < record.history.length; i++) {
      var event = record.history[i];
      current = eventRevision(event, revisions, current);
      var action = event.action;
      var item = revisions[current - 1];
      if (action === 'accept' && !approvalSafe(item)) {
        fail('preservation assessment blocks acceptance');
      }
      var actual = await digest('proposal', reviewContext(packet, item));
      if (event.digest !== actual) {
        fail('review history digest mismatch');
      }
      decision = action === 'edit' ? 'unreviewed' : action;
      accepted = action === 'accept' ? actual : null;
    }
    if (!isInt(record.current_revision) || record.current_revision !== current ||
        revisions.length !== current || record.decision !== decision ||
        record.accepted_digest !== accepted) {
      fail('inconsistent approval/revision state');
    }
  }

  async function sessionCheck(value) {
    fields(value, 'schema profile digest packet reviews');
    var packet = value.packet;
    await validateInner(packet, value.profile);
    if (packet.schema !== SCHEMAS.packet) {
      fail('session requires packet schema');
    }
    var originals = indexed(packet.proposals);
    sequence(value.reviews);
    var seen = new Set();
    for (var i = 0; i < value.reviews.length; i++) {
      var record = value.reviews[i];
      fields(record, 'proposal_id revisions history current_revision decision accepted_digest');
      var pid = record.proposal_id;
      text(pid);
      if (!originals.has(pid) || seen.has(pid)) {
        fail('unknown or duplicate review ID');
      }
      seen.add(pid);
      await replay(record, packet, originals.get(pid));
    }
    if (seen.size !== originals.size) {
      fail('missing proposal review state');
    }
  }

  async function newSession(packet) {
    var checked = await validateDocument(packet);
    if (checked.schema !== SCHEMAS.packet) {
      fail('session requires a packet');
    }
    var reviews = checked.proposals.map(function (item) {
      var r = newObject();
      r.proposal_id = item.id;
      r.revisions = [copyValue(item)];
      r.history = [];
      r.current_revision = 1;
      r.decision = 'unreviewed';
      r.accepted_digest = null;
      return r;
    });
    var doc = newObject();
    doc.schema = SCHEMAS.session;
    doc.profile = checked.profile;
    doc.packet = checked;
    doc.reviews = reviews;
    return sealDocument(doc);
  }

  function appendRevision(result, record, action, edited) {
    var current = record.revisions[record.revisions.length - 1];
    var item;
    if (action === 'edit') {
      if (edited === undefined || edited === null) {
        fail('edit requires full edited_proposal');
      }
      item = cloneJson(edited);
    } else {
      item = copyValue(current);
      item.revision = current.revision + 1;
      item.assessments = [];
    }
    if (isDict(item) && isList(item.assessments)) {
      // A copied prior assessment is not fresh confirmation of edited text.
      // Python compares with != so a boolean true equals revision 1; mirrored.
      item.assessments = item.assessments.filter(function (a) {
        if (!isDict(a)) {
          return true;
        }
        var rev = get(a, 'revision');
        var same = rev === current.revision || (rev === true && current.revision === 1);
        return !same;
      });
    }
    proposalCheck(item, result.packet);
    if (item.id !== record.proposal_id || item.revision !== current.revision + 1) {
      fail('edit must preserve ID and increment revision exactly');
    }
    record.revisions.push(item);
    return item;
  }

  async function reviewProposal(session, proposalId, action, options) {
    var opts = options || {};
    var edited = opts.editedProposal;
    var note = opts.note === undefined ? '' : opts.note;
    try {
      var result = await validateDocument(session);
      if (result.schema !== SCHEMAS.session || ACTIONS.indexOf(action) < 0) {
        fail('invalid session/action');
      }
      text(note, true);
      var record = null;
      for (var i = 0; i < result.reviews.length; i++) {
        if (result.reviews[i].proposal_id === proposalId) {
          record = result.reviews[i];
          break;
        }
      }
      if (record === null) {
        fail('unknown proposal ID');
      }
      var current = record.revisions[record.revisions.length - 1];
      if (action === 'edit' || action === 'request_revision') {
        current = appendRevision(result, record, action, edited);
      }
      if (action !== 'edit' && edited !== undefined && edited !== null) {
        fail('edited_proposal is only permitted for edit');
      }
      var d = await digest('proposal', reviewContext(result.packet, current));
      if (action === 'accept' && !approvalSafe(current)) {
        fail('preservation assessment blocks acceptance');
      }
      record.current_revision = current.revision;
      record.decision = action === 'edit' ? 'unreviewed' : action;
      record.accepted_digest = action === 'accept' ? d : null;
      var event = newObject();
      event.revision = current.revision;
      event.action = action;
      event.digest = d;
      event.note = note;
      record.history.push(event);
      return await sealDocument(result);
    } catch (e) {
      if (e && e.nachos === true) {
        throw e;
      }
      return fail('malformed reconciliation document');
    }
  }

  /* ------------------------------------------------------------------ export */

  function eligible(items) {
    var blocked = new Map();
    var owners = new Map();
    var followers = new Map();
    var pending = new Map();
    items.forEach(function (item, pid) {
      var i;
      for (i = 0; i < item.changes.length; i++) {
        var change = item.changes[i];
        if (!owners.has(change.source_id)) {
          owners.set(change.source_id, []);
        }
        owners.get(change.source_id).push([pid, canonicalString(change)]);
      }
      pending.set(pid, item.depends_on.length);
      for (i = 0; i < item.depends_on.length; i++) {
        var dep = item.depends_on[i];
        if (!followers.has(dep)) {
          followers.set(dep, []);
        }
        followers.get(dep).push(pid);
        if (!items.has(dep)) {
          blocked.set(pid, 'missing or unaccepted dependency');
        }
      }
    });
    owners.forEach(function (group) {
      var raws = new Set();
      group.forEach(function (entry) {
        raws.add(entry[1]);
      });
      if (raws.size > 1) {
        group.forEach(function (entry) {
          blocked.set(entry[0], 'conflicting changes to the same source');
        });
      }
    });
    var start = [];
    pending.forEach(function (count, pid) {
      if (count === 0 && !blocked.has(pid)) {
        start.push(pid);
      }
    });
    var queue = sortedStrings(start);
    var order = [];
    var head = 0;
    while (head < queue.length) {
      var next = queue[head++];
      order.push(next);
      var list = followers.get(next) || [];
      for (var f = 0; f < list.length; f++) {
        var follower = list[f];
        pending.set(follower, pending.get(follower) - 1);
        if (pending.get(follower) === 0 && !blocked.has(follower)) {
          queue.push(follower);
        }
      }
    }
    var selected = new Set(order);
    items.forEach(function (item, pid) {
      if (!selected.has(pid) && !blocked.has(pid)) {
        blocked.set(pid, 'cyclic or blocked dependency');
      }
    });
    return { order: order, blocked: blocked };
  }

  async function exportPlan(session) {
    try {
      var checked = await validateDocument(session);
      if (checked.schema !== SCHEMAS.session) {
        fail('export requires a review session');
      }
      var records = new Map();
      var candidates = new Map();
      var i;
      for (i = 0; i < checked.reviews.length; i++) {
        var record = checked.reviews[i];
        records.set(record.proposal_id, record);
        if (record.decision === 'accept') {
          candidates.set(record.proposal_id, record.revisions[record.revisions.length - 1]);
        }
      }
      var plan = eligible(candidates);
      var approved = [];
      for (i = 0; i < plan.order.length; i++) {
        var pid = plan.order[i];
        var context = reviewContext(checked.packet, candidates.get(pid));
        var entry = newObject();
        entry.proposal = context.proposal;
        entry.findings = context.findings;
        entry.sources = context.sources;
        entry.inventory_coverage = context.inventory_coverage;
        entry.review_digest = records.get(pid).accepted_digest;
        approved.push(entry);
      }
      var inOrder = new Set(plan.order);
      var ledger = [];
      var ids = sortedStrings(Array.from(records.keys()));
      for (i = 0; i < ids.length; i++) {
        var id = ids[i];
        if (inOrder.has(id)) {
          continue;
        }
        var rec = records.get(id);
        var status = rec.decision === 'reject' ? 'rejected' :
          (rec.decision === 'defer' ? 'deferred' : 'unreviewed');
        var row = newObject();
        row.proposal_id = id;
        row.revision = rec.current_revision;
        row.decision = rec.decision;
        row.status = plan.blocked.has(id) ? 'blocked' : status;
        row.reason = plan.blocked.has(id) ? plan.blocked.get(id) :
          'current revision is not explicitly accepted';
        ledger.push(row);
      }
      var doc = newObject();
      doc.schema = SCHEMAS.plan;
      doc.profile = checked.profile;
      doc.packet_digest = checked.packet.digest;
      doc.session_digest = checked.digest;
      doc.approved = approved;
      doc.ledger = ledger;
      doc.execution = 'not_performed';
      doc.source_validation = 'not_performed';
      return await sealDocument(doc);
    } catch (e) {
      if (e && e.nachos === true) {
        throw e;
      }
      return fail('malformed reconciliation document');
    }
  }

  async function approvedCheck(entry, plan) {
    fields(entry, 'proposal findings sources inventory_coverage review_digest');
    var sources = indexed(entry.sources);
    for (var i = 0; i < entry.sources.length; i++) {
      await sourceCheck(entry.sources[i]);
    }
    var findings = indexed(entry.findings);
    findings.forEach(function (item) {
      findingCheck(item, sources);
    });
    coverageCheck(entry.inventory_coverage);
    var item = entry.proposal;
    proposalCheck(item, { findings: entry.findings, inventory: { sources: entry.sources } });
    if (!approvalSafe(item)) {
      fail('preservation assessment blocks approval');
    }
    var ids = new Set(item.finding_ids);
    if (ids.size !== findings.size) {
      fail('extraneous plan findings');
    }
    ids.forEach(function (fid) {
      if (!findings.has(fid)) {
        fail('extraneous plan findings');
      }
    });
    var wanted = new Set();
    findings.forEach(function (f) {
      f.source_ids.forEach(function (sid) {
        wanted.add(sid);
      });
    });
    if (wanted.size !== sources.size) {
      fail('extraneous plan sources');
    }
    wanted.forEach(function (sid) {
      if (!sources.has(sid)) {
        fail('extraneous plan sources');
      }
    });
    var context = newObject();
    Object.keys(entry).forEach(function (key) {
      if (key !== 'review_digest') {
        context[key] = entry[key];
      }
    });
    context.profile = plan.profile;
    context.packet_digest = plan.packet_digest;
    if (entry.review_digest !== await digest('proposal', context)) {
      fail('approved review content digest mismatch');
    }
    return item;
  }

  var LEDGER_STATUS = new Map([
    ['reject', 'rejected'],
    ['defer', 'deferred'],
    ['accept', 'blocked'],
    ['unreviewed', 'unreviewed'],
    ['request_revision', 'unreviewed']
  ]);

  async function planCheck(value) {
    fields(value, 'schema profile digest packet_digest session_digest approved ledger execution ' +
      'source_validation');
    digestText(value.packet_digest);
    digestText(value.session_digest);
    if (value.execution !== 'not_performed' || value.source_validation !== 'not_performed') {
      fail('offline plan must not claim execution or source validation');
    }
    sequence(value.approved);
    var items = new Map();
    for (var i = 0; i < value.approved.length; i++) {
      var item = await approvedCheck(value.approved[i], value);
      if (items.has(item.id)) {
        fail('duplicate approved ID');
      }
      items.set(item.id, item);
    }
    var plan = eligible(items);
    var keys = Array.from(items.keys());
    if (keys.length !== plan.order.length) {
      fail('approved plan entries are not in dependency order');
    }
    for (var k = 0; k < keys.length; k++) {
      if (keys[k] !== plan.order[k]) {
        fail('approved plan entries are not in dependency order');
      }
    }
    if (plan.blocked.size > 0) {
      fail('plan includes conflicting or unsatisfied dependencies');
    }
    sequence(value.ledger);
    var seen = new Set(items.keys());
    for (var j = 0; j < value.ledger.length; j++) {
      var entry = value.ledger[j];
      fields(entry, 'proposal_id revision decision status reason');
      text(entry.proposal_id);
      text(entry.reason);
      if (seen.has(entry.proposal_id)) {
        fail('duplicate ledger/approved ID');
      }
      seen.add(entry.proposal_id);
      if (!isInt(entry.revision) || entry.revision < 1) {
        fail('invalid ledger revision');
      }
      if (typeof entry.decision !== 'string' || !LEDGER_STATUS.has(entry.decision) ||
          LEDGER_STATUS.get(entry.decision) !== entry.status) {
        fail('invalid decision ledger state');
      }
    }
  }

  async function validatePlan(plan) {
    var checked = await validateDocument(plan);
    if (checked.schema !== SCHEMAS.plan) {
      fail('validation requires an exported plan');
    }
    return checked;
  }

  /* ----------------------------------------------------- validation result */

  function validateValidationResult(result, plan) {
    try {
      fields(result, 'schema plan_digest profile checked_at status sources limitations');
      if (result.schema !== VALIDATION_SCHEMA) {
        fail('unsupported validation schema');
      }
      if (!isDict(plan) || plan.schema !== SCHEMAS.plan) {
        fail('validation requires an exported plan');
      }
      digestText(result.plan_digest);
      if (result.plan_digest !== plan.digest) {
        fail('validation plan digest mismatch');
      }
      profileCheck(result.profile);
      if (!deepEqual(result.profile, plan.profile)) {
        fail('validation profile mismatch');
      }
      timestamp(result.checked_at);
      if (result.status !== 'valid_at_check' && result.status !== 'stale' &&
          result.status !== 'unavailable') {
        fail('unsupported validation status');
      }
      strings(result.limitations);
      var reviewed = new Map();
      sequence(plan.approved);
      plan.approved.forEach(function (entry) {
        entry.sources.forEach(function (src) {
          var rec = reviewed.get(src.identity);
          if (rec === undefined) {
            rec = { sha256: src.sha256, kind: src.kind, locator: src.locator, proposals: new Set() };
            reviewed.set(src.identity, rec);
          }
          if (rec.sha256 !== src.sha256 || rec.kind !== src.kind || rec.locator !== src.locator) {
            fail('plan reviews one identity with inconsistent sources');
          }
          rec.proposals.add(entry.proposal.id);
        });
      });
      sequence(result.sources);
      var identities = sortedStrings(Array.from(reviewed.keys()));
      if (result.sources.length !== identities.length) {
        fail('validation rows do not match plan sources');
      }
      var statuses = new Set();
      var reasonRe = /^[a-z0-9_]{1,64}$/;
      for (var i = 0; i < identities.length; i++) {
        var row = result.sources[i];
        fields(row, 'identity proposal_ids expected_sha256 actual_sha256 status reason');
        var want = reviewed.get(identities[i]);
        if (row.identity !== identities[i]) {
          fail('validation rows do not match plan sources');
        }
        strings(row.proposal_ids, true);
        var pids = sortedStrings(Array.from(want.proposals));
        if (!deepEqual(row.proposal_ids, pids)) {
          fail('validation rows do not match plan sources');
        }
        digestText(row.expected_sha256);
        if (row.expected_sha256 !== want.sha256) {
          fail('validation rows do not match plan sources');
        }
        if (row.actual_sha256 !== null) {
          digestText(row.actual_sha256);
        }
        text(row.reason);
        if (!reasonRe.test(row.reason)) {
          fail('invalid validation reason');
        }
        var st = row.status;
        if (st === 'unchanged') {
          if (row.actual_sha256 !== row.expected_sha256 || row.reason !== 'sha256_match') {
            fail('inconsistent validation row');
          }
        } else if (st === 'changed') {
          if (row.actual_sha256 === null ||
              (row.reason !== 'kind_or_locator_changed' && row.reason !== 'sha256_changed') ||
              (row.reason === 'sha256_changed' && row.actual_sha256 === row.expected_sha256)) {
            fail('inconsistent validation row');
          }
        } else if (st === 'missing') {
          if (row.actual_sha256 !== null || row.reason !== 'source_not_found') {
            fail('inconsistent validation row');
          }
        } else if (st === 'unavailable') {
          if (row.actual_sha256 !== null) {
            fail('inconsistent validation row');
          }
        } else {
          fail('unsupported validation row status');
        }
        statuses.add(st);
      }
      var expectedStatus;
      if (statuses.has('unavailable')) {
        expectedStatus = 'unavailable';
      } else if (statuses.size === 0 && result.status === 'unavailable') {
        expectedStatus = 'unavailable'; // reader failed with nothing reviewed
      } else {
        var onlyUnchanged = true;
        statuses.forEach(function (s) {
          if (s !== 'unchanged') {
            onlyUnchanged = false;
          }
        });
        expectedStatus = onlyUnchanged ? 'valid_at_check' : 'stale';
      }
      if (result.status !== expectedStatus) {
        fail('inconsistent validation status');
      }
      return cloneJson(result);
    } catch (e) {
      if (e && e.nachos === true) {
        throw e;
      }
      return fail('malformed validation result');
    }
  }

  /* --------------------------------------------------------------- line diff */

  function lineDiff(a, b, options) {
    var opts = options || {};
    var maxLines = opts.maxLines === undefined ? LIMITS.DIFF_MAX_LINES : opts.maxLines;
    var maxOps = opts.maxOps === undefined ? LIMITS.DIFF_MAX_OPS : opts.maxOps;
    var fallback = { fallback: true, rows: [] };
    if (typeof a !== 'string' || typeof b !== 'string') {
      fail('diff requires text');
    }
    var x = a.split('\n');
    var y = b.split('\n');
    if (x.length > maxLines || y.length > maxLines) {
      return fallback;
    }
    var rows = [];
    var lo = 0;
    while (lo < x.length && lo < y.length && x[lo] === y[lo]) {
      lo++;
    }
    var xe = x.length;
    var ye = y.length;
    while (xe > lo && ye > lo && x[xe - 1] === y[ye - 1]) {
      xe--;
      ye--;
    }
    var i;
    for (i = 0; i < lo; i++) {
      rows.push({ type: 'eq', a: x[i], b: y[i] });
    }
    var mid = myers(x, y, lo, xe, ye, maxOps);
    if (mid === null) {
      return fallback;
    }
    for (i = 0; i < mid.length; i++) {
      rows.push(mid[i]);
    }
    for (i = 0; xe + i < x.length; i++) {
      rows.push({ type: 'eq', a: x[xe + i], b: y[ye + i] });
    }
    return { fallback: false, rows: rows };
  }

  /* Myers O(ND) on x[lo..xe) vs y[lo..ye); null when the op budget is exceeded. */
  function myers(x, y, lo, xe, ye, maxOps) {
    var n = xe - lo;
    var m = ye - lo;
    var out = [];
    var i;
    if (n === 0 || m === 0) {
      for (i = lo; i < xe; i++) {
        out.push({ type: 'del', a: x[i] });
      }
      for (i = lo; i < ye; i++) {
        out.push({ type: 'add', b: y[i] });
      }
      return out;
    }
    var max = n + m;
    var offset = max + 1;
    var v = new Int32Array(2 * max + 3);
    var trace = [];
    var ops = 0;
    var d;
    var k;
    var found = false;
    for (d = 0; d <= max && !found; d++) {
      trace.push(v.slice(offset - d - 1, offset + d + 2));
      for (k = -d; k <= d; k += 2) {
        ops++;
        var px;
        if (k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1])) {
          px = v[offset + k + 1];
        } else {
          px = v[offset + k - 1] + 1;
        }
        var py = px - k;
        while (px < n && py < m && x[lo + px] === y[lo + py]) {
          px++;
          py++;
          ops++;
        }
        v[offset + k] = px;
        if (px >= n && py >= m) {
          found = true;
          break;
        }
        if (ops > maxOps) {
          return null;
        }
      }
      if (ops > maxOps) {
        return null;
      }
    }
    // Backtrack. trace[d] holds v before round d, indexed from -d-1.
    var cx = n;
    var cy = m;
    var rev = [];
    for (d = trace.length - 1; d >= 0; d--) {
      var vv = trace[d];
      var base = d + 1;
      k = cx - cy;
      var prevK;
      if (k === -d || (k !== d && vv[base + k - 1] < vv[base + k + 1])) {
        prevK = k + 1;
      } else {
        prevK = k - 1;
      }
      var prevX = vv[base + prevK];
      var prevY = prevX - prevK;
      while (cx > prevX && cy > prevY && d > 0) {
        cx--;
        cy--;
        rev.push({ type: 'eq', a: x[lo + cx], b: y[lo + cy] });
      }
      if (d === 0) {
        while (cx > 0 && cy > 0) {
          cx--;
          cy--;
          rev.push({ type: 'eq', a: x[lo + cx], b: y[lo + cy] });
        }
        break;
      }
      if (cx === prevX) {
        cy--;
        rev.push({ type: 'add', b: y[lo + cy] });
      } else {
        cx--;
        rev.push({ type: 'del', a: x[lo + cx] });
      }
    }
    for (i = rev.length - 1; i >= 0; i--) {
      out.push(rev[i]);
    }
    return out;
  }

  /* ----------------------------------------------------------------- export */

  var api = {
    version: 'v1',
    LIMITS: LIMITS,
    SCHEMAS: SCHEMAS,
    parseStrict: parseStrict,
    canonicalBytes: canonicalBytes,
    digest: digest,
    sha256Hex: sha256Hex,
    fileSha256: fileSha256,
    memorySha256: memorySha256,
    validateDocument: validateDocument,
    dumps: dumps,
    sealDocument: sealDocument,
    newSession: newSession,
    reviewProposal: reviewProposal,
    exportPlan: exportPlan,
    validatePlan: validatePlan,
    approvalSafe: approvalSafe,
    proposalDigest: proposalDigest,
    reviewContext: reviewContext,
    validateValidationResult: validateValidationResult,
    lineDiff: lineDiff
  };
  Object.freeze(api);

  if (typeof module === 'object' && module !== null && typeof module.exports === 'object') {
    module.exports = api;
  }
  root.NachosCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : this));
