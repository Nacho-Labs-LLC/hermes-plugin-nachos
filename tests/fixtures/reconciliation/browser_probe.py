"""Dev-only real-Chromium toolchain probe, NOT a viewer or viewer E2E suite.

Generates a temporary inert file-import/download surface under TMPDIR, checks
browser SHA-256 parity on public Python fixtures, and never starts a server.
JSON.parse here deliberately is NOT the production strict import validator.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from nachos_core.reconciliation import loads, proposal_digest

HERE = Path(__file__).resolve().parent
SCRIPT = r"""
// Test-only canonical serializer: emit sorted pairs directly, avoiding JS's
// numeric-looking object-key enumeration and UTF-16 default sort differences.
const cmp = (a,b) => {
  const aa=Array.from(a, c=>c.codePointAt(0)), bb=Array.from(b, c=>c.codePointAt(0));
  for(let i=0;i<Math.min(aa.length,bb.length);i++) if(aa[i]!==bb[i]) return aa[i]-bb[i];
  return aa.length-bb.length;
};
function canonical(value) {
  let nodes=0;
  function string(s) {
    for(let i=0;i<s.length;i++) {
      const c=s.charCodeAt(i);
      if(c>=0xD800 && c<=0xDBFF) {
        const d=s.charCodeAt(++i);
        if(!(d>=0xDC00 && d<=0xDFFF)) throw Error('unpaired surrogate');
      } else if(c>=0xDC00 && c<=0xDFFF) throw Error('unpaired surrogate');
    }
    return JSON.stringify(s);
  }
  function walk(v,depth) {
    if(++nodes>100000 || depth>32) throw Error('resource limit');
    if(v===null) return 'null';
    if(typeof v==='string') return string(v);
    if(typeof v==='boolean') return v?'true':'false';
    if(typeof v==='number') { if(!Number.isSafeInteger(v)) throw Error('integer only'); return String(v); }
    if(typeof v!=='object') throw Error('JSON only');
    const keys=Object.keys(v);
    if(keys.length>10000) throw Error('collection limit');
    if(Array.isArray(v)) return '['+v.map(x=>walk(x,depth+1)).join(',')+']';
    return '{'+keys.sort(cmp).map(k=>string(k)+':'+walk(v[k],depth+1)).join(',')+'}';
  }
  const raw=walk(value,0);
  if(new TextEncoder().encode(raw).length>8388608) throw Error('byte limit');
  return raw;
}
const hex = bytes => Array.from(bytes, x=>x.toString(16).padStart(2,'0')).join('');
async function sha(raw) { return hex(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw)))); }
async function digest(domain,value) { return sha('nachos.reconciliation/'+domain+'/v1\0'+canonical(value)); }
let imported='';
document.querySelector('input').addEventListener('change',async event=>{
  imported=await event.target.files[0].text();
  // Display-only probe. Do not reuse this parser as a production validator.
  JSON.parse(imported);
  document.querySelector('pre').textContent=imported;
  document.querySelector('textarea').value=imported;
  document.querySelector('output').textContent='Fixture loaded (not validated by this prototype)';
});
document.querySelector('button').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([imported],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download='fixture-roundtrip.json'; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),0);
});
"""


def main():  # noqa: PLR0915 - linear fixture probe script
    root = os.environ.get("TMPDIR")
    if not root:
        raise RuntimeError("TMPDIR must identify the isolated scratch location")
    with tempfile.TemporaryDirectory(prefix="reconciliation-probe-", dir=root) as tmp_name:
        tmp = Path(tmp_name)
        script_hash = (
            __import__("base64").b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
        )
        html = (
            '<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'sha256-'
            + script_hash
            + "'; connect-src 'none'; img-src 'none'; style-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'\"><title>Fixture toolchain probe — not viewer</title><h1>Fixture toolchain probe — not viewer</h1><label>Fixture JSON<input type=\"file\" accept=\".json\"></label><output></output><pre></pre><textarea aria-label=\"Exact fixture text\"></textarea><button>Download fixture copy</button><script>"
            + SCRIPT
            + "</script>"
        )
        html_path = tmp / "probe.html"
        html_path.write_text(html, encoding="utf-8")
        vectors = json.loads((HERE / "hash-vectors.json").read_text())
        packet = loads((HERE / "packet.json").read_bytes())
        errors, external = [], []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True, service_workers="block")
            context.route("http://**/*", lambda route: route.abort())
            context.route("https://**/*", lambda route: route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda req: (
                    external.append(req.url)
                    if req.url.startswith(("http:", "https:", "ws:", "wss:"))
                    else None
                ),
            )
            page.goto(html_path.as_uri())
            assert page.evaluate("isSecureContext && !!crypto.subtle"), (
                "file:// WebCrypto unavailable"
            )
            for vector in vectors["canonical"]:
                result = page.evaluate(
                    "async v => ({utf8_hex:hex(new TextEncoder().encode(canonical(v))),proposal_domain_digest:await digest('proposal',v)})",
                    vector["value"],
                )
                assert result == {k: vector[k] for k in ("utf8_hex", "proposal_domain_digest")}
            artifact_count = 0
            for path in sorted(HERE.glob("*.json")):
                if path.name == "hash-vectors.json":
                    continue
                doc = loads(path.read_bytes())
                actual = page.evaluate(
                    "async d => {const {digest:ignored,...payload}=d; return digest(d.schema.split('.')[2],payload);}",
                    doc,
                )
                assert actual == doc["digest"], path.name
                artifact_count += 1
            for proposal in packet["proposals"]:
                actual = page.evaluate(
                    "async ({packet,item}) => {const findings=item.finding_ids.map(id=>packet.findings.find(f=>f.id===id)); const ids=new Set(findings.flatMap(f=>f.source_ids)); return digest('proposal',{profile:packet.profile,packet_digest:packet.digest,proposal:item,findings,sources:packet.inventory.sources.filter(s=>ids.has(s.id)),inventory_coverage:packet.inventory.coverage});}",
                    {"packet": packet, "item": proposal},
                )
                assert actual == proposal_digest(packet, proposal)
            for source in packet["inventory"]["sources"]:
                actual = page.evaluate(
                    "async s=>s.kind==='memory_record'?digest('memory-record',s.before):sha(s.before)",
                    source,
                )
                assert actual == source["sha256"]
            for name in ("packet.json", "session-reviewed.json"):
                raw = (HERE / name).read_text(encoding="utf-8")
                page.get_by_label("Fixture JSON").set_input_files(HERE / name)
                page.wait_for_function(
                    "document.querySelector('output').textContent.startsWith('Fixture loaded')"
                )
                page.wait_for_function(
                    "raw=>document.querySelector('pre').textContent===raw", arg=raw
                )
                assert page.get_by_label("Exact fixture text").input_value() == raw
                assert page.locator("img,iframe,object").count() == 0
                assert page.evaluate("window.fixtureExecuted") is None
                with page.expect_download() as download:
                    page.get_by_role("button", name="Download fixture copy").click()
                saved = tmp / name
                download.value.save_as(saved)
                assert saved.read_bytes() == (HERE / name).read_bytes()
                assert loads(saved.read_bytes()) == loads((HERE / name).read_bytes())
            assert not errors, errors
            assert not external, external
            print(
                json.dumps(
                    {
                        "browser": browser.version,
                        "transport": "file://",
                        "webcrypto": True,
                        "canonical_vectors": len(vectors["canonical"]),
                        "artifact_digests": artifact_count,
                        "proposal_digests": len(packet["proposals"]),
                        "source_hashes": len(packet["inventory"]["sources"]),
                        "native_file_import_download_roundtrips": 2,
                        "page_errors": errors,
                        "external_requests": external,
                        "scope": "fixture probe only; no production viewer E2E",
                    },
                    indent=2,
                )
            )
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
