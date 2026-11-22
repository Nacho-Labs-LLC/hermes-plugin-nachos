"""The harness itself: file:// only, every network attempt aborted AND recorded."""

import pytest

PAGE = """<!doctype html><meta charset="utf-8"><title>harness</title>
<input id="f" type="file"><a id="dl" href="#">save</a><pre id="out">idle</pre>
<script>
const out = document.getElementById('out');
document.getElementById('f').addEventListener('change', async e => {
  out.textContent = 'read:' + await e.target.files[0].text();
});
document.getElementById('dl').addEventListener('click', e => {
  e.preventDefault();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['saved\\u00e9'], {type: 'application/json'}));
  a.download = 'out.json';
  document.body.appendChild(a); a.click(); a.remove();
});
window.attempts = (async () => {
  const result = {};
  for (const [name, fn] of Object.entries({
    fetch: () => fetch('https://invalid.example/fetch'),
    xhr: () => new Promise((ok, no) => { const x = new XMLHttpRequest();
      x.open('GET', 'http://invalid.example/xhr'); x.onload = ok; x.onerror = () => no(new Error('xhr'));
      x.send(); }),
    ws: () => new Promise(ok => { new WebSocket('wss://invalid.example/ws'); ok(); }),
  })) {
    try { await fn(); result[name] = 'reached'; } catch (e) { result[name] = 'blocked'; }
  }
  const img = new Image(); img.src = 'https://invalid.example/img.png';
  await new Promise(r => { img.onerror = r; img.onload = r; });
  result.img = 'blocked';
  return result;
})();
</script>"""


@pytest.fixture
def page_path(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(PAGE, encoding="utf-8")
    return path


def test_network_is_denied_and_recorded(viewer_page, page_path):
    viewer_page.goto(page_path)
    assert viewer_page.page.title() == "harness"
    result = viewer_page.page.evaluate("window.attempts")
    assert result == {"fetch": "blocked", "xhr": "blocked", "ws": "blocked", "img": "blocked"}
    recorded = viewer_page.recorder.external
    for needle in ("/fetch", "/xhr", "/ws", "/img.png"):
        assert any(url.endswith(needle) for url in recorded), (needle, recorded)
    with pytest.raises(AssertionError, match=r"invalid\.example"):
        viewer_page.assert_isolated()
    assert viewer_page.recorder.requests[0].startswith("file://")


def test_clean_page_records_no_external_requests(viewer_page, tmp_path):
    clean = tmp_path / "clean.html"
    clean.write_text("<!doctype html><title>clean</title><p>hello", encoding="utf-8")
    viewer_page.goto(clean)
    viewer_page.assert_isolated()
    assert viewer_page.recorder.non_file == []


def test_file_import_and_download_round_trip(viewer_page, page_path, tmp_path):
    viewer_page.goto(page_path)
    source = tmp_path / "in.json"
    source.write_text('{"a":"\U0001f600"}', encoding="utf-8")
    viewer_page.import_file("#f", source)
    viewer_page.page.wait_for_function(
        "document.getElementById('out').textContent.startsWith('read:')"
    )
    assert viewer_page.page.inner_text("#out") == 'read:{"a":"\U0001f600"}'
    saved = viewer_page.capture_download(lambda: viewer_page.page.click("#dl"))
    assert saved.parent == viewer_page.downloads_dir
    assert saved.name == "out.json"
    assert saved.read_text(encoding="utf-8") == "savedé"


def test_canary_detection_and_viewports(viewer_page, narrow_viewer_page, page_path):
    viewer_page.goto(page_path)
    assert viewer_page.page.evaluate("innerWidth") == 1280
    assert not viewer_page.canary_fired()
    viewer_page.page.evaluate("window.__pwned = 1")
    assert viewer_page.canary_fired()
    narrow_viewer_page.goto(page_path)
    assert narrow_viewer_page.page.evaluate("innerWidth") == 375


def test_contexts_are_isolated(open_viewer_page, tmp_path):
    clean = tmp_path / "clean.html"
    clean.write_text("<!doctype html><title>x</title>", encoding="utf-8")
    one, two = open_viewer_page(), open_viewer_page("narrow")
    one.goto(clean)
    one.page.evaluate("localStorage.setItem('k', 'v')")
    two.goto(clean)
    assert two.page.evaluate("localStorage.getItem('k')") is None
    assert two.recorder.requests == [clean.as_uri()]
