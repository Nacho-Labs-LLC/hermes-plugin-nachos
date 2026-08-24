import pytest

from nachos_core.store.md_store import MDStore


def test_markdown_injection_rejection(tmp_path):
    store = MDStore(str(tmp_path / "store.md"))

    # Body containing H1 should be rejected
    with pytest.raises(ValueError, match="Markdown injection detected"):
        store.put(
            key="test-key",
            title="Title",
            summary="Summary",
            category="General",
            body="Normal body\n# Hacked Category"
        )

    # Title containing newline should be rejected
    with pytest.raises(ValueError, match="Markdown injection detected"):
        store.put(
            key="test-key2",
            title="Title\nOther",
            summary="Summary",
            category="General",
            body="Body"
        )

    # Category containing newline should be rejected
    with pytest.raises(ValueError, match="Markdown injection detected"):
        store.put(
            key="test-key3",
            title="Title",
            summary="Summary",
            category="General\nOther",
            body="Body"
        )

    # Body containing H2 should be rejected
    with pytest.raises(ValueError, match="Markdown injection detected"):
        store.put(
            key="test-key4",
            title="Title",
            summary="Summary",
            category="General",
            body="Normal body\n## Hacked Title"
        )

    # Summary containing H1 should be rejected
    with pytest.raises(ValueError, match="Markdown injection detected"):
        store.put(
            key="test-key5",
            title="Title",
            summary="Summary\n# Hacked Category",
            category="General",
            body="Body"
        )
