"""Tests for shared_media.py - verifies shared storage plot serving."""
import os
import hashlib
import unittest.mock as mock

import pytest
import matplotlib.pyplot as plt

import shared_media


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    """Patch shared_media module to use a temporary directory."""
    d = tmp_path / "media"
    d.mkdir()
    monkeypatch.setattr(shared_media, "MEDIA_DIR", str(d))
    monkeypatch.setattr(shared_media, "MEDIA_URL", "/media")
    return d


def test_shared_pyplot_writes_png(media_dir):
    """shared_pyplot writes a content-hashed PNG to SHARED_MEDIA_DIR."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    with mock.patch("streamlit.components.v1.html"):
        shared_media.shared_pyplot(fig)
    plt.close(fig)

    files = list(media_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".png"
    assert files[0].stat().st_size > 0


def test_shared_pyplot_uses_content_hash(media_dir):
    """The filename is a sha256 hash of the image content."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    with mock.patch("streamlit.components.v1.html"):
        shared_media.shared_pyplot(fig)
    plt.close(fig)

    written_file = list(media_dir.iterdir())[0]
    content = written_file.read_bytes()
    expected_name = hashlib.sha256(content).hexdigest()[:16] + ".png"
    assert written_file.name == expected_name


def test_shared_pyplot_emits_correct_url(media_dir):
    """The HTML img tag references /media/<hash>.png."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    with mock.patch("streamlit.components.v1.html") as mock_html:
        shared_media.shared_pyplot(fig)
    plt.close(fig)

    call_args = mock_html.call_args[0][0]
    written_file = list(media_dir.iterdir())[0]
    expected_url = f"/media/{written_file.name}"
    assert expected_url in call_args
    assert '<img src="' in call_args


def test_shared_pyplot_idempotent(media_dir):
    """Calling shared_pyplot with identical figures doesn't create duplicates."""
    for _ in range(3):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        with mock.patch("streamlit.components.v1.html"):
            shared_media.shared_pyplot(fig)
        plt.close(fig)

    files = list(media_dir.iterdir())
    assert len(files) == 1


def test_fallback_to_st_pyplot_when_unavailable(monkeypatch, tmp_path):
    """Falls back to st.pyplot when shared storage is unavailable."""
    # Create a file where the directory should be - makedirs will fail
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("block")
    monkeypatch.setattr(shared_media, "MEDIA_DIR", str(blocker / "subdir"))

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    with mock.patch("streamlit.pyplot") as mock_st_pyplot:
        shared_media.shared_pyplot(fig)
    plt.close(fig)

    mock_st_pyplot.assert_called_once()
