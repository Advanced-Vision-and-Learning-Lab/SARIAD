import io
import tarfile
import zipfile

import pytest

from SARIAD.utils.blob_utils import _extract_archive


def _tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_zip_single_top_level_dir_is_flattened(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("top/a.txt", "1")
        z.writestr("top/sub/b.txt", "2")
    out = tmp_path / "out"
    out.mkdir()
    _extract_archive(str(tmp_path / "a.zip"), str(out), "zip")
    assert (out / "a.txt").read_text() == "1" and (out / "sub" / "b.txt").read_text() == "2"


def test_tar_traversal_is_blocked(tmp_path):
    _tar(tmp_path / "evil.tar", [("../escaped.txt", b"x")])
    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_archive(str(tmp_path / "evil.tar"), str(tmp_path / "out"), "tar")
    assert not (tmp_path / "escaped.txt").exists()


def test_zip_traversal_is_blocked(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_archive(str(tmp_path / "evil.zip"), str(tmp_path / "out"), "zip")
    assert not (tmp_path / "escaped.txt").exists()
