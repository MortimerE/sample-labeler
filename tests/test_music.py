import pytest

from autolabel.domain import Key
from autolabel.music import key_dict, relation, relative


def test_relative_relation_is_symmetric():
    g_minor = Key(7, "minor")
    b_flat_major = Key(10, "major")
    assert relative(g_minor) == b_flat_major
    assert relative(b_flat_major) == g_minor
    assert relation(g_minor, b_flat_major) == "relative"
    assert relation(b_flat_major, g_minor) == "relative"


@pytest.mark.parametrize(
    ("key", "camelot"),
    [(Key(7, "minor"), "6A"), (Key(10, "major"), "6B"), (Key(0, "major"), "8B")],
)
def test_camelot_mapping(key, camelot):
    assert key_dict(key)["camelot"] == camelot


def test_fifth_relation_requires_same_mode():
    assert relation(Key(0, "major"), Key(7, "major")) == "fifth"
    assert relation(Key(0, "major"), Key(7, "minor")) is None

