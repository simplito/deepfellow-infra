import pytest

from server.utils.core import Utils


@pytest.fixture
def utils():
    return Utils()


@pytest.mark.parametrize(
    ("a", "b", "expectation"),
    [
        ("http://google.com", "article/123", "http://google.com/article/123"),
        ("http://google.com", "/article/123", "http://google.com/article/123"),
        ("http://google.com/", "article/123", "http://google.com/article/123"),
        ("http://google.com/", "/article/123", "http://google.com/article/123"),
    ],
)
def test_join_url(a: str, b: str, expectation: str, utils: Utils):
    result = utils.join_url(a, b)

    assert result == expectation
