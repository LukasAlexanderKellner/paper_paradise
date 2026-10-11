from core.sources import arxiv_key
from pipeline.dedupe import dedupe
from tests.conftest import make_item


def test_cross_source_duplicate_kept_in_huggingface(sample_items):
    result = dedupe(sample_items)
    assert [i.key for i in result["huggingface"]] == ["arxiv:2610.00001"]
    assert [i.key for i in result["arxiv"]] == ["arxiv:2610.00002"]


def test_source_order_is_preserved(sample_items):
    assert list(dedupe(sample_items)) == list(sample_items)


def test_duplicates_within_one_source_are_removed():
    items = {"blogs": [make_item("blogs", "https://x/1"), make_item("blogs", "https://x/1"), make_item("blogs", "https://x/2")]}
    assert [i.key for i in dedupe(items)["blogs"]] == ["https://x/1", "https://x/2"]


def test_input_is_not_modified(sample_items):
    before = {s: list(items) for s, items in sample_items.items()}
    dedupe(sample_items)
    assert sample_items == before


def test_arxiv_key_normalizes_versions_and_url_forms():
    assert arxiv_key("http://arxiv.org/abs/2610.12448v2") == "arxiv:2610.12448"
    assert arxiv_key("https://arxiv.org/pdf/2610.12448v1") == "arxiv:2610.12448"
    assert arxiv_key("https://arxiv.org/pdf/2610.12448.pdf") == "arxiv:2610.12448"
    assert arxiv_key("2610.12448") == "arxiv:2610.12448"
