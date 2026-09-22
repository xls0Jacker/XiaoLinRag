import pytest

from xiaolinrag.index_store import IndexEntry, ScoredHit
from xiaolinrag.retrieval import fuse_rrf, fuse_rrf_multi

K = 60


def hit(chunk_id: str, score: float = 1.0) -> ScoredHit:
    entry = IndexEntry(
        chunk_id=chunk_id,
        parent_id=chunk_id.split("#")[0] + "#p0",
        page_id=chunk_id.split("#")[0],
        section="rag",
        title="示例",
        source_url="https://example.com",
        heading_path="示例",
        child_text=chunk_id,
        context_note="",
        parent_text=chunk_id,
        seq=0,
    )
    return ScoredHit(entry=entry, score=score)


def ids(hits):
    return [h.entry.chunk_id for h in hits]


def test_both_paths_hit_ranks_first():
    vector = [hit("a#c0"), hit("b#c0")]
    keyword = [hit("b#c0"), hit("c#c0")]

    fused = fuse_rrf(vector, keyword, K)

    assert ids(fused)[0] == "b#c0", "双路都命中的子块应排最前"


def test_score_is_sum_of_reciprocal_ranks():
    vector = [hit("a#c0"), hit("b#c0")]
    keyword = [hit("b#c0")]

    fused = {h.entry.chunk_id: h.score for h in fuse_rrf(vector, keyword, K)}

    assert fused["a#c0"] == pytest.approx(1 / (K + 1))
    assert fused["b#c0"] == pytest.approx(1 / (K + 2) + 1 / (K + 1))


def test_deduplicates_across_paths():
    vector = [hit("a#c0"), hit("b#c0")]
    keyword = [hit("a#c0"), hit("b#c0")]

    fused = fuse_rrf(vector, keyword, K)

    assert len(fused) == 2
    assert len(ids(fused)) == len(set(ids(fused)))


def test_order_follows_rank_within_single_path():
    vector = [hit("a#c0"), hit("b#c0"), hit("c#c0")]

    assert ids(fuse_rrf(vector, [], K)) == ["a#c0", "b#c0", "c#c0"]


def test_one_path_empty():
    keyword = [hit("x#c0"), hit("y#c0")]

    assert ids(fuse_rrf([], keyword, K)) == ["x#c0", "y#c0"]


def test_both_paths_empty():
    assert fuse_rrf([], [], K) == []


def test_ties_are_deterministic():
    vector = [hit("a#c0"), hit("b#c0")]
    keyword = [hit("b#c0"), hit("a#c0")]

    first = ids(fuse_rrf(vector, keyword, K))
    second = ids(fuse_rrf(vector, keyword, K))

    assert first == second == ["a#c0", "b#c0"]  # 同分时按 chunk_id 稳定排序


# ---------- fuse_rrf_multi（多 query × 双通道泛化） ----------


def test_multi_three_paths_accumulate_scores():
    route1 = [hit("a#c0"), hit("b#c0")]
    route2 = [hit("b#c0"), hit("c#c0")]
    route3 = [hit("c#c0"), hit("d#c0")]

    fused = {h.entry.chunk_id: h.score for h in fuse_rrf_multi([route1, route2, route3], K)}

    assert fused["a#c0"] == pytest.approx(1 / (K + 1))
    assert fused["b#c0"] == pytest.approx(1 / (K + 2) + 1 / (K + 1))
    assert fused["c#c0"] == pytest.approx(1 / (K + 2) + 1 / (K + 1))
    assert fused["d#c0"] == pytest.approx(1 / (K + 2))


def test_multi_dedupes_across_more_than_two_paths():
    routes = [[hit("a#c0")], [hit("a#c0")], [hit("a#c0"), hit("b#c0")]]

    fused = fuse_rrf_multi(routes, K)

    assert len(fused) == 2
    assert ids(fused) == ["a#c0", "b#c0"]


def test_multi_empty_input():
    assert fuse_rrf_multi([], K) == []
    assert fuse_rrf_multi([[], []], K) == []


def test_multi_ties_deterministic():
    routes = [[hit("a#c0")], [hit("a#c0"), hit("b#c0")], [hit("b#c0")]]

    first = ids(fuse_rrf_multi(routes, K))
    second = ids(fuse_rrf_multi(routes, K))

    assert first == second == ["a#c0", "b#c0"]


def test_fuse_rrf_delegates_to_multi():
    vector = [hit("a#c0"), hit("b#c0")]
    keyword = [hit("b#c0")]

    two_way = fuse_rrf(vector, keyword, K)
    multi_way = fuse_rrf_multi([vector, keyword], K)

    assert ids(two_way) == ids(multi_way)
    assert {h.entry.chunk_id: h.score for h in two_way} == {
        h.entry.chunk_id: h.score for h in multi_way
    }
