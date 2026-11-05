"""倒数排名融合：纯函数，不连接 Qdrant。"""

from __future__ import annotations

from masskb.fusion import RRF_K, FusedHit, _fusion_sort_key, reciprocal_rank_fusion


def test_rrf_hand_value_for_one_shared_hit() -> None:
    fused = reciprocal_rank_fusion([("a", 0.2)], [("a", 0.4)], top_k=5)

    assert len(fused) == 1
    assert fused[0].score == 1 / (RRF_K + 1) + 1 / (RRF_K + 1)
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1
    assert fused[0].dense_score == 0.2
    assert fused[0].sparse_score == 0.4


def test_dense_only_and_sparse_only_keep_input_order() -> None:
    dense = reciprocal_rank_fusion([("b", 0.2), ("a", 0.9)], [], top_k=5)
    sparse = reciprocal_rank_fusion([], [("c", 1.0), ("d", 0.2)], top_k=5)

    assert [hit.chunk_id for hit in dense] == ["b", "a"]
    assert dense[0].sparse_rank is None
    assert dense[0].sparse_score is None
    assert [hit.chunk_id for hit in sparse] == ["c", "d"]
    assert sparse[0].dense_rank is None


def test_overlap_orders_by_rrf_then_dense_rank() -> None:
    fused = reciprocal_rank_fusion(
        [("a", 0.9), ("b", 0.1)],
        [("b", 0.9), ("a", 0.1)],
        top_k=5,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert fused[0].score == fused[1].score
    assert fused[0].dense_rank == 1


def test_top_k_truncates_and_duplicate_rank_keeps_first() -> None:
    truncated = reciprocal_rank_fusion([("a", 1.0), ("b", 1.0), ("c", 1.0)], [], top_k=1)
    duplicate = reciprocal_rank_fusion([("a", 0.9), ("a", 0.1)], [], top_k=5)

    assert [hit.chunk_id for hit in truncated] == ["a"]
    assert len(duplicate) == 1
    assert duplicate[0].dense_rank == 1
    assert duplicate[0].dense_score == 0.9


def test_chunk_id_breaks_ties_when_score_and_dense_rank_match() -> None:
    later = FusedHit("b", 0.5, None, 1.0, None, 1)
    earlier = FusedHit("a", 0.5, None, 1.0, None, 1)

    ordered = sorted([later, earlier], key=_fusion_sort_key)

    assert [hit.chunk_id for hit in ordered] == ["a", "b"]
