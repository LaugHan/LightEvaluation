import json

from eval_pipeline.core.io import append_jsonl, merge_shards, read_records, scan_done


def test_append_jsonl_flushes_records_and_scan_done_collects_id_sample_pairs(tmp_path):
    output = tmp_path / "output.jsonl"

    append_jsonl(output, [{"id": "a", "sample_idx": 0}, {"id": "b", "sample_idx": 2}], flush_every=1)

    assert scan_done(output) == {("a", 0), ("b", 2)}
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "a", "sample_idx": 0},
        {"id": "b", "sample_idx": 2},
    ]


def test_scan_done_ignores_bad_tail_line_only(tmp_path):
    output = tmp_path / "output.jsonl"
    output.write_text('{"id": "a", "sample_idx": 0}\n{"id": ')

    assert scan_done(output) == {("a", 0)}
    assert output.read_text() == '{"id": "a", "sample_idx": 0}\n'


def test_scan_done_raises_for_bad_middle_line(tmp_path):
    output = tmp_path / "output.jsonl"
    output.write_text('{"id": "a", "sample_idx": 0}\nnot-json\n{"id": "b", "sample_idx": 1}\n')

    try:
        scan_done(output)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("bad middle line should fail loudly")


def test_merge_shards_writes_unique_records_in_shard_order(tmp_path):
    shard0 = tmp_path / "output.rank0.jsonl"
    shard1 = tmp_path / "output.rank1.jsonl"
    merged = tmp_path / "output.jsonl"
    append_jsonl(shard0, [{"id": "a", "sample_idx": 0}, {"id": "b", "sample_idx": 0}])
    append_jsonl(shard1, [{"id": "a", "sample_idx": 0}, {"id": "c", "sample_idx": 1}])

    count = merge_shards([shard0, shard1], merged)

    assert count == 3
    assert read_records(merged) == [
        {"id": "a", "sample_idx": 0},
        {"id": "b", "sample_idx": 0},
        {"id": "c", "sample_idx": 1},
    ]
