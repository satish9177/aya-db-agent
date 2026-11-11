"""Expert episodes and SFT labels: replay audit, assistant-only masking, turn-ending tokens."""

import pytest

from ayadb.expert import build_episode, is_verified
from ayadb.tasks import SHARED, file_sha256, generate_tasks, load_tasks


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from ayadb.engine import ollama_blob, tokenizer_from_gguf

        return tokenizer_from_gguf(ollama_blob("hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0"))
    except Exception as e:  # needs the local GGUF; the HF tokenizer is checked on Kaggle
        pytest.skip(f"Tiny Aya GGUF not available: {e}")


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    db, tasks = generate_tasks("pool1101", 1101, SHARED, 2, d)
    return db, load_tasks(tasks)


@pytest.mark.parametrize("mistake", [None, "column", "table"])
def test_every_family_builds_verified_replayable_episodes(pool, tokenizer, mistake):
    from ayadb.sftdata import replay_audit
    from ayadb.smoke import FORMATS, SYSTEM_PROMPT

    db, tasks = pool
    for fam in SHARED:
        t = next(t for t in tasks if t.family == fam)
        ep = build_episode(t, db, mistake)
        assert is_verified(ep), (t.id, ep.answer, t.answer)
        if mistake:
            wrong = [c for c in ep.calls if not c["train"]]
            assert len(wrong) == 1 and str(wrong[0]["result"]).startswith(("Error", "SQL error"))
        assert replay_audit(ep, 1101, file_sha256(db), tokenizer, SYSTEM_PROMPT,
                            FORMATS["multistep-close"]["fallback"]) == []


def test_labels_cover_exactly_the_trained_assistant_turns(pool, tokenizer):
    from ayadb.sft import tool_schemas
    from ayadb.sftdata import conversation, system_message, tokenize
    from ayadb.smoke import FORMATS, SYSTEM_PROMPT

    db, tasks = pool
    t = next(t for t in tasks if t.family == "customer_shipped_year")
    ep = build_episode(t, db, "column")
    schemas = tool_schemas()
    conv = conversation(t.prompt, ep.messages, system_message(
        SYSTEM_PROMPT, FORMATS["multistep-close"]["fallback"], schemas))
    flags = [False, False] + [m.get("train", False) for m in ep.messages]
    ex = tokenize(tokenizer, conv, flags, schemas, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    spans, cur = [], []
    for i, lab in zip(ids, labels):
        if lab == -100:
            if cur:
                spans.append(cur)
                cur = []
        else:
            assert lab == i
            cur.append(i)
    if cur:
        spans.append(cur)
    texts = [tokenizer.decode(s, skip_special_tokens=False) for s in spans]
    end = "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    want = [m["content"] + end for m in ep.messages if m["role"] == "assistant" and m["train"]]
    assert texts == want
    wrong = next(m["content"] for m in ep.messages if m["role"] == "assistant" and not m["train"])
    assert not any(wrong in x for x in texts)          # the deliberate mistake is not a target
    assert not any("Tool result for" in x for x in texts)  # tool observations are not targets
