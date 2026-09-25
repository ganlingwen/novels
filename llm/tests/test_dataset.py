import pytest

from LocalNovelDataset import NovelDPODataset
from NovelSFTDataset import NovelSFTDataset


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize
        assert not enable_thinking
        prompt = list(range(10))
        if add_generation_prompt:
            return prompt
        return prompt + list(range(100, 112))


def test_long_prompt_is_left_truncated_to_preserve_response_labels():
    raw = [{"messages": [{"role": "user"}, {"role": "assistant"}]}]
    dataset = NovelSFTDataset(raw, FakeTokenizer(), max_length=8)

    item = dataset[0]

    assert item["input_ids"].tolist() == [6, 7, 8, 9, 100, 101, 102, 103]
    assert item["labels"].tolist() == [-100, -100, -100, -100, 100, 101, 102, 103]
    assert item["labels"][1:].ne(-100).any()


def test_max_length_requires_room_for_a_prediction():
    raw = [{"messages": [{"role": "user"}, {"role": "assistant"}]}]

    with pytest.raises(ValueError, match="at least 2"):
        NovelSFTDataset(raw, FakeTokenizer(), max_length=1)


def test_dpo_dataset_preserves_response_labels_after_prompt_truncation():
    dataset = NovelDPODataset(
        [{"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}], FakeTokenizer(), max_length=8
    )

    item = dataset[0]

    assert item["prompt_ids"].tolist() == [6, 7, 8, 9]
    assert item["chosen_ids"].tolist() == [100, 101, 102, 103]
    assert item["rejected_ids"].tolist() == [100, 101, 102, 103]
