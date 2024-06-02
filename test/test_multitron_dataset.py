import numpy as np
from torch import allclose, tensor
from torch.utils.data.dataloader import DataLoader

from src.dataset import MultiTronDataset, label_session


def test_label_session():
    session = [{
        "aid": 33838,
        "ts": 1464127201.187,
        "type": "clicks"
    }, {
        "aid": 33838,
        "ts": 1464127201.522,
        "type": "carts"
    }, {
        "aid": 4759,
        "ts": 1464127218.472,
        "type": "clicks"
    }, {
        "aid": 27601,
        "ts": 1464127251.938,
        "type": "clicks"
    }, {
        "aid": 15406,
        "ts": 1464127265.936,
        "type": "clicks"
    }, {
        "aid": 4759,
        "ts": 1464127406.279,
        "type": "orders"
    }]
    expected_clicks = [33838, 4759, 27601]
    expected_click_labels = [4759, 27601, 15406]
    expected_order_labels = [1., 0., 0.]
    expected_session_len = 3
    expected_mask = [1., 1., 1.]

    clicks, click_labels, order_labels, session_len, mask = label_session(session)

    assert clicks == expected_clicks
    assert click_labels == expected_click_labels
    assert order_labels == expected_order_labels
    assert session_len == expected_session_len
    assert mask == expected_mask


def test_dataset():
    session_path = "test/resources/multitron_train.jsonl"

    dataset = MultiTronDataset(session_path,
                               total_sessions=10,
                               num_items=40_000,
                               max_seqlen=4,
                               shuffling_style="no_shuffling",
                               num_uniform_negatives=3,
                               num_in_batch_negatives=0,
                               reject_uniform_session_items=False,
                               sampling_style="eventwise")

    expected_first_session = {
        'clicks': [33838, 4759, 12887, 27601],
        'click_labels': [4759, 12887, 27601, 15406],
        'order_labels': [1., 0., 0., 0.],
        'mask': [1., 1., 1., 1.],
        'session_len': 4,
    }

    expected_second_session = {'clicks': [], 'click_labels': [], 'order_labels': [], 'session_len': 0, 'mask': []}

    expected_third_session = {
        'clicks': [31292],
        'click_labels': [12957],
        'order_labels': [0.],
        'session_len': 1,
        'mask': [1.],
    }

    expected_fourth_session = {
        'clicks': [],
        'click_labels': [],
        'order_labels': [],
        'session_len': 0,
        'mask': [],
    }

    first_session = dataset.__getitem__(0)
    second_session = dataset.__getitem__(1)
    third_session = dataset.__getitem__(2)
    fourth_session = dataset.__getitem__(3)

    assert first_session['clicks'] == expected_first_session['clicks']
    assert first_session['click_labels'] == expected_first_session['click_labels']
    assert first_session['order_labels'] == expected_first_session['order_labels']
    assert first_session['mask'] == expected_first_session['mask']
    assert first_session['session_len'] == expected_first_session['session_len']
    assert np.array(first_session['uniform_negatives']).shape == (4, 3)

    assert second_session['clicks'] == expected_second_session['clicks']
    assert second_session['click_labels'] == expected_second_session['click_labels']
    assert second_session['order_labels'] == expected_second_session['order_labels']
    assert second_session['mask'] == expected_second_session['mask']
    assert second_session['session_len'] == expected_second_session['session_len']
    assert np.array(second_session['uniform_negatives']).shape == (0, )

    assert third_session['clicks'] == expected_third_session['clicks']
    assert third_session['click_labels'] == expected_third_session['click_labels']
    assert third_session['order_labels'] == expected_third_session['order_labels']
    assert third_session['mask'] == expected_third_session['mask']
    assert third_session['session_len'] == expected_third_session['session_len']
    assert np.array(third_session['uniform_negatives']).shape == (1, 3)

    assert fourth_session['clicks'] == expected_fourth_session['clicks']
    assert fourth_session['click_labels'] == expected_fourth_session['click_labels']
    assert fourth_session['order_labels'] == expected_fourth_session['order_labels']
    assert fourth_session['mask'] == expected_fourth_session['mask']
    assert fourth_session['session_len'] == expected_fourth_session['session_len']
    assert np.array(fourth_session['uniform_negatives']).shape == (0, )


def test_datalaoder():
    session_path = "test/resources/multitron_train.jsonl"
    dataset = MultiTronDataset(sessions_path=session_path,
                               total_sessions=10,
                               num_items=40_000,
                               max_seqlen=4,
                               shuffling_style="no_shuffling",
                               num_uniform_negatives=3,
                               num_in_batch_negatives=1,
                               reject_uniform_session_items=True,
                               sampling_style="eventwise")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=dataset.dynamic_collate)

    expected_first_batch = {
        'clicks': tensor([[33838, 4759, 12887, 27601], [0, 0, 0, 0], [0, 0, 0, 31292]]),
        'click_labels': tensor([[4759, 12887, 27601, 15406], [0, 0, 0, 0], [0, 0, 0, 12957]]),
        'order_labels': tensor([[1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
        'session_len': tensor([4, 0, 1]),
        'mask': tensor([[1., 1., 1., 1.], [0., 0., 0., 0.], [0., 0., 0., 1.]]),
    }

    for batch in dataloader:
        print(batch["click_labels"])
        assert allclose(batch['clicks'], expected_first_batch['clicks'])
        assert allclose(batch['click_labels'], expected_first_batch['click_labels'])
        print(batch['order_labels'])
        assert allclose(batch['order_labels'], expected_first_batch['order_labels'])
        assert allclose(batch['mask'], expected_first_batch['mask'])
        assert allclose(batch['session_len'], expected_first_batch['session_len'])
        assert batch['in_batch_negatives'].shape == (3, 1)
        assert batch['uniform_negatives'].shape == (3, 4, 3)
        assert set(batch['in_batch_negatives'].tolist()[0]).issubset([31292])
        assert set(batch['in_batch_negatives'].tolist()[1]).issubset([33838, 4759, 12887, 27601, 31292])
        assert set(batch['in_batch_negatives'].tolist()[2]).issubset([33838, 4759, 12887, 27601])
        break

    dataset.sampling_style = "sessionwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3, 3)

    dataset.sampling_style = "batchwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3, )


def test_datalaoder_no_uniform_negatives():
    session_path = "test/resources/multitron_train.jsonl"
    dataset = MultiTronDataset(sessions_path=session_path,
                               total_sessions=10,
                               num_items=40_000,
                               max_seqlen=6,
                               shuffling_style="no_shuffling",
                               num_uniform_negatives=0,
                               num_in_batch_negatives=1,
                               reject_uniform_session_items=True,
                               sampling_style="eventwise")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=dataset.dynamic_collate)

    for batch in dataloader:
        assert batch['uniform_negatives'].shape == (3, 6, 0)
        break

    dataset.sampling_style = "sessionwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3, 0)

    dataset.sampling_style = "batchwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (0, )


def test_datalaoder_no_in_batch_negatives():
    session_path = "test/resources/multitron_train.jsonl"
    dataset = MultiTronDataset(sessions_path=session_path,
                               total_sessions=10,
                               num_items=40_000,
                               max_seqlen=6,
                               shuffling_style="no_shuffling",
                               num_uniform_negatives=3,
                               num_in_batch_negatives=0,
                               reject_uniform_session_items=True,
                               sampling_style="eventwise")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=dataset.dynamic_collate)

    for batch in dataloader:
        assert batch['in_batch_negatives'].shape == (3, 0)
        break
