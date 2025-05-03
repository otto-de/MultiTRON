import torch
from torch import tensor

from src.loss import (_elementwise_sampled_softmax_loss, approx_epo_loss, binary_cross_entropy_loss, click_ssm_order_be_loss,
                      cosine_loss, sampled_softmax_loss, weighted_loss, distortion_loss, click_ssm_distortion_loss)


def test_elementwise_sampled_softmax_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]],
        dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    target = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_elementwise_loss = tensor([[2.4938, 4.4197, 6.4093, 4.1488, 4.1488], [0.0000, 6.4093, 4.4197, 2.4938, 4.1488]])
    assert torch.allclose(expected_elementwise_loss,
                          _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target),
                          atol=1e-5)


def test_sampled_softmax_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]],
        dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    expected_loss = tensor(39.0920 / 9.)
    ssm_loss, _ = sampled_softmax_loss(pos_logits, neg_logits, mask)
    assert torch.allclose(expected_loss, ssm_loss)


def test_binary_cross_entropy_loss():
    logits = tensor([[[1.], [-2.]], [[3.], [4.]]])
    mask = tensor([[1., 1.], [0., 1.]])

    order_labels = tensor([[1., 0.], [0., 0.]])

    loss, _ = binary_cross_entropy_loss(logits, order_labels, mask)

    expected_loss = (0.3132617 + 0.1269280 + 0. + 4.0181499) / 3

    assert loss == expected_loss

def test_distortion_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]],
        dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])

    expected_elementwise_loss = tensor(
        [[1.7438, 2.1697, 2.6593, 2.8988, 2.8988], [0., 2.6593, 2.1697, 1.7438, 2.8988]], dtype=torch.float)
    expected_loss = tensor((1.7438 + 2.1697 + 2.6593 + 2.8988 + 2.8988 + 0. + 2.6593 + 2.1697 + 1.7438 + 2.8988) / 9)

    loss, elementwise_loss = distortion_loss(pos_logits, neg_logits, mask)

    assert torch.allclose(elementwise_loss, expected_elementwise_loss, atol=0.0001)
    assert torch.allclose(loss, expected_loss)


def test_click_ssm_order_be_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]],
        dtype=torch.float)

    order_labels = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 0., 0.]])
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    expected_ssm_loss = tensor(39.0920 / 9.)
    expected_bce_loss = tensor(
        (0.313262 + 0.126928 + 0.048587 + 0.01815 + 0.01815 + 0. + 0.048587 + 0.126928 + 1.313262 + 4.01815) / 9)

    ssm_loss, _, bce_loss, _ = click_ssm_order_be_loss(pos_logits, neg_logits, order_labels, mask)

    assert torch.allclose(ssm_loss, expected_ssm_loss)
    assert torch.allclose(bce_loss, expected_bce_loss)

def test_click_ssm_distortion_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]],
        dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    expected_ssm_loss = tensor(39.0920 / 9.)
    expected_distortion_loss = tensor((1.7438 + 2.1697 + 2.6593 + 2.8988 + 2.8988 + 0. + 2.6593 + 2.1697 + 1.7438 + 2.8988) / 9)

    ssm_loss, _, dis_loss, _ = click_ssm_distortion_loss(pos_logits, neg_logits, None, mask)

    assert torch.allclose(ssm_loss, expected_ssm_loss)
    assert torch.allclose(dis_loss, expected_distortion_loss)

def test_weighted_loss():
    click_loss = tensor(4.)
    second_loss = tensor(0.5)
    beta = tensor([[0.6, 0.3]])
    expected_loss = tensor(2.55)

    assert torch.allclose(weighted_loss(click_loss, second_loss, beta), expected_loss)


def test_cosine_loss():
    beta = tensor([[0.5, 0.5]])
    click_elementwise_loss = tensor([[1., -1.], [1., 0.]])
    second_elementwise_loss = tensor([[1., 1.], [1 / 2, 0.]])
    mask = tensor([[1., 1.], [1., 1.]])

    expected_cosine_loss = tensor((-1 - 0. - 0.9486833 - 0.) / 4.)

    assert torch.allclose(cosine_loss(click_elementwise_loss, second_elementwise_loss, mask, beta), expected_cosine_loss)


def test_approx_epo():
    beta = tensor([[0.6, 0.4]])
    mask = tensor([[1., 1.], [1., 0.]])
    click_elementwise_loss = tensor([[1., 3.], [1., 0.]])
    second_elementwise_loss = tensor([[1., 1.], [1 / 2, 0.]])

    # weighted  tensor([[[1., 1.], [3., 1.]],
    #                   [[1., 0.5], [0., 0.]]])

    # weighted  tensor([[[0.6, 0.4], [1.8, 0.4]],
    #                   [[0.6., 0.2], [0., 0.]]])

    #normalized tensor([[[0.6, 0.4], [9./11., 2./11.]],
    #                   [[0.75., 0.25], [0., 0.]]])

    # CE(.,[0.5,0.5]) = tensor([0.6981, 0.7429, 0.7241, 0.])

    expected = tensor((0.6981 + 0.7429 + 0.7241) / 3)

    approx_epo = approx_epo_loss(click_elementwise_loss, second_elementwise_loss, mask, beta)

    assert torch.allclose(approx_epo, expected, atol=1e-5)

