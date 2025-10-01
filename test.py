import torch

def test_function():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    assert torch.equal(z, torch.tensor([5.0, 7.0, 9.0])), "Test failed!"
    print("Test passed!")