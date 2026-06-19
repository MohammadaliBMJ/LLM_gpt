from src.positional_enocders.positional_encoders import SinusoidalPE, LearnablePE, RoPE, ALiBi
import pytest
import torch


# Sinusoidal positional encoder tests
@pytest.fixture
def sinusoidal_pe():
    return SinusoidalPE(32, 128)

def test_sinusoidal_pe_shape(sinusoidal_pe):
    """Test the Sinusoidal PE output shape to be (batch_size, seq_len, d_model)"""
    batch_size = 4
    d_model = 32
    seq_len = 64

    x = torch.zeros(batch_size, seq_len, d_model)
    
    assert sinusoidal_pe(x).shape == (batch_size, seq_len, d_model),\
    f"Expected shape {(batch_size, seq_len, d_model)}, got {sinusoidal_pe(x).shape}"

def test_sinusoidal_pe_deterministic(sinusoidal_pe):
    """Test Sinusoidal PE returns same output for same input"""
    batch_size = 4
    seq_len = 64
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    out1 = sinusoidal_pe(x)
    out2 = sinusoidal_pe(x)

    assert torch.allclose(out1, out2), \
        "SinusoidalPE is not deterministic: outputs differ for identical inputs"
    
def test_sinusoidal_pe_has_no_parameters(sinusoidal_pe):
    """Test Sinusoidal PE has no learnable parameters and it is pure math"""
    total_params = sum(p.numel() for p in sinusoidal_pe.parameters())

    assert total_params == 0, \
        f"SinusoidalPE should have 0 parameters, found {total_params}"
    
def test_sinusoidal_pe_slicing(sinusoidal_pe):
    """Test for seq_len less than max_len the PE works correctly"""
    batch_size = 2
    seq_len = 64    
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)
    out = sinusoidal_pe(x)

    # The PE used should be exactly the first seq_len positions
    expected_pe = sinusoidal_pe.pe[:, :seq_len, :].to(out.device)
    expected_pe = expected_pe.expand(batch_size, -1, -1)

    assert torch.allclose(out - x, expected_pe, atol=1e-6), \
        "SinusoidalPE did not correctly slice positional encodings for shorter sequences"
    
def test_sinusoidal_pe_device_movement(sinusoidal_pe):
    """Test output device is the same as input device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 4
    seq_len = 64
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model).cuda()
    out = sinusoidal_pe(x)

    assert out.device == x.device, \
        f"Expected output on {x.device}, got {out.device}"
    
def test_sinusoidal_pe_broadcasting(sinusoidal_pe):
    """Test all batches receive the same PE"""
    batch_size = 4
    seq_len = 64
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)
    out = sinusoidal_pe(x)
    pe0 = out[0] - x[0]   # (seq_len, d_model)


    for b in range(1, batch_size):
        assert torch.allclose(out[b] - x[b], pe0, atol=1e-6), \
            f"Positional encoding differs for batch index {b}"
        

# Learnable positional encoder tests
@pytest.fixture
def learnable_pe():
    return LearnablePE(32, 128)

def test_learnable_pe_shape(learnable_pe):
    """Test output shape for smaller seq_len"""
    batch_size = 4
    seq_len = 64
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    out = learnable_pe(x)

    assert x.shape == out.shape,\
    f"Expected shape {(batch_size, seq_len, d_model)}, got {sinusoidal_pe(x).shape}"

def test_learnable_pe_has_parameters(learnable_pe):
    """Verify LearnablePE contains exactly one learnable parameter of correct shape."""
    d_model = learnable_pe.d_model
    max_len = learnable_pe.max_len

    params = list(learnable_pe.parameters())
    assert len(params) == 1, "LearnablePE should have exactly 1 parameter"

    assert params[0].shape == (1, max_len, d_model), \
        f"Expected parameter shape (1, {max_len}, {d_model}), got {params[0].shape}"

def test_learnable_pe_slicing(learnable_pe):
    """Check that LearnablePE correctly slices positional encodings for shorter sequences."""
    x = torch.randn(2, 64, 32)

    out = learnable_pe(x)
    added = out - x

    expected = learnable_pe.pe[:, :64, :].expand_as(added)

    assert torch.allclose(added, expected, atol=1e-6),\
            "LearnablePE did not correctly slice positional encodings for shorter sequences"

def test_learnable_pe_broadcasting(learnable_pe):
    "Test the broadcasting is correct and all the batches get the same PE"
    x = torch.randn(4, 64, 32)

    out = learnable_pe(x)
    pe0 = out[0] - x[0]

    for b in range(1, 4):
        assert torch.allclose(out[b] - x[b], pe0, atol=1e-6),\
        f"Batch index {b} received different positional encodings"

def test_learnable_pe_device_movement(learnable_pe):
    "Test input and output device are the same"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(2, 64, 32).cuda()

    out = learnable_pe(x)

    assert out.device == x.device, \
    f"Output device {out.device} does not match input device {x.device}"


# RoPE positional encoder tests
@pytest.fixture
def rope():
    return RoPE(32)

def test_rope_output_shape(rope):
    """RoPE should return Q and K with the same shape as the inputs."""
    batch = 2
    seq = 16
    d = 32

    Q = torch.randn(batch, seq, d)
    K = torch.randn(batch, seq, d)

    Q_out, K_out = rope(Q.clone(), K.clone())

    assert Q_out.shape == (batch, seq, d), \
        f"Q_out shape {Q_out.shape} does not match expected {(batch, seq, d)}"

    assert K_out.shape == (batch, seq, d), \
        f"K_out shape {K_out.shape} does not match expected {(batch, seq, d)}"

def test_rope_has_no_parameters(rope):
    """RoPE should contain no learnable parameters (pure math)."""
    params = list(rope.parameters())

    assert len(params) == 0, \
        f"RoPE should have 0 parameters, found {len(params)}"

def test_rope_device_movement(rope):
    """RoPE should correctly operate on CUDA tensors when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    Q = torch.randn(2, 16, 32).cuda()
    K = torch.randn(2, 16, 32).cuda()

    Q_out, K_out = rope(Q.clone(), K.clone())

    assert Q_out.device == Q.device, \
        f"Q_out device {Q_out.device} does not match input device {Q.device}"

    assert K_out.device == K.device, \
        f"K_out device {K_out.device} does not match input device {K.device}"

def test_rope_rotation_correctness(rope):
    """RoPE should correctly rotate each (even, odd) pair according to cos/sin."""
    batch = 1
    seq = 4
    d = 32

    Q = torch.randn(batch, seq, d)
    K = torch.randn(batch, seq, d)

    Q_in = Q.clone()
    K_in = K.clone()

    Q_out, K_out = rope(Q, K)

    # recompute rotation manually for verification
    position = torch.arange(seq).unsqueeze(1).float()
    w = (1 / rope.rope_base) ** (torch.arange(0, d, 2).float() / d)
    angle = position * w

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    for pos in range(seq):
        for i in range(0, d, 2):
            idx = i // 2

            q0 = Q_in[0, pos, i]
            q1 = Q_in[0, pos, i+1]

            expected_q0 = q0 * cos[pos, idx] - q1 * sin[pos, idx]
            expected_q1 = q0 * sin[pos, idx] + q1 * cos[pos, idx]

            assert torch.allclose(Q_out[0, pos, i], expected_q0, atol=1e-5), \
                f"Incorrect Q rotation at pos={pos}, dim={i}"

            assert torch.allclose(Q_out[0, pos, i+1], expected_q1, atol=1e-5), \
                f"Incorrect Q rotation at pos={pos}, dim={i+1}"

            k0 = K_in[0, pos, i]
            k1 = K_in[0, pos, i+1]

            expected_k0 = k0 * cos[pos, idx] - k1 * sin[pos, idx]
            expected_k1 = k0 * sin[pos, idx] + k1 * cos[pos, idx]

            assert torch.allclose(K_out[0, pos, i], expected_k0, atol=1e-5), \
                f"Incorrect K rotation at pos={pos}, dim={i}"

            assert torch.allclose(K_out[0, pos, i+1], expected_k1, atol=1e-5), \
                f"Incorrect K rotation at pos={pos}, dim={i+1}"

def test_rope_qk_independence(rope):
    """RoPE should rotate Q and K independently and not mix their values."""
    Q = torch.randn(2, 8, 32)
    K = torch.randn(2, 8, 32)

    Q_out, K_out = rope(Q.clone(), K.clone())

    assert not torch.allclose(Q_out, K_out), \
        "Q_out and K_out should not be identical after RoPE rotation"
    

# ALiBi positional encoder tests
@pytest.fixture
def alibi():
    return ALiBi(8)

def test_alibi_output_shape(alibi):
    """Output shape should match input shape."""
    x = torch.randn(2, 8, 16, 16)

    y = alibi(x)

    assert y.shape == x.shape, \
        f"Expected output shape {x.shape}, got {y.shape}"
    
def test_alibi_same_bias_for_all_batches(alibi):
    """The same ALiBi bias should be applied to every batch."""
    x = torch.zeros(2, 8, 16, 16)

    y = alibi(x)

    assert torch.allclose(y[0], y[1]), \
        "ALiBi bias differs between batches."
    
def test_alibi_diagonal_is_zero(alibi):
    """Bias should be zero when query and key positions are equal."""
    x = torch.zeros(1, 8, 16, 16)

    y = alibi(x)

    diag = torch.diagonal(y[0], dim1=-2, dim2=-1)

    assert torch.allclose(diag, torch.zeros_like(diag)), \
        "Diagonal entries should have zero ALiBi bias."
    
def test_alibi_penalizes_distant_tokens(alibi):
    """More distant positions should receive a larger penalty."""
    x = torch.zeros(1, 8, 16, 16)

    y = alibi(x)

    head = 0

    assert y[0, head, 5, 5] > y[0, head, 5, 6], \
        "Distance 1 should be penalized more than distance 0."

    assert y[0, head, 5, 6] > y[0, head, 5, 10], \
        "Larger distances should receive larger penalties."
    
def test_alibi_heads_have_different_biases(alibi):
    """Different heads should use different slopes."""
    x = torch.zeros(1, 8, 16, 16)

    y = alibi(x)

    assert not torch.allclose(y[0, 0], y[0, 1]), \
        "Different attention heads should have different ALiBi biases."
    
