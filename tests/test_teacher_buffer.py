"""Dynamic-correctness tests for TeacherBuffer (task 1.8).

These tests verify the VRAM-safety invariants of the AMS teacher snapshot
buffer, which is the centerpiece of the project's response to the project spec
gotcha #9 (only one teacher in VRAM at a time). Specifically:

1. Snapshots are stored on CPU.
2. Appending then mutating the source model does not corrupt the snapshot.
3. The iterator yields a module in ``eval()`` mode with ``requires_grad=False``.
4. ``len(buffer)`` matches snapshot count and the iterator yields that many items.
5. The iterator yields the SAME ``nn.Module`` object across iterations.
6. GPU memory does not grow with the number of stored teachers.
7. ``state_dict()`` / ``load_state_dict()`` round-trip preserves teacher outputs.
8. ``state_dict()`` returns an independent (deepcopied) view.

Tester note: the spec mentions a ``swap_into(model)`` API in a generic tester
template, but this buffer's actual API uses ``__iter__`` instead (the
equivalent operation). Tests are written against ``__iter__``.
"""

import copy

import pytest
import torch
import torch.nn as nn

from src.models import preact_resnet18
from src.training.teacher_buffer import TeacherBuffer


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_factory():
    """A small MLP factory for fast tests where the architecture doesn't matter."""

    def make():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    return make


def test_state_dicts_on_cpu():
    """Every tensor in every stored snapshot must live on CPU (gotcha #9)."""
    torch.manual_seed(0)
    device = _device()
    buf = TeacherBuffer(_tiny_factory(), device=device)

    # Put two snapshots into the buffer from a CUDA-resident model (if available),
    # to make the test meaningful — the source is on GPU, the snapshots must
    # nevertheless be on CPU.
    m = _tiny_factory()().to(device)
    buf.append(m)
    # mutate and re-snapshot to get a second entry that is materially different.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)
    buf.append(m)

    assert len(buf._snapshots) == 2, "expected 2 snapshots after two appends"
    for i, snap in enumerate(buf._snapshots):
        for k, v in snap.items():
            assert v.device.type == "cpu", (
                f"snapshot[{i}] tensor '{k}' is on {v.device.type}, expected cpu — "
                f"CPU-storage invariant (the project spec gotcha #9) is broken"
            )


def test_swap_roundtrip():
    """Snapshot, mutate the source, iterate — outputs must match ORIGINAL weights.

    This proves that ``append`` performs an honest copy via ``to('cpu', copy=True)``
    and does not store an aliasing view on the source model's tensors. If the
    snapshot aliased the source, in-place mutation of the source after append
    would propagate into the snapshot and the iterator would yield mutated
    outputs.
    """
    torch.manual_seed(0)
    device = _device()

    # Use the tiny factory for speed; the invariant under test is general.
    factory = _tiny_factory()
    model_a = factory().to(device).eval()

    # Fixed input for output comparison.
    x = torch.rand(4, 3, 8, 8, device=device)

    # Reference output of model_A at the moment of snapshotting.
    with torch.no_grad():
        y_ref = model_a(x).clone()

    buf = TeacherBuffer(factory, device=device)
    buf.append(model_a)

    # Now MUTATE model_a so any aliasing snapshot would diverge from y_ref.
    with torch.no_grad():
        for p in model_a.parameters():
            p.add_(0.5)

    # Sanity: the mutated model should produce different outputs now.
    with torch.no_grad():
        y_mutated = model_a(x)
    assert not torch.allclose(y_ref, y_mutated, atol=1e-4), (
        "test bug: parameter mutation didn't change outputs — pick a bigger delta"
    )

    # Iterate the buffer; the yielded scratch model must reproduce y_ref.
    teachers = list(buf)
    assert len(teachers) == 1
    with torch.no_grad():
        y_snap = teachers[0](x)

    assert torch.allclose(y_snap, y_ref, atol=1e-5), (
        "buffer snapshot output diverged from the original model_a output at "
        "snapshot time — append() may be aliasing the source state_dict instead "
        "of copying it"
    )


def test_iter_yields_eval_mode():
    """After the iterator yields a teacher, the module must be in eval() mode."""
    torch.manual_seed(0)
    device = _device()
    buf = TeacherBuffer(_tiny_factory(), device=device)
    buf.append(_tiny_factory()().to(device))

    for teacher in buf:
        assert teacher.training is False, (
            "iterator yielded a teacher with training=True — would let BN "
            "statistics drift (the project spec gotcha #4)"
        )


def test_iter_yields_no_grad_params():
    """Every parameter of the yielded teacher must have requires_grad=False."""
    torch.manual_seed(0)
    device = _device()
    buf = TeacherBuffer(_tiny_factory(), device=device)
    buf.append(_tiny_factory()().to(device))

    for teacher in buf:
        for name, p in teacher.named_parameters():
            assert p.requires_grad is False, (
                f"yielded teacher has parameter '{name}' with requires_grad=True"
                " — gradient could flow back through the teacher (gotcha #4)"
            )


def test_iter_count_matches_len():
    """With N appends, len(buf) == N and iteration produces exactly N items."""
    torch.manual_seed(0)
    device = _device()
    buf = TeacherBuffer(_tiny_factory(), device=device)

    N = 3
    for _ in range(N):
        buf.append(_tiny_factory()().to(device))

    assert len(buf) == N, f"len(buf) = {len(buf)}, expected {N}"

    count = sum(1 for _ in buf)
    assert count == N, f"iteration produced {count} items, expected {N}"


def test_iter_yields_same_object():
    """All iterations yield the same nn.Module instance (the scratch model).

    Confirms the VRAM invariant: there is exactly one teacher module in GPU
    memory ever, regardless of how many snapshots are stored.
    """
    torch.manual_seed(0)
    device = _device()
    buf = TeacherBuffer(_tiny_factory(), device=device)
    for _ in range(3):
        buf.append(_tiny_factory()().to(device))

    ids_run_1 = [id(t) for t in buf]
    ids_run_2 = [id(t) for t in buf]

    # All ids — within a run AND across runs — must be identical.
    all_ids = ids_run_1 + ids_run_2
    assert len(set(all_ids)) == 1, (
        f"iterator yielded distinct module objects (ids={all_ids}) — the "
        f"scratch-model invariant is broken; VRAM will grow with the buffer"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory test requires CUDA")
def test_buffer_does_not_grow_gpu_memory():
    """Appending more snapshots must NOT proportionally grow GPU memory.

    With one scratch model in VRAM and snapshots on CPU, the GPU footprint
    should be roughly constant in the number of appended teachers. We allow
    up to 50 MB of slack between "after 1 append" and "after 4 appends" to
    tolerate cuDNN workspace and CUDA caching allocator behavior.
    """
    device = torch.device("cuda")

    # Build a buffer with the actual PreActResNet-18 factory used in the project.
    buf = TeacherBuffer(lambda: preact_resnet18(num_classes=10), device=device)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    def append_one():
        # Construct a fresh PreActResNet-18 on CPU then move to GPU, append,
        # then explicitly drop the local reference + empty_cache so the only
        # GPU memory the buffer should be "responsible" for is the scratch
        # model.
        m = preact_resnet18(num_classes=10).to(device)
        buf.append(m)
        del m
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    append_one()
    mem_after_1 = torch.cuda.memory_allocated()

    for _ in range(3):
        append_one()
    mem_after_4 = torch.cuda.memory_allocated()

    # Primary check: don't grow by more than ~50 MB between 1 and 4 appends.
    delta = mem_after_4 - mem_after_1
    assert delta < 50 * 2**20, (
        f"GPU memory grew by {delta / 2**20:.1f} MB between 1 and 4 snapshots "
        f"(baseline={baseline / 2**20:.1f} MB, after_1={mem_after_1 / 2**20:.1f} MB, "
        f"after_4={mem_after_4 / 2**20:.1f} MB) — snapshots are leaking onto the GPU"
    )

    # Secondary, looser check: absolute footprint above baseline is bounded.
    # One PreActResNet-18 in fp32 is ~45 MB of params + small overhead; with
    # cuDNN workspace we allow up to 300 MB.
    abs_footprint = mem_after_4 - baseline
    assert abs_footprint < 300 * 2**20, (
        f"buffer's total GPU footprint is {abs_footprint / 2**20:.1f} MB above "
        f"baseline — well over what a single scratch PreActResNet-18 should cost"
    )


def test_state_dict_roundtrip():
    """state_dict / load_state_dict round-trip preserves teacher outputs."""
    torch.manual_seed(0)
    device = _device()
    factory = _tiny_factory()

    # Original buffer with 2 distinct snapshots.
    buf = TeacherBuffer(factory, device=device)
    m1 = factory().to(device)
    buf.append(m1)
    m2 = factory().to(device)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.3)  # make m2 materially different from m1
    buf.append(m2)

    x = torch.rand(4, 3, 8, 8, device=device)
    with torch.no_grad():
        orig_outputs = [t(x).clone() for t in buf]
    assert len(orig_outputs) == 2

    # Save, build a fresh buffer, load.
    sd = buf.state_dict()
    fresh = TeacherBuffer(factory, device=device)
    fresh.load_state_dict(sd)

    assert len(fresh) == 2, f"restored buffer has len={len(fresh)}, expected 2"
    with torch.no_grad():
        fresh_outputs = [t(x).clone() for t in fresh]

    for i, (yo, yf) in enumerate(zip(orig_outputs, fresh_outputs)):
        assert torch.allclose(yo, yf, atol=1e-5), (
            f"restored teacher {i} produced different output than original — "
            f"state_dict round-trip is lossy"
        )


def test_state_dict_is_deepcopied():
    """state_dict() returns an independent snapshot list; further appends do
    not appear in a previously-returned dict."""
    torch.manual_seed(0)
    device = _device()
    factory = _tiny_factory()
    buf = TeacherBuffer(factory, device=device)

    buf.append(factory().to(device))
    buf.append(factory().to(device))

    sd = buf.state_dict()
    assert isinstance(sd, dict) and "snapshots" in sd
    snapshots_at_call = len(sd["snapshots"])
    assert snapshots_at_call == 2, (
        f"state_dict() at the time of call returned {snapshots_at_call} "
        f"snapshots, expected 2"
    )

    # Now append a 3rd model. The previously returned dict MUST be unaffected.
    buf.append(factory().to(device))
    assert len(buf) == 3, "internal _snapshots did not grow after append"
    assert len(sd["snapshots"]) == 2, (
        f"previously returned state_dict mutated to len={len(sd['snapshots'])} "
        f"after a later append — state_dict() aliased internal storage instead "
        f"of deepcopying"
    )

    # Additionally, mutating a tensor inside the returned dict must not affect
    # the buffer's stored snapshots (independence both ways).
    first_key = next(iter(sd["snapshots"][0].keys()))
    orig_buf_tensor = buf._snapshots[0][first_key].clone()
    sd["snapshots"][0][first_key].add_(99.0)
    assert torch.allclose(buf._snapshots[0][first_key], orig_buf_tensor), (
        "mutating the returned state_dict's tensor changed the buffer's "
        "internal snapshot — state_dict() returned aliasing tensors"
    )
