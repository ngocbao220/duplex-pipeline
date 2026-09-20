from core.resource_tuning import GpuSnapshot, choose_workers_per_gpu


def test_auto_tuning_uses_peak_memory_with_a_conservative_reserve():
    plan = choose_workers_per_gpu(
        GpuSnapshot(index="0", total_memory_mib=40_000, free_memory_mib=38_000, peak_memory_used_mib=6_000),
        cpu_workers_per_gpu=12,
        reserve_ratio=0.10,
        max_workers_per_gpu=8,
    )

    assert plan == 6


def test_auto_tuning_never_returns_less_than_one_worker():
    plan = choose_workers_per_gpu(
        GpuSnapshot(index="0", total_memory_mib=24_000, free_memory_mib=20_000, peak_memory_used_mib=30_000),
        cpu_workers_per_gpu=8,
        reserve_ratio=0.10,
        max_workers_per_gpu=8,
    )

    assert plan == 1


def test_auto_tuning_obeys_cpu_and_explicit_worker_caps():
    plan = choose_workers_per_gpu(
        GpuSnapshot(index="0", total_memory_mib=80_000, free_memory_mib=79_000, peak_memory_used_mib=4_000),
        cpu_workers_per_gpu=3,
        reserve_ratio=0.10,
        max_workers_per_gpu=2,
    )

    assert plan == 2
