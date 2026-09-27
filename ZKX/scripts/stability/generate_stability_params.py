#!/usr/bin/env python3
"""Generate the immutable Stage-10 Task1/Task2 paired perturbation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MASTER_SEED = 0x01353A70
TOTAL_CASES = 1000
PARAMETER_SET_ID = "stage10_stability_formal_1000_v1"


class XorShift32:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state

    def unit(self) -> float:
        return self.next_u32() / 4294967295.0

    def symmetric(self, radius: float) -> float:
        return (2.0 * self.unit() - 1.0) * radius


def rounded(value: float) -> float:
    return round(value, 9)


def generate() -> dict[str, object]:
    rng = XorShift32(MASTER_SEED)
    cases: list[dict[str, object]] = []
    for index in range(TOTAL_CASES):
        seed = rng.next_u32()
        task1_noise_seed = rng.next_u32()
        task2_noise_seed = rng.next_u32()
        t1_delay = 171 + int(rng.next_u32() % 7) - 3
        t1_doppler = 8 + int(rng.next_u32() % 3) - 1
        t1_amplitude = rounded(1.0 + rng.symmetric(0.05))
        t1_noise = rounded(0.035 * (1.0 + rng.symmetric(0.05)))
        t2_delay = 3276 + int(rng.next_u32() % 65) - 32
        t2_target = rounded(0.80 * (1.0 + rng.symmetric(0.05)))
        t2_gaussian = rounded(0.16 * (1.0 + rng.symmetric(0.05)))
        t2_sawtooth = rounded(0.06 * (1.0 + rng.symmetric(0.05)))
        t2_square = rounded(0.06 * (1.0 + rng.symmetric(0.05)))
        t2_noise = rounded(0.04 * (1.0 + rng.symmetric(0.05)))
        t2_cutoff = rounded(90.0 * (1.0 + rng.symmetric(0.02)))
        t2_kalman_r = rounded(0.002 * (1.0 + rng.symmetric(0.05)))
        case_id = f"stage10_pair_{index:04d}_{seed:08x}"
        summary = (
            f"Task1(delay={t1_delay},doppler_bin={t1_doppler},amplitude={t1_amplitude},"
            f"noise_std={t1_noise});Task2(delay={t2_delay},target={t2_target},"
            f"gaussian={t2_gaussian},sawtooth={t2_sawtooth},square={t2_square},"
            f"noise_amplitude={t2_noise},cutoff_hz={t2_cutoff},kalman_R={t2_kalman_r})"
        )
        cases.append({
            "stability_case_index": index,
            "case_id": case_id,
            "seed": seed,
            "Task1": {
                "baseline_scale_id": "task1_s10_baseline",
                "target_delay_samples": t1_delay,
                "doppler_bin": t1_doppler,
                "target_amplitude": t1_amplitude,
                "noise_std": t1_noise,
                "noise_seed": task1_noise_seed,
            },
            "Task2": {
                "baseline_scale_id": "task2_s08_baseline",
                "target_delay_samples": t2_delay,
                "target_weight": t2_target,
                "gaussian_weight": t2_gaussian,
                "sawtooth_weight": t2_sawtooth,
                "square_weight": t2_square,
                "noise_amplitude": t2_noise,
                "noise_seed": task2_noise_seed,
                "filter_cutoff_hz": t2_cutoff,
                "kalman_R": t2_kalman_r,
            },
            "perturbation_summary": summary,
        })
    return {
        "schema_version": 1,
        "parameter_set_id": PARAMETER_SET_ID,
        "coverage": "formal_1000_paired",
        "master_seed": MASTER_SEED,
        "stability_total_cases": TOTAL_CASES,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = generate()
    if args.check:
        actual = json.loads(args.output.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("parameter file differs from deterministic generator output")
        print(f"[STABILITY][PARAMS][PASS] cases={TOTAL_CASES} file={args.output}")
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing parameter file: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[STABILITY][PARAMS][PASS] cases={TOTAL_CASES} file={args.output}")


if __name__ == "__main__":
    main()
