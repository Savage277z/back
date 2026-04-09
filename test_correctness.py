import sys
import os
import math

sys.path.insert(0, "/home/attention-kernel-challenge")
sys.path.insert(0, "/home/back/submission")

import torch
import numpy as np

from attention_kernel_challenge.cases import materialize_case, build_suite
from attention_kernel_challenge.reference import reference_block_sparse_attn_fwd
from attention_kernel_challenge.validation import validate_outputs
from attention_kernel_challenge.spec import Tolerances

from submission import block_sparse_attn_fwd


def run_test(case_name, spec, tolerances=None):
    if tolerances is None:
        tolerances = Tolerances()

    mc = materialize_case(spec, device="cpu")
    q, k, v = mc.q, mc.k, mc.v
    row_ptr, col_idx, seq_lens = mc.row_ptr, mc.col_idx, mc.seq_lens

    ref_o, ref_lse = reference_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)
    sub_o, sub_lse = block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)

    result = validate_outputs(sub_o, sub_lse, ref_o, ref_lse, tolerances)

    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {case_name}: output_diff={result.output_max_abs_diff:.6g}, lse_diff={result.lse_max_abs_diff:.6g}")
    if not result.passed:
        print(f"    {result.message}")

        finite_ref_o = torch.isfinite(ref_o.float())
        finite_sub_o = torch.isfinite(sub_o.float())
        if not torch.all(finite_ref_o == finite_sub_o):
            ref_inf_count = (~finite_ref_o).sum().item()
            sub_inf_count = (~finite_sub_o).sum().item()
            print(f"    Output non-finite count: ref={ref_inf_count}, sub={sub_inf_count}")

        finite_ref_lse = torch.isfinite(ref_lse)
        finite_sub_lse = torch.isfinite(sub_lse)
        if not torch.all(finite_ref_lse == finite_sub_lse):
            ref_inf_count = (~finite_ref_lse).sum().item()
            sub_inf_count = (~finite_sub_lse).sum().item()
            print(f"    LSE non-finite count: ref={ref_inf_count}, sub={sub_inf_count}")

    return result.passed


def main():
    print("=== Block-Sparse Attention Correctness Tests ===\n")

    all_passed = True

    print("--- Smoke Suite ---")
    smoke_suite = build_suite("smoke")
    for spec in smoke_suite:
        passed = run_test(spec.case_id, spec)
        all_passed = all_passed and passed

    print("\n--- Local-Dev Suite ---")
    local_dev_suite = build_suite("local-dev")
    for spec in local_dev_suite:
        passed = run_test(spec.case_id, spec)
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
