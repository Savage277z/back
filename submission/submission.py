import math
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {"name": "default"},
]


@triton.jit
def _block_sparse_attn_fwd(
    Q, K, V, O, LSE,
    ROW_PTR, COL_IDX, SEQ_LENS,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lseb, stride_lseh, stride_lset,
    stride_rpb, stride_rph, stride_rpn,
    stride_cib, stride_cih, stride_cin,
    num_heads,
    num_q_blocks,
    max_nnz,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_q_blocks
    qb = pid % num_q_blocks
    batch_idx = bh // num_heads
    head_idx = bh % num_heads

    seq_len = tl.load(SEQ_LENS + batch_idx)
    q_start = qb * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_positions = q_start + offs_m

    o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + q_positions[:, None] * stride_ot + offs_d[None, :] * stride_od
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + q_positions * stride_lset

    if q_start >= seq_len:
        tl.store(o_ptrs, tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.bfloat16), mask=(offs_m[:, None] < BLOCK_M) & (offs_d[None, :] < BLOCK_D))
        tl.store(lse_ptrs, float('-inf') + tl.zeros([BLOCK_M], dtype=tl.float32))
        return

    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + q_positions[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = q_positions[:, None] < seq_len
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    rp_base = ROW_PTR + batch_idx * stride_rpb + head_idx * stride_rph
    row_start = tl.load(rp_base + qb * stride_rpn)
    row_end = tl.load(rp_base + (qb + 1) * stride_rpn)

    if row_start == row_end:
        tl.store(o_ptrs, tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.bfloat16), mask=(offs_m[:, None] < BLOCK_M) & (offs_d[None, :] < BLOCK_D))
        tl.store(lse_ptrs, float('-inf') + tl.zeros([BLOCK_M], dtype=tl.float32))
        return

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    ci_base = COL_IDX + batch_idx * stride_cib + head_idx * stride_cih

    for slot in range(row_start, row_end):
        k_block = tl.load(ci_base + slot * stride_cin)
        k_start = k_block * BLOCK_N
        k_positions = k_start + offs_n

        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + k_positions[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k_mask = k_positions[:, None] < seq_len
        k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q_block, tl.trans(k_chunk)).to(tl.float32) * SCALE

        k_valid = k_positions[None, :] < seq_len
        q_valid = q_positions[:, None] < seq_len
        mask = k_valid & q_valid

        is_diag = (k_block == qb)
        if is_diag:
            causal = k_positions[None, :] <= q_positions[:, None]
            mask = mask & causal

        s = tl.where(mask, s, float('-inf'))

        block_max = tl.max(s, axis=1)

        new_m = tl.maximum(m_i, block_max)

        safe_old_m = tl.where(m_i > float('-inf'), m_i, new_m)
        alpha = tl.exp(safe_old_m - new_m)

        safe_new_m = tl.where(new_m > float('-inf'), new_m, 0.0)
        exp_s = tl.exp(s - safe_new_m[:, None])
        exp_s = tl.where(s > float('-inf'), exp_s, 0.0)

        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + k_positions[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=k_positions[:, None] < seq_len, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(exp_s.to(tl.bfloat16), v_chunk).to(tl.float32)
        l_i = l_i * alpha + tl.sum(exp_s, axis=1)
        m_i = tl.where(block_max > float('-inf'), new_m, m_i)

    valid_l = l_i > 0
    o_val = tl.where(valid_l[:, None], acc / l_i[:, None], 0.0)

    q_out_valid = q_positions[:, None] < seq_len
    o_val = tl.where(q_out_valid, o_val, 0.0)
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=(offs_m[:, None] < BLOCK_M) & (offs_d[None, :] < BLOCK_D))

    lse_val = tl.where(valid_l, m_i + tl.log(l_i), float('-inf'))
    q_lse_valid = q_positions < seq_len
    lse_val = tl.where(q_lse_valid, lse_val, float('-inf'))
    tl.store(lse_ptrs, lse_val)


def _triton_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    num_q_blocks = t_max // BLOCK_SIZE
    max_nnz = col_idx.shape[-1]

    o = torch.empty_like(q)
    lse = torch.full((batch_size, num_heads, t_max), float('-inf'),
                     device=q.device, dtype=torch.float32)

    grid = (batch_size * num_heads * num_q_blocks,)
    _block_sparse_attn_fwd[grid](
        q, k, v, o, lse,
        row_ptr, col_idx, seq_lens,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        row_ptr.stride(0), row_ptr.stride(1), row_ptr.stride(2),
        col_idx.stride(0), col_idx.stride(1), col_idx.stride(2),
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        max_nnz=max_nnz,
        BLOCK_M=128, BLOCK_N=128, BLOCK_D=128,
        num_warps=4, num_stages=2,
    )
    return o, lse


def _pytorch_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    device = q.device
    batch_heads = batch_size * num_heads
    num_q_blocks = row_ptr.shape[-1] - 1
    num_k_blocks = t_max // BLOCK_SIZE

    q_f = q.to(torch.float32).reshape(batch_heads, t_max, head_dim)
    k_blocks = k.to(torch.float32).reshape(batch_heads, num_k_blocks, BLOCK_SIZE, head_dim)
    v_blocks = v.to(torch.float32).reshape(batch_heads, num_k_blocks, BLOCK_SIZE, head_dim)

    flat_k_blocks = k_blocks.reshape(batch_heads * num_k_blocks, BLOCK_SIZE, head_dim)
    flat_v_blocks = v_blocks.reshape(batch_heads * num_k_blocks, BLOCK_SIZE, head_dim)

    row_ptr_2d = row_ptr.reshape(batch_heads, num_q_blocks + 1).to(torch.int64)
    col_idx_2d = col_idx.reshape(batch_heads, -1).to(torch.int64)
    seq_lens_2d = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).to(torch.int64)

    output = torch.zeros((batch_heads, t_max, head_dim), device=device, dtype=torch.float32)
    lse = torch.full((batch_heads, t_max), -torch.inf, device=device, dtype=torch.float32)

    batch_head_block_base = (
        torch.arange(batch_heads, device=device, dtype=torch.int64)[:, None] * num_k_blocks
    )
    block_token_offsets = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)
    slot_offsets_cache = {}

    for q_block in range(num_q_blocks):
        q_start = q_block * BLOCK_SIZE
        q_end = q_start + BLOCK_SIZE
        q_chunk = q_f[:, q_start:q_end]

        row_start = row_ptr_2d[:, q_block]
        row_end = row_ptr_2d[:, q_block + 1]
        degrees = row_end - row_start
        max_degree = int(degrees.max().item())
        if max_degree <= 0:
            continue

        slot_offsets = slot_offsets_cache.get(max_degree)
        if slot_offsets is None:
            slot_offsets = torch.arange(max_degree, device=device, dtype=torch.int64)[None, :]
            slot_offsets_cache[max_degree] = slot_offsets

        slot_valid = slot_offsets < degrees[:, None]
        gather_positions = torch.clamp(row_start[:, None] + slot_offsets, max=col_idx_2d.shape[1] - 1)
        gathered_block_indices = torch.gather(col_idx_2d, 1, gather_positions)
        gathered_block_indices = torch.where(
            slot_valid, gathered_block_indices, torch.zeros_like(gathered_block_indices),
        )

        flat_block_indices = batch_head_block_base + gathered_block_indices
        gathered_k_blocks = flat_k_blocks.index_select(0, flat_block_indices.reshape(-1)).reshape(
            batch_heads, max_degree, BLOCK_SIZE, head_dim
        )
        gathered_v_blocks = flat_v_blocks.index_select(0, flat_block_indices.reshape(-1)).reshape(
            batch_heads, max_degree, BLOCK_SIZE, head_dim
        )

        key_positions = (
            gathered_block_indices[:, :, None] * BLOCK_SIZE + block_token_offsets[None, None, :]
        ).reshape(batch_heads, max_degree * BLOCK_SIZE)
        key_valid = (
            slot_valid[:, :, None] & (key_positions.reshape(batch_heads, max_degree, BLOCK_SIZE) < seq_lens_2d[:, None, None])
        ).reshape(batch_heads, max_degree * BLOCK_SIZE)
        diag_key = (
            (gathered_block_indices == q_block)[:, :, None]
            .expand(batch_heads, max_degree, BLOCK_SIZE)
            .reshape(batch_heads, max_degree * BLOCK_SIZE)
        )

        q_positions = q_start + block_token_offsets[None, :]
        query_valid = q_positions < seq_lens_2d[:, None]

        k_tokens = gathered_k_blocks.reshape(batch_heads, max_degree * BLOCK_SIZE, head_dim)
        v_tokens = gathered_v_blocks.reshape(batch_heads, max_degree * BLOCK_SIZE, head_dim)

        scores = torch.matmul(q_chunk, k_tokens.transpose(1, 2)) * SCALE

        mask = key_valid[:, None, :] & query_valid[:, :, None]
        causal_ok = key_positions[:, None, :] <= q_positions[:, :, None]
        mask = mask & ((~diag_key)[:, None, :] | causal_ok)

        scores = scores.masked_fill(~mask, -torch.inf)
        row_max = torch.max(scores, dim=-1).values
        valid_rows = query_valid & torch.isfinite(row_max)
        row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))

        exp_scores = torch.exp(scores - row_max_safe[:, :, None]) * mask.to(torch.float32)
        denom = exp_scores.sum(dim=-1)
        denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))

        out_block = torch.matmul(exp_scores, v_tokens) / denom_safe[:, :, None]
        lse_block = torch.where(
            valid_rows,
            row_max_safe + torch.log(denom_safe),
            torch.full_like(row_max_safe, -torch.inf),
        )

        output[:, q_start:q_end] = torch.where(
            valid_rows[:, :, None], out_block, output[:, q_start:q_end],
        )
        lse[:, q_start:q_end] = torch.where(
            valid_rows, lse_block, lse[:, q_start:q_end],
        )

    return output.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16), lse.reshape(
        batch_size, num_heads, t_max
    )


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    if q.device.type == "cuda":
        return _triton_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)
    else:
        return _pytorch_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)


def setup(suite_specs, device, variants):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None

    seen = set()
    for spec in suite_specs:
        key = (spec.batch_size, spec.num_heads, spec.t_max)
        if key in seen:
            continue
        seen.add(key)

        b, h, t = spec.batch_size, spec.num_heads, spec.t_max
        q = torch.randn(b, h, t, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        nqb = t // BLOCK_SIZE
        rp = torch.zeros(b, h, nqb + 1, dtype=torch.int32, device="cuda")
        for i in range(nqb):
            rp[:, :, i + 1] = rp[:, :, i] + 1
        ci = torch.zeros(b, h, nqb, dtype=torch.int32, device="cuda")
        for i in range(nqb):
            ci[:, :, i] = i
        sl = torch.full((b,), t, dtype=torch.int32, device="cuda")
        block_sparse_attn_fwd(q, k, v, rp, ci, sl)
        torch.cuda.synchronize()
        del q, k, v, rp, ci, sl
        torch.cuda.empty_cache()
