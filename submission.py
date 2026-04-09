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
def _block_sparse_attn_fwd_v2(
    Q, K, V, O, LSE,
    BLOCK_INDICES,
    DEGREES,
    SEQ_LENS,
    t_max,
    num_q_blocks,
    stride_bi_bh, stride_bi_qb, stride_bi_s,
    stride_deg_bh, stride_deg_qb,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_q_blocks
    qb = pid % num_q_blocks

    seq_len = tl.load(SEQ_LENS + bh)
    q_start = qb * BLOCK_SIZE

    offs = tl.arange(0, BLOCK_SIZE)

    bh_base = bh * t_max * BLOCK_SIZE

    o_ptrs = O + bh_base + (q_start + offs[:, None]) * BLOCK_SIZE + offs[None, :]
    lse_ptrs = LSE + bh * t_max + q_start + offs

    if q_start >= seq_len:
        tl.store(o_ptrs, tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.bfloat16))
        tl.store(lse_ptrs, tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32))
        return

    num_k = tl.load(DEGREES + bh * stride_deg_bh + qb * stride_deg_qb)
    if num_k == 0:
        tl.store(o_ptrs, tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.bfloat16))
        tl.store(lse_ptrs, tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32))
        return

    q_ptrs = Q + bh_base + (q_start + offs[:, None]) * BLOCK_SIZE + offs[None, :]
    q_mask = (q_start + offs[:, None]) < seq_len
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    bi_base = BLOCK_INDICES + bh * stride_bi_bh + qb * stride_bi_qb

    m_i = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    q_positions = q_start + offs

    for slot in range(MAX_BLOCKS):
        k_block_idx = tl.load(bi_base + slot * stride_bi_s)
        k_start = k_block_idx * BLOCK_SIZE
        k_positions = k_start + offs

        k_ptrs = K + bh_base + k_positions[:, None] * BLOCK_SIZE + offs[None, :]
        k_chunk = tl.load(k_ptrs, mask=k_positions[:, None] < seq_len, other=0.0)

        s = tl.dot(q_block, tl.trans(k_chunk)).to(tl.float32) * SCALE

        valid = (slot < num_k)
        seq_mask = (q_positions[:, None] < seq_len) & (k_positions[None, :] < seq_len)
        full_mask = valid & seq_mask

        is_diag = (k_block_idx == qb)
        if is_diag:
            causal = k_positions[None, :] <= q_positions[:, None]
            full_mask = full_mask & causal

        s = tl.where(full_mask, s, float('-inf'))

        block_max = tl.max(s, axis=1)
        new_m = tl.maximum(m_i, block_max)

        safe_old_m = tl.where(m_i > float('-inf'), m_i, new_m)
        alpha = tl.exp(safe_old_m - new_m)

        safe_new_m = tl.where(new_m > float('-inf'), new_m, 0.0)
        exp_s = tl.exp(s - safe_new_m[:, None])
        exp_s = tl.where(full_mask, exp_s, 0.0)

        v_ptrs = V + bh_base + k_positions[:, None] * BLOCK_SIZE + offs[None, :]
        v_chunk = tl.load(v_ptrs, mask=k_positions[:, None] < seq_len, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(exp_s.to(tl.bfloat16), v_chunk).to(tl.float32)
        l_i = l_i * alpha + tl.sum(exp_s, axis=1)
        m_i = tl.where(block_max > float('-inf'), new_m, m_i)

    valid_l = l_i > 0
    o_val = tl.where(valid_l[:, None], acc / l_i[:, None], 0.0)
    q_valid = (q_start + offs[:, None]) < seq_len
    o_val = tl.where(q_valid, o_val, 0.0)
    tl.store(o_ptrs, o_val.to(tl.bfloat16))

    lse_val = tl.where(valid_l, m_i + tl.log(l_i), float('-inf'))
    q_lse_valid = (q_start + offs) < seq_len
    lse_val = tl.where(q_lse_valid, lse_val, float('-inf'))
    tl.store(lse_ptrs, lse_val)


def _triton_block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    device = q.device
    batch_heads = batch_size * num_heads
    num_q_blocks = t_max // BLOCK_SIZE

    q_flat = q.reshape(batch_heads, t_max, head_dim).contiguous()
    k_flat = k.reshape(batch_heads, t_max, head_dim).contiguous()
    v_flat = v.reshape(batch_heads, t_max, head_dim).contiguous()

    row_ptr_flat = row_ptr.reshape(batch_heads, num_q_blocks + 1).to(torch.int64)
    col_idx_flat = col_idx.reshape(batch_heads, -1).to(torch.int64)

    row_starts = row_ptr_flat[:, :-1]
    degrees = (row_ptr_flat[:, 1:] - row_starts).to(torch.int32)
    max_degree = max(int(degrees.max().item()), 1)

    MAX_BLOCKS = 1
    while MAX_BLOCKS < max_degree:
        MAX_BLOCKS *= 2
    MAX_BLOCKS = min(MAX_BLOCKS, 32)

    slot_offsets = torch.arange(MAX_BLOCKS, device=device, dtype=torch.int64)
    ci_max_idx = col_idx_flat.shape[1] - 1
    gather_idx = (row_starts[:, :, None] + slot_offsets[None, None, :]).clamp(max=ci_max_idx)
    flat_gather = gather_idx.reshape(batch_heads, -1)
    block_indices = torch.gather(col_idx_flat, 1, flat_gather).reshape(batch_heads, num_q_blocks, MAX_BLOCKS).to(torch.int32)

    valid_mask = slot_offsets[None, None, :] < degrees[:, :, None].to(torch.int64)
    block_indices = torch.where(valid_mask, block_indices, torch.zeros_like(block_indices))

    seq_lens_bh = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).contiguous().to(torch.int32)

    o_flat = torch.empty_like(q_flat)
    lse_flat = torch.full((batch_heads, t_max), float('-inf'), device=device, dtype=torch.float32)

    grid = (batch_heads * num_q_blocks,)
    _block_sparse_attn_fwd_v2[grid](
        q_flat, k_flat, v_flat, o_flat, lse_flat,
        block_indices, degrees, seq_lens_bh,
        t_max, num_q_blocks,
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2),
        degrees.stride(0), degrees.stride(1),
        BLOCK_SIZE=BLOCK_SIZE, MAX_BLOCKS=MAX_BLOCKS, SCALE=SCALE,
        num_warps=8, num_stages=3,
    )

    return o_flat.reshape(batch_size, num_heads, t_max, head_dim), lse_flat.reshape(batch_size, num_heads, t_max)


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

    max_blocks_needed = set()
    seen_shapes = set()
    for spec in suite_specs:
        upper = spec.window_blocks + spec.global_blocks + spec.retrieval_blocks
        mb = 1
        while mb < upper:
            mb *= 2
        mb = min(mb, 32)
        max_blocks_needed.add(max(mb, 1))

        shape_key = (spec.batch_size, spec.num_heads, spec.t_max)
        seen_shapes.add(shape_key)

    for (b, h, t) in seen_shapes:
        for mb in max_blocks_needed:
            nqb = t // BLOCK_SIZE
            q = torch.randn(b, h, t, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            batch_heads = b * h
            actual_deg = min(mb, nqb)
            rp = torch.zeros(b, h, nqb + 1, dtype=torch.int32, device="cuda")
            for i in range(nqb):
                rp[:, :, i + 1] = rp[:, :, i] + actual_deg
            total_nnz = nqb * actual_deg
            ci = torch.zeros(b, h, total_nnz, dtype=torch.int32, device="cuda")
            for i in range(nqb):
                for d in range(actual_deg):
                    block_idx = max(0, i - d)
                    ci[:, :, i * actual_deg + d] = block_idx
            sl = torch.full((b,), t, dtype=torch.int32, device="cuda")

            block_sparse_attn_fwd(q, k, v, rp, ci, sl)
            torch.cuda.synchronize()
            del q, k, v, rp, ci, sl
            torch.cuda.empty_cache()
