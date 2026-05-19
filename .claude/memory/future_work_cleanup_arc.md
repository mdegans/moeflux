# Future work — cleanup candidates (post-Phase-4)

Noted during the prefill-arc Phase 4 session (2026-05-19), for the
planned cleanup/organization arc (see `kernel_arc_session12_landed`
— "cleanup/organization arc next, stable surfaces only"). None of
these block anything; each is a small, safe, stable-surface change.

## bytemuck-friendly pool IO

`BufferPool::upload` / `download` / `upload_at` take `&[u8]` /
`&mut [u8]`. Callers with typed data (`&[f32]`, `&[i32]`) either
`bytemuck::cast_slice` at the call site (Phase 4 head/tail do this)
or hand-write `unsafe slice::from_raw_parts` (older code). Make the
trait methods generic `<T: bytemuck::Pod>` — backward-compatible,
since `&[u8]` still satisfies `T: Pod`. Touches the `BufferPool`
trait + the Metal and CPU pool impls. Removes the cast boilerplate
everywhere and kills a class of `unsafe`. Mike's idea, raised in
the Phase 4 design chat.

## from_raw_parts reinterpret sweep

Several `&[T]` ↔ `&[u8]` reinterprets remain — linear_attn_forward.rs
(~1941, ~2530, ~2536), mod.rs MLA path (~2382, ~2498), gpu_mla.rs
(~441). Bytemuck-replaceable once the generic pool IO above lands.
Distinct from the genuine `buf.contents() as *const T` raw-FFI reads
(metal.rs, snapshot/state.rs, gpu/mod.rs) — those are unavoidable
Metal-pointer `unsafe` and stay. The Phase 4 head/tail reinterprets
are already gone (replaced inline with `bytemuck::cast_slice`).

## gpu_norm.rs is misnamed

`backend/gpu/gpu_norm.rs` now holds `encode_residual_add_n_tokens_
into`, `encode_rope_n_tokens_into`, `encode_embed_gather_4bit_into`,
`encode_buffer_copy_f32` — none are norms. Rename to `gpu_encode.rs`
(or split norm vs misc encode helpers). Pure file/symbol move.
