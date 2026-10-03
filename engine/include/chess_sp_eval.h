/*
 * chess_sp_eval.h — the batch-evaluator contract of the parallel engine
 * (Phase 3.3b).
 *
 * The evaluator is the ONE synchronization point of the engine
 * (PARALLEL_DESIGN §1): worker threads park descents and seal request
 * buffers; once per epoch the engine assembles every sealed request into a
 * batch and calls exactly one evaluator. Two implementations exist:
 *
 *   - the embedded-CPython evaluator (src/sp_evaluator.c, F8): wraps a
 *     Python object (torch net or fixture) behind PyGILState_Ensure —
 *     workers never touch the interpreter;
 *   - plain C fixture evaluators in the unit tests (batch-invariant, so
 *     the whole engine is deterministic and parallel ≡ sequential can be
 *     asserted exactly).
 *
 * The batch is assembled by the ENGINE before the call: planes are the u8
 * quantized feature tensors (see SpReq in chess_search.h — dim 116 of each
 * token carries the RAW halfmove clock; un-quantize with
 * (float)clock/100.0f), slots are the legal policy slots in movegen
 * ENUMERATION order, concatenated across the batch with a cumulative
 * offsets array (request i's slots are slots[offsets[i]:offsets[i+1]]).
 * The evaluator fills the preallocated outputs: logp (f32) aligned with
 * the slots array — enumeration order, exactly what sp_deliver_node
 * expects — and the WDL triple (f32, pW/pD/pL from the parked position's
 * mover's view) per request. All multi-byte data is little-endian.
 */
#ifndef CHESS_SP_EVAL_H
#define CHESS_SP_EVAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpEvalBatch SpEvalBatch;   /* opaque; defined in sp_engine.c */

int sp_batch_size(const SpEvalBatch *b);          /* requests in the batch  */
int sp_batch_total_legal(const SpEvalBatch *b);   /* Σ n_legal              */

const uint8_t  *sp_batch_planes(const SpEvalBatch *b);  /* [size × 7488] u8 */
const uint16_t *sp_batch_slots(const SpEvalBatch *b);   /* [total_legal] u16 */
const int32_t  *sp_batch_offsets(const SpEvalBatch *b); /* [size + 1] i32   */
const int32_t  *sp_batch_mover(const SpEvalBatch *b);   /* [size] i32       */

/* Outputs — preallocated by the engine, filled by the evaluator. */
float *sp_batch_out_logp(SpEvalBatch *b);   /* [total_legal] f32, enum order */
float *sp_batch_out_wdl(SpEvalBatch *b);    /* [size × 3] f32 (pW, pD, pL)   */

#ifdef __cplusplus
}
#endif

#endif /* CHESS_SP_EVAL_H */
