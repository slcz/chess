/*
 * chess_nn_eval.h — the real-net evaluator behind the PUCT eval seam.
 *
 * Embeds the Python interpreter (the engine binary owns the process; the
 * chess_ai.net.nn_worker module owns the model) and swaps the global
 * landing zone's row buffers for the worker's pinned host tensors, so
 * walkers fill them directly and the io thread's eval call is one H2D +
 * forward + D2H. All Python runs on the landing zone's single io thread;
 * the waker is suspended during eval rounds. The interpreter is never
 * finalized — process-lifetime. One evaluator per system.
 */
#ifndef CHESS_NN_EVAL_H
#define CHESS_NN_EVAL_H

#include "chess_puct.h"

/* Load the checkpoint (legacy Shaw configs pinned by Trainer.load), size
 * the landing zone to `rows`, swap in the worker's pinned tensors and set
 * the eval fn. 0 = ok. Single-game callers pass their walker count. */
int puct_attach_nn(const char *ckpt, const char *device, int rows);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_NN_EVAL_H */
