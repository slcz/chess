/*
 * chess_nn_eval.h — the real-net evaluator behind the PUCT eval seam.
 *
 * Embeds the Python interpreter (the engine binary owns the process; the
 * chess_ai.net.nn_worker module owns the model) and swaps a game's staging
 * buffers for the worker's pinned host tensors, so walkers fill them
 * directly and the IO thread's eval call is one H2D + forward + D2H.
 * All Python runs on the search's IO thread; the waker is suspended during
 * eval rounds. The interpreter is never finalized — process-lifetime.
 */
#ifndef CHESS_NN_EVAL_H
#define CHESS_NN_EVAL_H

#include "chess_puct.h"

/* Load the checkpoint (legacy Shaw configs pinned by Trainer.load), swap
 * g's staging buffers for the worker's pinned ones and set the eval fn.
 * 0 = ok. The staging becomes BORROWED: puct_game_free will not free it. */
int puct_attach_nn(PuctGame *g, const char *ckpt, const char *device);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_NN_EVAL_H */
