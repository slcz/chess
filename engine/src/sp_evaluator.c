/*
 * sp_evaluator.c — the embedded-CPython batch evaluator (PARALLEL_DESIGN
 * F8, decided round 4): only this file (and thread) touches the Python
 * interpreter; workers are pure C and never hold the GIL. The GIL is held
 * only around the actual `eval_batch` call and its setup — the batch
 * assembly into engine staging and the result delivery are pure C.
 *
 * Contract with the Python side (chess_ai/selfplay/sp_evaluator.py):
 *   make_evaluator(spec) -> object with eval_batch(planes, slots, offsets,
 *       mover, out_logp, out_wdl)
 * where each argument is a read-only (outputs: writable) memoryview over
 * the engine's staging buffers — zero-copy on the boundary; the Python
 * side wraps them with np.frombuffer and writes the outputs in place.
 *
 * The interpreter must already be live (the library is built for loading
 * *into* a Python process — cffi dlopen from chess_ai.selfplay.spengine).
 * A future standalone binary would call Py_Initialize() in main() first.
 *
 * This file is compiled only into libchess_sp (the Python-hosted build);
 * the pure-C test binaries inject fixture evaluators instead, so `make
 * check` has no Python dependency.
 */
#include <Python.h>
#include <stdlib.h>
#include <string.h>

#include "chess_sp.h"
#include "chess_sp_eval.h"

typedef struct {
    char       *module;   /* our own copies — the caller's strings may not */
    char       *spec;     /* outlive the engine (cffi buffers are GC'd)    */
    PyObject   *evaluator;   /* the object with eval_batch() */
} PyEvalCtx;

static int py_eval_init(void *ctx_) {
    PyEvalCtx *c = ctx_;
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *mod = PyImport_ImportModule(c->module);
    if (!mod) {
        PyErr_Print();
        PyGILState_Release(gil);
        return -1;
    }
    PyObject *obj = PyObject_CallMethod(mod, "make_evaluator", "(s)", c->spec);
    Py_DECREF(mod);
    if (!obj) {
        PyErr_Print();
        PyGILState_Release(gil);
        return -1;
    }
    c->evaluator = obj;
    PyGILState_Release(gil);
    return 0;
}

static int py_eval_batch(void *ctx_, SpEvalBatch *b) {
    PyEvalCtx *c = ctx_;
    PyGILState_STATE gil = PyGILState_Ensure();

    PyObject *mv_planes = PyMemoryView_FromMemory(
        (char *)sp_batch_planes(b), (Py_ssize_t)sp_batch_size(b) * 7488,
        PyBUF_READ);
    PyObject *mv_slots = PyMemoryView_FromMemory(
        (char *)sp_batch_slots(b),
        (Py_ssize_t)sp_batch_total_legal(b) * 2, PyBUF_READ);
    PyObject *mv_offsets = PyMemoryView_FromMemory(
        (char *)sp_batch_offsets(b),
        ((Py_ssize_t)sp_batch_size(b) + 1) * 4, PyBUF_READ);
    PyObject *mv_mover = PyMemoryView_FromMemory(
        (char *)sp_batch_mover(b), (Py_ssize_t)sp_batch_size(b) * 4,
        PyBUF_READ);
    PyObject *mv_logp = PyMemoryView_FromMemory(
        (char *)sp_batch_out_logp(b),
        (Py_ssize_t)sp_batch_total_legal(b) * 4, PyBUF_WRITE);
    PyObject *mv_wdl = PyMemoryView_FromMemory(
        (char *)sp_batch_out_wdl(b), (Py_ssize_t)sp_batch_size(b) * 3 * 4,
        PyBUF_WRITE);
    PyObject *result = NULL;
    int rc = -1;
    if (mv_planes && mv_slots && mv_offsets && mv_mover && mv_logp && mv_wdl) {
        result = PyObject_CallMethod(c->evaluator, "eval_batch", "(OOOOOO)",
                                     mv_planes, mv_slots, mv_offsets, mv_mover,
                                     mv_logp, mv_wdl);
        rc = result ? 0 : -1;
        if (!result)
            PyErr_Print();
    } else {
        PyErr_Print();
    }
    Py_XDECREF(result);
    Py_XDECREF(mv_planes);
    Py_XDECREF(mv_slots);
    Py_XDECREF(mv_offsets);
    Py_XDECREF(mv_mover);
    Py_XDECREF(mv_logp);
    Py_XDECREF(mv_wdl);
    PyGILState_Release(gil);
    return rc;
}

static void py_eval_free(void *ctx_) {
    PyEvalCtx *c = ctx_;
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_CLEAR(c->evaluator);
    PyGILState_Release(gil);
    free(c->module);
    free(c->spec);
    free(c);
}

int sp_py_evaluator_make(const char *module, const char *spec,
                         SpEvaluator *ev) {
    PyEvalCtx *c = malloc(sizeof *c);
    if (!c) return -1;
    c->module = strdup(module);
    c->spec = strdup(spec);
    if (!c->module || !c->spec) {
        free(c->module);
        free(c->spec);
        free(c);
        return -1;
    }
    c->evaluator = NULL;
    ev->ctx = c;
    ev->init = py_eval_init;
    ev->batch = py_eval_batch;
    ev->free_fn = py_eval_free;
    return 0;
}
