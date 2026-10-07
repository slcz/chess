/* nn_eval.c — embedded-Python evaluator (see chess_nn_eval.h). */
#include "chess_nn_eval.h"

#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject *worker;
} NnCtx;

static void nn_eval(void *ctx, PuctEvalOp op, int bank, int n,
                    const float *planes, const uint8_t *masks, float *logp,
                    float *wdl) {
    (void)planes;
    (void)masks;
    (void)logp;
    (void)wdl;
    NnCtx *c = ctx;
    const char *meth = op == PUCT_EVAL_PUSH ? "push"
                     : op == PUCT_EVAL_PULL ? "pull" : "wait_ready";
    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *r = PyObject_CallMethod(c->worker, (char *)meth, "(ii)",
                                      bank, n);
    if (!r)
        PyErr_Print();
    Py_XDECREF(r);
    PyGILState_Release(st);
}

static int python_init(void) {
    static int initialized = 0;
    if (initialized)
        return 0;
    Py_Initialize();
    PyRun_SimpleString(
        "import sys\n"
        "sys.path.insert(0, '" PUCT_REPO_ROOT "')\n"
        "sys.path.insert(0, '" PUCT_PURELIB "')\n");
    PyEval_SaveThread();       /* the interpreter is now up for grabs */
    initialized = 1;
    return 0;
}

/* One evaluator per system: size the landing zone (rows per bank), swap in
 * the worker's pinned host tensors — one set per bank — and set the eval
 * fn. */
int puct_attach_nn(const char *ckpt, const char *device, int rows) {
    python_init();
    puct_zone_init(rows);

    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *mod = PyImport_ImportModule("chess_ai.net.nn_worker");
    if (!mod) {
        PyErr_Print();
        PyGILState_Release(st);
        return -1;
    }
    PyObject *res =
        PyObject_CallMethod(mod, "setup", "(siii)", ckpt, device, rows,
                            PUCT_BANKS);
    if (!res) {
        PyErr_Print();
        Py_DECREF(mod);
        PyGILState_Release(st);
        return -1;
    }
    for (int b = 0; b < PUCT_BANKS; b++) {
        PyObject *pv = PyDict_GetItemString(res, "planes");
        PyObject *mv = PyDict_GetItemString(res, "mask");
        PyObject *lv = PyDict_GetItemString(res, "logp");
        PyObject *wv = PyDict_GetItemString(res, "wdl");
        if (!pv || !mv || !lv || !wv ||
            !PyList_Check(pv) || PyList_Size(pv) != PUCT_BANKS) {
            Py_DECREF(res);
            Py_DECREF(mod);
            PyGILState_Release(st);
            return -1;
        }
        puct_zone_set_buffers(b,
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(pv, b)),
                              (uint8_t *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(mv, b)),
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(lv, b)),
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(wv, b)));
    }
    NnCtx *c = malloc(sizeof *c);
    c->worker = mod;
    puct_set_eval(nn_eval, c);
    Py_DECREF(res);
    PyGILState_Release(st);
    return 0;
}
