/* nn_eval.c — embedded-Python evaluator (see chess_nn_eval.h). */
#include "chess_nn_eval.h"

#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject *worker;
} NnCtx;

static void nn_eval(void *ctx, int n, const float *planes,
                    const uint8_t *masks, float *logp, float *wdl) {
    (void)planes;
    (void)masks;
    (void)logp;
    (void)wdl;
    NnCtx *c = ctx;
    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *r = PyObject_CallMethod(c->worker, "forward", "(i)", n);
    if (!r)
        PyErr_Print();
    Py_XDECREF(r);
    PyGILState_Release(st);
}

int puct_attach_nn(PuctGame *g, const char *ckpt, const char *device) {
    static int initialized = 0;
    if (!initialized) {
        Py_Initialize();
        PyRun_SimpleString(
            "import sys\n"
            "sys.path.insert(0, '" PUCT_REPO_ROOT "')\n"
            "sys.path.insert(0, '" PUCT_PURELIB "')\n");
        PyEval_SaveThread();       /* the interpreter is now up for grabs */
        initialized = 1;
    }

    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *mod = PyImport_ImportModule("chess_ai.net.nn_worker");
    if (!mod) {
        PyErr_Print();
        PyGILState_Release(st);
        return -1;
    }
    PyObject *res =
        PyObject_CallMethod(mod, "setup", "(ssi)", ckpt, device,
                            (int)PUCT_MAX_WALKERS);
    if (!res) {
        PyErr_Print();
        Py_DECREF(mod);
        PyGILState_Release(st);
        return -1;
    }
    PyObject *pv = PyDict_GetItemString(res, "planes");
    PyObject *mv = PyDict_GetItemString(res, "mask");
    PyObject *lv = PyDict_GetItemString(res, "logp");
    PyObject *wv = PyDict_GetItemString(res, "wdl");
    if (!pv || !mv || !lv || !wv) {
        Py_DECREF(res);
        Py_DECREF(mod);
        PyGILState_Release(st);
        return -1;
    }
    NnCtx *c = malloc(sizeof *c);
    c->worker = mod;

    puct_set_staging(g,
                     (float *)PyLong_AsUnsignedLongLong(pv),
                     (uint8_t *)PyLong_AsUnsignedLongLong(mv),
                     (float *)PyLong_AsUnsignedLongLong(lv),
                     (float *)PyLong_AsUnsignedLongLong(wv));
    puct_set_eval(g, nn_eval, c);
    Py_DECREF(res);
    PyGILState_Release(st);
    return 0;
}
