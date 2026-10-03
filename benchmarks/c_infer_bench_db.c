/*
 * benchmarks/c_infer_bench_db.c — double-buffered (ping-pong) driver for
 * the "speed-of-light MCTS" experiment: the net always works on B
 * positions (slot 0/1 alternating), while the host fills the OTHER
 * slot's input and drains the finished slot's output.
 *
 * Host protocol per batch (slot = k & 1):
 *   1. wait_free(slot)   — block until the slot's previous D2H completed
 *   2. argmax(slot)      — deliver-side read (numpy, in the worker)
 *   3. fill slot input   — memcpy into pinned buffers via raw pointers
 *   4. enqueue(slot)     — H2D -> forward -> D2H, fully async
 * Steps 2-4 of batch k overlap the GPU's forward of batch k-1.
 *
 * Build:  gcc -O2 $(python3.10-config --cflags) benchmarks/c_infer_bench_db.c \
 *             $(python3.10-config --ldflags --embed) -o build/c_infer_bench_db
 * Run:    PYTHONPATH=.:/workspace/chess/benchmarks ./build/c_infer_bench_db \
 *             <batch_size> <iters> <compile 0|1>
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill_row(uint8_t *row, long seq) {
    static uint8_t tmpl[64 * 117 * 4];
    static int init = 0;
    if (!init) {
        float *f = (float *)tmpl;
        for (int i = 0; i < 64 * 117; i++)
            f[i] = 0.01f * (float)(i % 13);
        init = 1;
    }
    memcpy(row, tmpl, sizeof tmpl);
    ((float *)row)[0] = (float)(seq % 1000);
    ((float *)row)[1] = (float)(seq % 7);
}

static void fill_mask_row(uint8_t *row, long seq) {
    memset(row, 0, 4240);
    for (int i = 0; i < 35; i++)
        row[(i * 79 + (int)(seq % 79)) % 4240] = 1;
}

int main(int argc, char **argv) {
    long B     = argc > 1 ? atol(argv[1]) : 8192;
    long iters = argc > 2 ? atol(argv[2]) : 10;
    int  comp  = argc > 3 ? atoi(argv[3]) : 1;
    size_t row_bytes = (size_t)(64 * 117 * 4);

    Py_Initialize();
    PyRun_SimpleString(
        "import sys; sys.path[:0] = ['.', './benchmarks']");
    PyObject *mod = PyImport_ImportModule("infer_worker_db");
    if (!mod) { PyErr_Print(); return 1; }
    PyObject *r = PyObject_CallMethod(mod, "setup", "(li)", B, comp);
    if (!r) { PyErr_Print(); return 1; }

    uint8_t *x_buf[2], *mask_buf[2];
    for (int slot = 0; slot < 2; slot++) {
        char key[32];
        snprintf(key, sizeof key, "x%d_ptr", slot);
        x_buf[slot] = (uint8_t *)(intptr_t)
            PyLong_AsLongLong(PyDict_GetItemString(r, key));
        snprintf(key, sizeof key, "mask%d_ptr", slot);
        mask_buf[slot] = (uint8_t *)(intptr_t)
            PyLong_AsLongLong(PyDict_GetItemString(r, key));
    }

    /* warmup: one full cycle through both slots (compile + autotune) */
    for (int k = 0; k < 4; k++) {
        int slot = k & 1;
        for (long i = 0; i < B; i++)
            fill_row(x_buf[slot] + i * row_bytes, i);
        for (long i = 0; i < B; i++)
            fill_mask_row(mask_buf[slot] + i * 4240, i);
        PyObject *s = PyObject_CallMethod(mod, "enqueue", "(i)", slot);
        if (!s) { PyErr_Print(); return 1; }
        Py_DECREF(s);
    }
    PyObject *sy = PyObject_CallMethod(mod, "sync", NULL);
    Py_DECREF(sy);
    for (int slot = 0; slot < 2; slot++) {
        PyObject *s = PyObject_CallMethod(mod, "argmax", "(i)", slot);
        Py_DECREF(s);
    }

    double wait_s = 0, read_s = 0, fill_s = 0, enq_s = 0;
    double t_all0 = now_s();
    for (long k = 0; k < iters; k++) {
        int slot = k & 1;
        double t0 = now_s();
        PyObject *s = PyObject_CallMethod(mod, "wait_free", "(i)", slot);
        Py_DECREF(s);
        double t1 = now_s();
        s = PyObject_CallMethod(mod, "argmax", "(i)", slot);
        Py_DECREF(s);
        double t2 = now_s();
        for (long i = 0; i < B; i++) {
            long seq = k * B + i;
            fill_row(x_buf[slot] + i * row_bytes, seq);
            fill_mask_row(mask_buf[slot] + i * 4240, seq);
        }
        double t3 = now_s();
        s = PyObject_CallMethod(mod, "enqueue", "(i)", slot);
        Py_DECREF(s);
        double t4 = now_s();
        wait_s += t1 - t0; read_s += t2 - t1;
        fill_s += t3 - t2; enq_s += t4 - t3;
    }
    sy = PyObject_CallMethod(mod, "sync", NULL);
    Py_DECREF(sy);
    double wall = now_s() - t_all0;

    double pos = (double)B * iters;
    printf("DB B=%ld iters=%ld compile=%d\n", B, iters, comp);
    printf("  wait   %7.2f ms/batch   read/argmax %7.2f ms/batch\n",
           wait_s / iters * 1e3, read_s / iters * 1e3);
    printf("  fill   %7.2f ms/batch   enqueue     %7.2f ms/batch\n",
           fill_s / iters * 1e3, enq_s / iters * 1e3);
    printf("  host total %7.2f ms/batch   wall %7.2f ms/batch\n",
           (wait_s + read_s + fill_s + enq_s) / iters * 1e3,
           wall / iters * 1e3);
    printf("  => positions/s %10.0f   tokens/s %10.0f\n",
           pos / wall, pos * 64 / wall);
    Py_Finalize();
    return 0;
}
