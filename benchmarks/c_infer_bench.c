/*
 * benchmarks/c_infer_bench.c — the "speed-of-light MCTS" producer
 * (user experiment 2026-10-03): a C program that plays the role of the
 * search with ZERO search cost. Each iteration it
 *
 *   1. FILLS   the pinned staging buffers (planes Bx64x117 f32 + legal
 *              mask Bx4240 u8) with template memcpys — one row template,
 *              per-row tweak so no two rows are identical (defeats the
 *              cache serving stale data);
 *   2. CALLS   PyTorch through the embedded interpreter (one step() =
 *              H2D copies + one forward under inference_mode+bf16);
 *   3. CONSUMEs (optional) the deliver half: policy Bx4240 + WDL copied
 *              back to the CPU, as the C search would read per request.
 *
 * It reports the per-phase costs (fill / call / consume per batch) and
 * the pipelined wall-clock throughput (positions/s and tokens/s) — the
 * ceiling a real MCTS could reach if selection/expansion cost nothing.
 *
 * Architecture = the production sp engine's (embedded CPython, single
 * interpreter, GIL owned by the driver): Py_Initialize -> import
 * infer_worker -> setup() hands back HOST pointers of the pinned
 * buffers -> C loops -> final sync bounds the wall clock.
 *
 * Build (Linux node):
 *   gcc -O2 $(python3.10-config --cflags) benchmarks/c_infer_bench.c \
 *       $(python3.10-config --ldflags --embed) -o build/c_infer_bench
 * Run (repo root, on the GPU box):
 *   ./build/c_infer_bench <batch_size> <iters> <compile 0|1> <consume 0|1>
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

/* one row of synthetic planes: 64 tokens x 117 f32 — the encoder output a
 * real search would have just written for one position */
static void fill_row(uint8_t *row, long seq) {
    static uint8_t tmpl[64 * 117 * 4];
    static int init = 0;
    if (!init) {
        float *f = (float *)tmpl;
        for (int i = 0; i < 64 * 117; i++)
            f[i] = 0.01f * (float)(i % 13);   /* deterministic pattern */
        init = 1;
    }
    memcpy(row, tmpl, sizeof tmpl);
    ((float *)row)[0] = (float)(seq % 1000);  /* per-row identity */
    ((float *)row)[1] = (float)(seq % 7);     /* second tweak */
}

/* one row of the legal mask: ~35 legal slots of 4240 (midgame mobility) */
static void fill_mask_row(uint8_t *row, long seq) {
    memset(row, 0, 4240);
    for (int i = 0; i < 35; i++)
        row[(i * 79 + (int)(seq % 79)) % 4240] = 1;
}

int main(int argc, char **argv) {
    long B      = argc > 1 ? atol(argv[1]) : 4096;
    long iters  = argc > 2 ? atol(argv[2]) : 30;
    int  comp   = argc > 3 ? atoi(argv[3]) : 1;
    int  consume= argc > 4 ? atoi(argv[4]) : 1;
    long warmup = argc > 5 ? atol(argv[5]) : 5;

    Py_Initialize();
    PyRun_SimpleString(
        "import sys; sys.path[:0] = ['.', './benchmarks']");
    PyObject *mod = PyImport_ImportModule("infer_worker");
    if (!mod) { PyErr_Print(); return 1; }
    PyObject *r = PyObject_CallMethod(mod, "setup", "(lsi)", B, "cuda", comp);
    if (!r) { PyErr_Print(); return 1; }
    PyObject *px  = PyDict_GetItemString(r, "x_ptr");
    PyObject *pm  = PyDict_GetItemString(r, "mask_ptr");
    PyObject *pxb = PyDict_GetItemString(r, "x_bytes");
    uint8_t *x_buf    = (uint8_t *)(intptr_t)PyLong_AsLongLong(px);
    uint8_t *mask_buf = (uint8_t *)(intptr_t)PyLong_AsLongLong(pm);
    size_t   row_bytes = (size_t)(64 * 117 * 4);
    size_t   x_bytes   = (size_t)PyLong_AsLongLong(pxb);
    (void)x_bytes;

    /* warmup: compile + kernel autotune live here */
    for (long i = 0; i < warmup; i++) {
        PyObject *s = PyObject_CallMethod(mod, "step", "(ii)", (int)B, consume);
        if (!s) { PyErr_Print(); return 1; }
        Py_DECREF(s);
    }

    double fill_s = 0, call_s = 0, consume_s = 0;
    double t_all0 = now_s();
    for (long it = 0; it < iters; it++) {
        double t0 = now_s();
        for (long i = 0; i < B; i++) {
            long seq = it * B + i;
            fill_row(x_buf + i * row_bytes, seq);
            fill_mask_row(mask_buf + i * 4240, seq);
        }
        double t1 = now_s();
        PyObject *s = PyObject_CallMethod(mod, "step", "(ii)", (int)B, consume);
        if (!s) { PyErr_Print(); return 1; }
        Py_DECREF(s);
        double t2 = now_s();
        /* with consume=1 the call itself includes the deliver half (D2H
         * policy + per-position argmax read, done in the worker); there
         * is no separate C-side phase to time */
        double t3 = t2;
        fill_s += t1 - t0; call_s += t2 - t1; consume_s += t3 - t2;
    }
    /* bound the wall clock: without consume the stream runs async */
    PyObject *sy = PyObject_CallMethod(mod, "sync", NULL);
    if (!sy) { PyErr_Print(); return 1; }
    Py_DECREF(sy);
    double wall = now_s() - t_all0;

    double pos = (double)B * iters;
    printf("B=%ld iters=%ld compile=%d consume=%d\n", B, iters, comp, consume);
    printf("  fill    %8.2f ms/batch  (%6.1f GB/s host memcpy)\n",
           fill_s / iters * 1e3, (double)B * row_bytes / (fill_s / iters) / 1e9);
    printf("  call    %8.2f ms/batch  (H2D + forward%s)\n",
           call_s / iters * 1e3, consume ? " + D2H deliver" : "");
    printf("  wall    %8.2f ms/batch  (pipelined average)\n",
           wall / iters * 1e3);
    printf("  => positions/s %10.0f   tokens/s %10.0f\n",
           pos / wall, pos * 64 / wall);
    Py_Finalize();
    return 0;
}
