#!/usr/bin/env node
/*
 * bridge_node.mjs — end-to-end smoke of the single-threaded stepping
 * protocol (puct_think_start / puct_zone_pending / puct_zone_bank /
 * puct_zone_publish in puct.c): the fake evaluator is a plain async
 * function on the SAME thread, replying with the dummy evaluator's exact
 * outputs (uniform priors + a draw) after a macrotask hop — so the bank
 * genuinely sits QUEUED across a turn of the event loop, with walkers
 * parked, before the publish delivers them. The transcript must equal
 * the native dummy-search transcript bit-for-bit.
 *
 * Usage: node web/test/bridge_node.mjs
 */
import { createEngine, diffTranscripts, jsDummyEval, nativeTranscript,
         runTranscript } from './parity_node.mjs';

const M = await createEngine();
M._chess_init();
M._puct_attach_js(8);            // zone only — the eval runs in JS

const evalFn = jsDummyEval(M, () => new Promise((r) => setTimeout(r, 0)));
diffTranscripts(nativeTranscript(), await runTranscript(M, evalFn), 'bridge');
