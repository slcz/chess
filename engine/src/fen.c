/*
 * fen.c — FEN parse / serialize.
 *
 * FEN is the external text format for positions (the internal format is the
 * State struct). Parse is strict enough to reject malformed input (the fuzz
 * harness feeds FENs produced by python-chess, and the tests rely on errors
 * being reported rather than half-parsed garbage). Halfmove/fullmove fields
 * may be omitted, matching common fixture files like Kiwipete.
 *
 * One deliberate choice: the en-passant field is stored *exactly as given*,
 * like python-chess does — we do not check whether an ep capture is actually
 * possible. Legal-move generation simply ignores an unusable ep square.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"

static const char PIECE_CHARS[13] = "PNBRQKpnbrqk"; /* index = piece code; see below */

void state_init_startpos(State *s) {
    if (fen_parse(s, CHESS_START_FEN) != 0) {
        /* unreachable: the string above is a constant we control */
        fprintf(stderr, "internal error: built-in start FEN failed to parse\n");
        exit(1);
    }
}

static int parse_placement(State *s, const char **p) {
    int rank = 7, file = 0;

    for (; **p && **p != ' '; (*p)++) {
        char c = **p;
        if (c == '/') {
            if (file != 8 || rank == 0) return -1;
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return -1;
        } else {
            const char *q = strchr(PIECE_CHARS, c);
            /* 12 (EMPTY sentinel) never matches, so q < PIECE_CHARS+12 here */
            if (!q || q - PIECE_CHARS >= 12 || file > 7) return -1;
            int pc = (int)(q - PIECE_CHARS);
            int sq = rank * 8 + file;
            s->board[sq] = (uint8_t)pc;
            s->bb[pc / 6][pc % 6] |= 1ULL << sq;
            s->occ[pc / 6] |= 1ULL << sq;
            file++;
        }
    }
    if (rank != 0 || file != 8) return -1;
    return 0;
}

int fen_parse(State *s, const char *fen) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < 64; i++) s->board[i] = EMPTY;
    s->occ[WHITE] = s->occ[BLACK] = 0;
    s->ply = 0;

    const char *p = fen;

    /* 1. Piece placement. */
    if (parse_placement(s, &p) != 0) return -1;
    while (*p == ' ') p++;

    /* 2. Side to move. */
    if (*p != 'w' && *p != 'b') return -1;
    s->side = (*p == 'w') ? WHITE : BLACK;
    p++;
    while (*p == ' ') p++;

    /* 3. Castling availability. */
    if (*p == '\0') return -1;
    if (*p == '-') {
        p++;
    } else {
        while (*p && *p != ' ') {
            if (*p == 'K')      s->castling |= CASTLE_WK;
            else if (*p == 'Q') s->castling |= CASTLE_WQ;
            else if (*p == 'k') s->castling |= CASTLE_BK;
            else if (*p == 'q') s->castling |= CASTLE_BQ;
            else return -1;
            p++;
        }
    }
    while (*p == ' ') p++;

    /* 4. En-passant target square. */
    if (*p == '\0') return -1;
    if (*p == '-') {
        s->ep = NO_SQ;
        p++;
    } else {
        if (p[0] < 'a' || p[0] > 'h' || p[1] < '1' || p[1] > '8') return -1;
        s->ep = (uint8_t)((p[1] - '1') * 8 + (p[0] - 'a'));
        p += 2;
    }
    while (*p == ' ') p++;

    /* 5/6. Halfmove clock and fullmove number (optional; default 0 and 1). */
    s->halfmove = 0;
    s->fullmove = 1;
    if (sscanf(p, "%hu %u", &s->halfmove, &s->fullmove) != 2) {
        char junk[2];
        if (sscanf(p, "%1s", junk) == 1) return -1; /* trailing garbage */
        s->halfmove = 0;
        s->fullmove = 1;
    }

    /* Sanity: lsb() on an empty king bitboard would be undefined behavior. */
    if (__builtin_popcountll(s->bb[WHITE][KING]) != 1) return -1;
    if (__builtin_popcountll(s->bb[BLACK][KING]) != 1) return -1;
    return 0;
}

void fen_serialize(const State *s, char *out, size_t n) {
    char *o = out;

    /* 1. Piece placement, rank 8 first. */
    for (int rank = 7; rank >= 0; rank--) {
        int empty_run = 0;
        for (int file = 0; file < 8; file++) {
            int pc = s->board[rank * 8 + file];
            if (pc == EMPTY) {
                empty_run++;
            } else {
                if (empty_run) { *o++ = (char)('0' + empty_run); empty_run = 0; }
                *o++ = PIECE_CHARS[pc];
            }
        }
        if (empty_run) *o++ = (char)('0' + empty_run);
        if (rank) *o++ = '/';
    }
    *o++ = ' ';

    /* 2. Side to move. */
    *o++ = s->side == WHITE ? 'w' : 'b';
    *o++ = ' ';

    /* 3. Castling rights (canonical KQkq order, '-' if none). */
    if (!s->castling) {
        *o++ = '-';
    } else {
        if (s->castling & CASTLE_WK) *o++ = 'K';
        if (s->castling & CASTLE_WQ) *o++ = 'Q';
        if (s->castling & CASTLE_BK) *o++ = 'k';
        if (s->castling & CASTLE_BQ) *o++ = 'q';
    }
    *o++ = ' ';

    /* 4. En-passant target. */
    if (s->ep == NO_SQ) {
        *o++ = '-';
    } else {
        *o++ = (char)('a' + (s->ep & 7));
        *o++ = (char)('1' + (s->ep >> 3));
    }

    /* 5/6. Clocks. */
    snprintf(o, n - (size_t)(o - out), " %u %u", s->halfmove, s->fullmove);
}
