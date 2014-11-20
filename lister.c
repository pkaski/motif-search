/* 
 * This file is part of an experimental software implementation of
 * graph motif search utilizing the constrained multilinear 
 * sieving framework; cf. 
 * 
 *    A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri,
 *    "Engineering motif search for large graphs",
 *    ALENEX15 Meeting on Algorithm Engineering and Experiments,
 *    5 January 2015, San Diego, CA.
 * 
 * This experimental source code is supplied to accompany the 
 * aforementioned paper. 
 * 
 * The source code is configured for a gcc build to a native 
 * microarchitecture that must support the AVX2 and PCLMULQDQ 
 * instruction set extensions. Other builds are possible but 
 * require manual configuration of 'Makefile' and 'builds.h'.
 * 
 * The source code is subject to the following license.
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2014 A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<ctype.h>
#include<omp.h>

/************************************************************* Configuration. */

#define MAX_K          32
#define MAX_SHADES     32

#define PREFETCH_PAD   32
#define MAX_THREADS   128

#include"builds.h"        // get build config

typedef long int index_t; // default to 64-bit indexing

#include"gf.h"       // finite fields
#include"ffprng.h"   // fast-forward pseudorandom number generator


/********************************************************************* Flags. */

index_t flag_bin_input    = 0; // default to ASCII input

/************************************************************* Common macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/*********************************************************** Error reporting. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

/********************************************************* Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/********************************************************* Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/********************************************** Memory allocation & tracking. */

#define MALLOC(x) malloc_wrapper(x)
#define FREE(x) free_wrapper(x)

index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 256
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;    
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);
    
    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

/******************************************************** Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/******************************************************************* Sorting. */

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

void heapsort_indext(index_t n, index_t *a)
{
    /* Shift index origin from 0 to 1 for convenience. */
    a--; 
    /* Build heap */
    for(index_t i = 2; i <= n; i++) {
        index_t x = i;
        while(x > 1) {
            index_t y = PARENT(x);
            if(a[x] <= a[y]) {
                /* heap property ok */
                break;              
            }
            /* Exchange a[x] and a[y] to enforce heap property */
            index_t t = a[x];
            a[x] = a[y];
            a[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(index_t i = n; i > 1; i--) {
        index_t t = a[i];
        /* Delete max */
        a[i] = a[1];
        /* Insert t */
        index_t x = 1;
        index_t y, z;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[y] < a[z]) {
                index_t s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[y] >= a[z] */
            if(t >= a[y]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move a[y] up the heap */
            a[x] = a[y];
            x = y;
        }
        /* Insert here */
        a[x] = t; 
    }
}

/******************************************************* Bitmap manipulation. */

void bitset(index_t *map, index_t j, index_t value)
{
    assert((value & (~1UL)) == 0);
    map[j/64] = (map[j/64] & ~(1UL << (j%64))) | ((value&1) << (j%64));  
}

index_t bitget(index_t *map, index_t j)
{
    return (map[j/64]>>(j%64))&1UL;
}

/*************************************************** Random numbers and such. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

/***************************************************** (Parallel) prefix sum. */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    index_t run = 0;
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);    
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif

    return run; 
}


/************************ Search for an interval of values in a sorted array. */

inline index_t get_interval(index_t n, index_t *a, 
                            index_t lo_val, index_t hi_val,
                            index_t *iv_start, index_t *iv_end)
{
    assert(n >= 0);
    if(n == 0) {
        *iv_start = 0; 
        return 0;
    }
    assert(lo_val <= hi_val);
    // find first element in interval (if any) with binary search
    index_t lo = 0;
    index_t hi = n-1;
    // at or above lo, and at or below hi (if any)
    while(lo < hi) {
        index_t mid = (lo+hi)/2; // lo <= mid < hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi (if any)
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo (if any), lo <= hi
            else
                hi = mid;   // at or below hi (exists) 
        }
        // 0 <= lo <= n-1
    }
    if(a[lo] < lo_val || a[lo] > hi_val) {
        // array contains no values in interval
        if(a[lo] < lo_val) {
            lo++;
            assert(lo == n || a[lo+1] > hi_val);
        } else {
            assert(lo == 0 || a[lo-1] < lo_val);
        }
        *iv_start = lo; 
        *iv_end   = hi;
        return 0; 
    }
    assert(lo_val <= a[lo] && a[lo] <= hi_val);
    *iv_start = lo;
    // find interval end (last index in interval) with binary search
    lo = 0;
    hi = n-1;
    // last index (if any) is at or above lo, and at or below hi
    while(lo < hi) {
        index_t mid = (lo+hi+1)/2; // lo < mid <= hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi, lo <= hi
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo
            else
                lo = mid;   // at or above lo, lo <= hi
        }
    }
    assert(lo == hi);
    *iv_end = lo; // lo == hi
    return 1+*iv_end-*iv_start; // return cut size
}



/****************************************************************** Sieving. */

long long int num_muls;
long long int trans_bytes;

#define SHADE_LINES ((MAX_SHADES+SCALARS_IN_LINE-1)/SCALARS_IN_LINE)
typedef unsigned int shade_map_t;

void constrained_sieve_pre(index_t         n,
                           index_t         k,
                           index_t         g,
                           index_t         pfx,
                           index_t         num_shades,
                           shade_map_t     *d_s,
                           ffprng_scalar_t seed,
                           line_array_t    *d_x)
{
    assert(g == SCALARS_IN_LINE);   
    assert(num_shades <= MAX_SHADES);

    line_t   wdj[SHADE_LINES*MAX_K];

    ffprng_t base;
    FFPRNG_INIT(base, seed);
    for(index_t j = 0; j < k; j++) {
        for(index_t dl = 0; dl < SHADE_LINES; dl++) {
            index_t jsdl = j*SHADE_LINES+dl;
            LINE_SET_ZERO(wdj[jsdl]);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, base);
                scalar_t rs = (scalar_t) rnd;
                LINE_STORE_SCALAR(wdj[jsdl], a, rs);   // W: [cached]
            }
        }
    }

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        FFPRNG_FWD(gen, SHADE_LINES*SCALARS_IN_LINE*start, base);
        line_t vd[SHADE_LINES];
        for(index_t j = 0; j < SHADE_LINES; j++) {
            LINE_SET_ZERO(vd[j]); // to cure an annoying compiler warning
        }       
        for(index_t u = start; u <= stop; u++) {
            scalar_t uu[MAX_K];
            shade_map_t shades_u = d_s[u];            // R: n   shade_map_t
            for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                    index_t d = dl*SCALARS_IN_LINE + a;
                    ffprng_scalar_t rnd;
                    FFPRNG_RAND(rnd, gen);
                    scalar_t rs = (scalar_t) rnd;
                    rs = rs & (-((scalar_t)((shades_u >> d)&(d < num_shades))));  
                    LINE_STORE_SCALAR(vd[dl], a, rs); // W: [cached]
                }
            }
            for(index_t j = 0; j < k; j++) {
                scalar_t uj;
                SCALAR_SET_ZERO(uj);
                for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                    index_t jsdl = j*SHADE_LINES+dl;
                    line_t ln;
                    LINE_MUL(ln, wdj[jsdl], vd[dl]);  // R: [cached]
                                                      // MUL: n*SHADE_LINES*g*k
                    scalar_t lns;
                    LINE_SUM(lns, ln);
                    SCALAR_ADD(uj, uj, lns);
                }
                uu[j] = uj;
            }
            line_t ln;
            LINE_SET_ZERO(ln);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                index_t ap = a < (1L << k) ? pfx+a : 0;
                scalar_t xua;
                SCALAR_SET_ZERO(xua);
                for(index_t j = 0; j < k; j++) {
                    scalar_t z_uj = uu[j];            // R: [cached]
                    z_uj = z_uj & (-((scalar_t)(((ap) >> j)&1)));
                    SCALAR_ADD(xua, xua, z_uj);
                }
                LINE_STORE_SCALAR(ln, a, xua);
            }
            LINE_STORE(d_x, u, ln);                  // W: ng scalar_t
        }
    }

    num_muls    += n*SHADE_LINES*g*k;
    trans_bytes += sizeof(scalar_t)*n*g + sizeof(shade_map_t)*n;
}

/***************************************************************** Line sum. */

scalar_t line_sum(index_t      l, 
                  index_t      g,
                  line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        SCALAR_SET_ZERO(ts[t]);
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            LINE_LOAD(ln, d_s, i);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[t] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t t = 0; t < nt; t++) {
        SCALAR_ADD(sum, sum, ts[t]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;

    return sum;
}


scalar_t line_sum_stride(index_t      l, 
                         index_t      g,
                         index_t      stride,
                         line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        SCALAR_SET_ZERO(ts[t]);
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            index_t ii = i*stride;
            LINE_LOAD(ln, d_s, ii);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[t] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t t = 0; t < nt; t++) {
        SCALAR_ADD(sum, sum, ts[t]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;

    return sum;
}


/****************************** k-arborescence generating function (mark 1). */

#define ARB_LINE_IDX(b, k, l, u) ((k)*(u)+(l)-1)

void k_arborescence_genf1_round(index_t         n,
                                index_t         m,
                                index_t         k,
                                index_t         g,
                                index_t         l,
                                index_t         *d_pos,
                                index_t         *d_adj,
                                ffprng_scalar_t yl_seed,
                                line_array_t    *d_s) 
{
    assert(g == SCALARS_IN_LINE);   

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[start]-start;
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[u];                // R: n  index_t   [hw pref]
            index_t deg = d_adj[pu];               // R: n  index_t   [hw pref]
            line_t pul;
            LINE_SET_ZERO(pul);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];           // R: m  index_t   [hw pref]
#ifdef BUILD_PREFETCH
                index_t nv = d_adj[pu+j+(j < deg ? 1 : 2)];
                    // stride user prefetch over the degree gap in d_adj
#endif
                line_t s;
                LINE_SET_ZERO(s);
                for(index_t l1 = 1; l1 < l; l1++) {
                    line_t pul1, pvl2;
                    index_t l2 = l-l1; // l2 runs a decreasing index
                    index_t i_v_l2 = ARB_LINE_IDX(b, k, l2, v); // l1>=1
                    LINE_LOAD(pvl2, d_s, i_v_l2);  
                                              // R: (l-1)mg scalar_t [user pref]
                    index_t i_u_l1 = ARB_LINE_IDX(b, k, l1, u); // l2>=1
                    LINE_LOAD(pul1, d_s, i_u_l1);  
                                              // R: (l-1)mg scalar_t [hw pref]
#ifdef BUILD_PREFETCH
                    index_t i_nv_l2 = ARB_LINE_IDX(b, k, l2, nv);
                    LINE_PREFETCH(d_s, i_nv_l2);  // issue user prefetch
#endif
                    line_t p;
                    LINE_MUL(p, pul1, pvl2);       // MUL: (l-1)mg
                    LINE_ADD(s, s, p);
                }
                ffprng_scalar_t rnd;               
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luv = (scalar_t) rnd;
                line_t sy;
                LINE_MUL_SCALAR(sy, s, y_luv);     // MUL: ng
                LINE_ADD(pul, pul, sy);
            }
            index_t i_u_l = ARB_LINE_IDX(b, k, l, u);
            LINE_STORE(d_s, i_u_l, pul);      // W: ng  scalar_t
        }
    }

    trans_bytes += (2*n+m)*sizeof(index_t) + (2*(l-1)*m+n)*g*sizeof(scalar_t);
    num_muls    += ((l-1)*m+n)*g;
}

scalar_t k_arborescence_genf1(index_t         n,
                              index_t         m,
                              index_t         k,
                              index_t         g,
                              index_t         *d_pos,
                              index_t         *d_adj, 
                              ffprng_scalar_t y_seed,
                              line_array_t    *d_x) 
{

    assert(g == SCALARS_IN_LINE);   
    assert(k >= 1);

    line_array_t *d_s  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(k*n*g));

    /* Save the base case to d_s. */

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        line_t xu;
        LINE_LOAD(xu, d_x, u);              // R: ng  scalar_t [hw prefetched]
        index_t i_u_1 = ARB_LINE_IDX(b, k, 1, u);
        LINE_STORE(d_s, i_u_1, xu);         // W: ng  scalar_t
    }

    /* Run the recurrence. */
    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        ffprng_scalar_t yl_seed = irand(); // different y-values for every round
        k_arborescence_genf1_round(n,m,k,g,l,d_pos,d_adj,yl_seed,d_s);
    }

    /* Sum up. */

    scalar_t sum = line_sum_stride(n, g, k, 
                                          ((line_array_t *)(((line_t *) d_s) + k - 1)));

    FREE(d_s);

    trans_bytes += 2*n*g*sizeof(scalar_t);
    num_muls    += 0;

    return sum;
}




/****************************** k-arborescence generating function (mark 2). */

void k_arborescence_genf2_round(index_t         n,
                                index_t         m,
                                index_t         k,
                                index_t         g,
                                index_t         l,
                                index_t         *d_pos,
                                index_t         *d_adj,
                                ffprng_scalar_t yl_seed,
                                line_array_t    *d_in,
                                line_array_t    *d_out)
{
    assert(g == SCALARS_IN_LINE);   

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);


#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[start]-start;
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[u];                // R: n  index_t   [hw pref]
            index_t deg = d_adj[pu];               // R: n  index_t   [hw pref]
#ifdef BUILD_PREFETCH
            index_t poff = 4;
            index_t vvp = d_adj[pu+0+poff];      // look ahead adjacency list
            LINE_PREFETCH(d_in, vvp);            // issue user prefetch
#endif
            line_t pyvsum;
            LINE_SET_ONE(pyvsum);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];           // R: m  index_t   [hw pref]
                line_t pvin;
                LINE_LOAD(pvin, d_in, v);     // R: mg scalar_t [user prefetch]
#ifdef BUILD_PREFETCH
                index_t vp = d_adj[pu+j+poff];   // look ahead adjacency list
                LINE_PREFETCH(d_in, vp);         // issue user prefetch 
#endif
                ffprng_scalar_t rnd;               
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luv = (scalar_t) rnd;
                line_t yuvpvin;
                LINE_MUL_SCALAR(yuvpvin, pvin, y_luv);   // MUL: mg
                LINE_ADD(pyvsum, pyvsum, yuvpvin);
            }
            line_t puin;
            LINE_LOAD(puin, d_in, u);              // R: ng  scalar_t [hw pref]
            line_t puout;
            LINE_MUL(puout, puin, pyvsum);               // MUL: ng
            LINE_STORE(d_out, u, puout);           // W: ng  scalar_t
        }
    }

    trans_bytes += (2*n+m)*sizeof(index_t) + (m+2*n)*g*sizeof(scalar_t);
    num_muls    += (m+n)*g;
}

scalar_t k_arborescence_genf2(index_t         n,
                              index_t         m,
                              index_t         k,
                              index_t         g,
                              index_t         *d_pos,
                              index_t         *d_adj, 
                              ffprng_scalar_t y_seed,
                              line_array_t    *d_x) 
{

    assert(g == SCALARS_IN_LINE);   
    assert(k >= 1);

    line_array_t *d_s0  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));
    line_array_t *d_s1  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));

    scalar_t sum;
    SCALAR_SET_ZERO(sum);

    // Evaluate at 2^{k-1}+1 distinct z values and do Lagrange interpolation
    // to recover the coefficient of the monomial with degree k
    // (polynomial has degree at most 2^{k-1} in z, the coefficient of
    //  degree 0 monomial is 0; could get rid of the +1)
    index_t deg = 1UL << (k-1);
    for(index_t d = 0; d <= deg; d++) {
        scalar_t zd = d+1; 
        
        /* Save the base case to d_s0. */
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            line_t xu;
            LINE_LOAD(xu, d_x, u);       // R: (deg+1)*ng  scalar_t [hw pref]
            line_t xuzd;
            LINE_MUL_SCALAR(xuzd, xu, zd);  // MUL: deg*ng
            LINE_STORE(d_s0, u, xuzd);   // W: (deg+1)*ng  scalar_t
        }

        /* Run the recurrence. */
        srand(y_seed);
        for(index_t l = 2; l <= k; l++) {
            ffprng_scalar_t yl_seed = irand(); 
                // different y-values for every round
                // but the same values repeat for each d
            k_arborescence_genf2_round(n,m,k,g,l,d_pos,d_adj,yl_seed,d_s0,d_s1);
            line_array_t *tmp = d_s0;
            d_s0 = d_s1;
            d_s1 = tmp;
        }
        // Invariant: d_s0 contains output

        /* Sum up & accumulate interpolation. */
        scalar_t zdsum = line_sum(n, g, d_s0);
        scalar_t ldk = lagrange_coeff(deg, d, k);
        scalar_t pk;
        SCALAR_MUL(pk, ldk, zdsum);
        SCALAR_ADD(sum, sum, pk); 
    }

    FREE(d_s0);
    FREE(d_s1);

    trans_bytes += (deg+1)*2*n*g*sizeof(scalar_t);
    num_muls    += (deg+1)*n*g;

    return sum;
}



/********************************** Initialize an array with random scalars. */

void randinits_scalar(scalar_t *a, index_t s, ffprng_scalar_t seed) 
{
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t nt = num_threads();
    index_t block_size = s/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? s-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            scalar_t rs = (scalar_t) rnd;           
            a[i] = rs;
        }
    }
}

/************************************************************ The oracle(s). */

index_t oracle(index_t         n,
               index_t         k,
               index_t         *h_pos,
               index_t         *h_adj,
               index_t         num_shades,
               shade_map_t     *h_s,
               ffprng_scalar_t y_seed,
               ffprng_scalar_t z_seed) 
{
    push_memtrack();
    assert(k >= 1 && k < 31);
    index_t m = h_pos[n-1]+h_adj[h_pos[n-1]]+1-n;
    index_t sum_size = 1 << k;       

    index_t g = SCALARS_IN_LINE;
    index_t outer = (sum_size + g-1) / g; 
       // number of iterations for outer loop

    num_muls = 0;
    trans_bytes = 0;

    index_t *d_pos     = h_pos;
    index_t *d_adj     = h_adj;
    line_array_t *d_x  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));

    /* Run the work & time it. */

    push_time();

    scalar_t master_sum;
    SCALAR_SET_ZERO(master_sum);
    for(index_t out = 0; out < outer; out++) {

        constrained_sieve_pre(n, k, g, g*out, num_shades, h_s, z_seed, d_x);
#if BUILD_GENF == 1
        scalar_t sum = k_arborescence_genf1(n, m, k, g, d_pos, d_adj, y_seed, d_x);
#define GENF_TYPE "k_arborescence_genf1"
#else
#if BUILD_GENF == 2
        scalar_t sum = k_arborescence_genf2(n, m, k, g, d_pos, d_adj, y_seed, d_x);
#define GENF_TYPE "k_arborescence_genf2"
#else
#error BUILD_GENF should be either 1 or 2
#endif
#endif

        SCALAR_ADD(master_sum, master_sum, sum);
    }

    double time = pop_time();
    double trans_rate = trans_bytes / (time/1000.0);
    double mul_rate = num_muls / time;
    FREE(d_x);

    fprintf(stdout, 
            SCALAR_FORMAT_STRING
            " %10.2lf ms [%8.2lfGiB/s, %8.2lfGHz] %d",
            (long) master_sum,
            time,
            trans_rate/((double) (1 << 30)),
            mul_rate/((double) 1e6),
            master_sum != 0);
    fprintf(stdout, " ");
    print_pop_memtrack();
    fprintf(stdout, " ");   
    print_current_mem();   
    fflush(stdout);

    return master_sum != 0;
}



/************************************************* Rudimentary graph builder. */

typedef struct 
{
    index_t num_vertices;
    index_t num_edges;
    index_t edge_capacity;
    index_t *edges;
    index_t *colors;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    index_t i;
    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->num_vertices = n;
    g->num_edges = 0;
    g->edge_capacity = 100;
    g->edges = enlarge(2*g->edge_capacity, 0, (void *) 0);
    g->colors = (index_t *) MALLOC(sizeof(index_t)*n);
    for(i = 0; i < n; i++)
        g->colors[i] = -1;
    return g;
}

void graph_free(graph_t *g)
{
    FREE(g->edges);
    FREE(g->colors);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(4*g->edge_capacity, 2*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 2*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(2*g->edge_capacity+2*cap, 2*g->edge_capacity, g->edges);
    index_t *e = g->edges + 2*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

void graph_set_color(graph_t *g, index_t u, index_t c)
{
    assert(u >= 0 && u < g->num_vertices && c >= 0);
    g->colors[u] = c;
}

/************************************* Basic motif query processing routines. */

struct motifq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;  
    index_t     ns;
    shade_map_t *shade;
};

typedef struct motifq_struct motifq_t;

void adjsort(index_t n, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        heapsort_indext(deg, adj + pu + 1);
    }
}

void motifq_free(motifq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
    }
    FREE(q);
}

index_t motifq_execute(motifq_t *q)
{
    if(q->is_stub)
        return 0;
    return oracle(q->n, q->k, q->pos, q->adj, q->ns, q->shade, irand(), irand());
}

/*************** Project a query by cutting out a given interval of vertices. */

index_t get_poscut(index_t n, index_t *pos, index_t *adj, 
                   index_t lo_v, index_t hi_v,
                   index_t *poscut)
{
    // Note: assumes the adjacency lists are sorted
    assert(lo_v <= hi_v);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < lo_v; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(deg, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        poscut[u] = deg - l;
    }

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = hi_v+1; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(deg, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        poscut[u-hi_v-1+lo_v] = deg - l;
    }

    index_t ncut = n - (hi_v-lo_v+1);
    index_t run = prefixsum(ncut, poscut, 1);
    return run;
}

motifq_t *motifq_cut(motifq_t *q, index_t lo_v, index_t hi_v)
{
    // Note: assumes the adjacency lists are sorted

    index_t n = q->n;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
    assert(0 <= lo_v && lo_v <= hi_v && hi_v < n);

    // Fast-forward a stub NO when the interval 
    // [lo_v,hi_v] contains an element in q->l
    for(index_t i = 0; i < q->nl; i++) {
        if(q->l[i] >= lo_v && q->l[i] <= hi_v) {
            motifq_t *qs = (motifq_t *) MALLOC(sizeof(motifq_t));
            qs->is_stub = 1;
            return qs;
        }
    }

    index_t ncut = n - (hi_v-lo_v+1);
    index_t *poscut = alloc_idxtab(ncut);
    index_t bcut = get_poscut(n, pos, adj, lo_v, hi_v, poscut);
    index_t *adjcut = alloc_idxtab(bcut);
    index_t gap = hi_v-lo_v+1;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < ncut; v++) {
        index_t u = v;
        if(u >= lo_v)
            u += gap;
        index_t pu = pos[u];
        index_t degu = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(degu, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        index_t pv = poscut[v];
        index_t degv = degu - l;
        adjcut[pv] = degv;
        // could parallelize this too
        for(index_t i = 0; i < cs; i++)
            adjcut[pv + 1 + i] = adj[pu + 1 + i];
        // could parallelize this too
        for(index_t i = cs; i < degv; i++)
            adjcut[pv + 1 + i] = adj[pu + 1 + i + l] - gap;
    }

    motifq_t *qq = (motifq_t *) MALLOC(sizeof(motifq_t));
    qq->is_stub = 0;
    qq->n = ncut;
    qq->k = q->k;
    qq->pos = poscut;
    qq->adj = adjcut;
    qq->nl = q->nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*qq->nl);
    for(index_t i = 0; i < qq->nl; i++) {
        index_t u = q->l[i];
        assert(u < lo_v || u > hi_v);
        if(u > hi_v)
            u -= gap;
        qq->l[i] = u;
    }
    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*ncut);
    for(index_t v = 0; v < ncut; v++) {
        index_t u = v;
        if(u >= lo_v)
            u += gap;
        qq->shade[v] = q->shade[u];
    }

    return qq;
}

/****************** Project a query with given projection & embedding arrays. */

#define PROJ_UNDEF 0xFFFFFFFFFFFFFFFFUL

index_t get_posproj(index_t n, index_t *pos, index_t *adj, 
                    index_t nproj, index_t *proj, index_t *embed,
                    index_t *posproj)
{

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                degproj++;
        }
        posproj[v] = degproj;
    }

    index_t run = prefixsum(nproj, posproj, 1);
    return run;
}

motifq_t *motifq_project(motifq_t *q, 
                         index_t nproj, index_t *proj, index_t *embed,
                         index_t nl, index_t *l)
{
    index_t n = q->n;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
 
    index_t *posproj = alloc_idxtab(nproj);
    index_t bproj = get_posproj(n, pos, adj, nproj, proj, embed, posproj);
    index_t *adjproj = alloc_idxtab(bproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t pv = posproj[v];
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                adjproj[pv + 1 + degproj++] = w;
        }
        adjproj[pv] = degproj;
    }

    motifq_t *qq = (motifq_t *) MALLOC(sizeof(motifq_t));
    qq->is_stub = 0;
    qq->n = nproj;
    qq->k = q->k;
    qq->pos = posproj;
    qq->adj = adjproj;

    // Now project the l array

    assert(q->nl == 0); // l array comes from lister    
    qq->nl = nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*nl);
    for(index_t i = 0; i < nl; i++) {
        index_t u = proj[l[i]];
        assert(u != PROJ_UNDEF); // query is a trivial NO !
        qq->l[i] = u;
    }

    // Next set up the projected shades

    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*nproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t v = proj[u];
        if(v != PROJ_UNDEF)
            qq->shade[v] = q->shade[u];
    }

    // Reserve a unique shade to every vertex in l
    // while keeping the remaining shades available

    // Reserve shades first ... 
    index_t *l_shade = (index_t *) MALLOC(sizeof(index_t)*nl);
    shade_map_t reserved_shades = 0;
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        index_t j = 0;
        for(; j < qq->ns; j++)
            if(((qq->shade[v] >> j)&1) == 1 && 
               ((reserved_shades >> j)&1) == 0)
                break;
        assert(j < qq->ns);
        reserved_shades |= 1UL << j;
        l_shade[i] = j;
    }
    // ... then clear all reserved shades in one pass

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++)
        qq->shade[v] &= ~reserved_shades;

    // ... and finally set reserved shades
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        qq->shade[v] = 1UL << l_shade[i];
    }
    FREE(l_shade);

    return qq;
}

/**************************************************** The interval extractor. */

struct ivlist_struct
{
    index_t start;
    index_t end;
    struct ivlist_struct *prev;
    struct ivlist_struct *next;
};

typedef struct ivlist_struct ivlist_t;

typedef struct ivext_struct 
{
    index_t     n;
    index_t     k;
    ivlist_t    *queue;
    ivlist_t    *active_queue_head;
    ivlist_t    *spare_queue_head;
    ivlist_t    *embed_list;
} ivext_t;

void ivext_enqueue_spare(ivext_t *e, ivlist_t *iv)
{
    pnlinknext(e->spare_queue_head,iv);
}

void ivext_enqueue_active(ivext_t *e, ivlist_t *iv)
{
    pnlinkprev(e->active_queue_head,iv);
}

ivlist_t *ivext_dequeue_first_nonsingleton(ivext_t *e)
{
    ivlist_t *iv = e->active_queue_head->next;  
    for(; 
        iv != e->active_queue_head; 
        iv = iv->next)
        if(iv->end - iv->start + 1 > 1)
            break;
    assert(iv != e->active_queue_head);
    pnunlink(iv);
    return iv;
}

ivlist_t *ivext_get_spare(ivext_t *e)
{
    assert(e->spare_queue_head->next != e->spare_queue_head);
    ivlist_t *iv = e->spare_queue_head->next;
    pnunlink(iv);
    return iv;
}

void ivext_reset(ivext_t *e)
{
    e->active_queue_head = e->queue + 0;
    e->spare_queue_head  = e->queue + 1;
    e->active_queue_head->next = e->active_queue_head;
    e->active_queue_head->prev = e->active_queue_head;
    e->spare_queue_head->prev  = e->spare_queue_head;
    e->spare_queue_head->next  = e->spare_queue_head;  
    e->embed_list = (ivlist_t *) 0;

    for(index_t i = 0; i < e->k + 2; i++)
        ivext_enqueue_spare(e, e->queue + 2 + i); // rot-safe
    ivlist_t *iv = ivext_get_spare(e);
    iv->start = 0;
    iv->end = e->n-1;
    ivext_enqueue_active(e, iv);
}

ivext_t *ivext_alloc(index_t n, index_t k)
{
    ivext_t *e = (ivext_t *) MALLOC(sizeof(ivext_t));
    e->n = n;
    e->k = k;
    e->queue = (ivlist_t *) MALLOC(sizeof(ivlist_t)*(k+4)); // rot-safe
    ivext_reset(e);
    return e;
}

void ivext_free(ivext_t *e)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        ivlist_t *temp = el;
        el = el->next;
        FREE(temp);
    }
    FREE(e->queue);
    FREE(e);
}

void ivext_project(ivext_t *e, ivlist_t *iv)
{
    for(ivlist_t *z = e->active_queue_head->next; 
        z != e->active_queue_head; 
        z = z->next) {
        assert(z->end < iv->start ||
               z->start > iv->end);
        if(z->start > iv->end) {
            z->start -= iv->end-iv->start+1;
            z->end   -= iv->end-iv->start+1;
        }
    }

    ivlist_t *em = (ivlist_t *) MALLOC(sizeof(ivlist_t));
    em->start    = iv->start;
    em->end      = iv->end;
    em->next     = e->embed_list;
    e->embed_list = em;
}

index_t ivext_embed(ivext_t *e, index_t u)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        if(u >= el->start)
            u += el->end - el->start + 1;
        el = el->next;
    }
    return u;
}

ivlist_t *ivext_halve(ivext_t *e, ivlist_t *iv)
{
    assert(iv->end - iv->start + 1 >= 2);
    index_t mid = (iv->start + iv->end)/2;  // mid < iv->end    
    ivlist_t *h = ivext_get_spare(e);
    h->start = iv->start;
    h->end = mid;
    iv->start = mid+1;
    return h;
}
    
index_t ivext_queue_size(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s += iv->end-iv->start+1;
    return s;
}

index_t ivext_num_active_intervals(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s++;
    return s;
}

void ivext_queue_print(FILE *out, ivext_t *e, index_t rot)
{
    index_t j = 0;
    char x[16384];
    char y[16384];
    y[0] = '\0';
    sprintf(x, "%c%12ld [", 
            rot == 0 ? ' ' : 'R',
            ivext_queue_size(e));
    strcat(y, x);
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next) {
        assert(iv->start <= iv->end);
        if(iv->start < iv->end)
            sprintf(x, 
                    "%s[%ld:%ld]", 
                    j++ == 0 ? "" : ",",
                    ivext_embed(e, iv->start),
                    ivext_embed(e, iv->end));
        else
            sprintf(x, 
                    "%s[%ld]", 
                    j++ == 0 ? "[" : ",",
                    ivext_embed(e, iv->start));
        strcat(y, x);
    }   
    strcat(y, "] ");
    fprintf(out, "%-120s", y);
    fflush(out);
}

index_t extract_match(index_t is_root, motifq_t *query, index_t *match)
{
    // Assumes adjancency lists of query are sorted.

    fprintf(stdout, "extract: %ld %ld %ld\n", query->n, query->k, query->nl);
    push_time();
    assert(query->k <= query->n);
    ivext_t *e = ivext_alloc(query->n, query->k);
    ivext_queue_print(stdout, e, 0);
    if(!motifq_execute(query)) {
        fprintf(stdout, " -- false\n");
        ivext_free(e);
        if(!is_root)
            motifq_free(query);
        double time = pop_time();
        fprintf(stdout, "extract done [%.2lf ms]\n", time);
        return 0;
    }
    fprintf(stdout, " -- true\n");
           
    while(ivext_queue_size(e) > e->k) {
        ivlist_t *iv = ivext_dequeue_first_nonsingleton(e);
        ivlist_t *h = ivext_halve(e, iv);
        ivext_enqueue_active(e, iv);
        motifq_t *qq = motifq_cut(query, h->start, h->end);
        ivext_queue_print(stdout, e, 0);
        if(motifq_execute(qq)) {
            fprintf(stdout, " -- true\n");
            if(!is_root)
                motifq_free(query);
            query = qq;
            is_root = 0;
            ivext_project(e, h);
            ivext_enqueue_spare(e, h);
        } else {
            fprintf(stdout, " -- false\n");
            motifq_free(qq);
            pnunlink(iv);
            ivext_enqueue_active(e, h);
            qq = motifq_cut(query, iv->start, iv->end);
            ivext_queue_print(stdout, e, 0);
            if(motifq_execute(qq)) {
                fprintf(stdout, " -- true\n");
                if(!is_root)
                    motifq_free(query);
                query = qq;
                is_root = 0;
                ivext_project(e, iv);
                ivext_enqueue_spare(e, iv);
            } else {
                fprintf(stdout, " -- false\n");
                motifq_free(qq);
                ivext_enqueue_active(e, iv);
                while(ivext_num_active_intervals(e) > e->k) {
                    // Rotate queue until outlier is out ...
                    ivlist_t *iv = e->active_queue_head->next;  
                    pnunlink(iv);
                    qq = motifq_cut(query, iv->start, iv->end);
                    ivext_queue_print(stdout, e, 1);
                    if(motifq_execute(qq)) {
                        fprintf(stdout, " -- true\n");
                        if(!is_root)
                            motifq_free(query);
                        query = qq;
                        is_root = 0;
                        ivext_project(e, iv);
                        ivext_enqueue_spare(e, iv);
                    } else {
                        fprintf(stdout, " -- false\n");
                        motifq_free(qq);
                        ivext_enqueue_active(e, iv);
                    }
                }
            }
        }
    }
    for(index_t i = 0; i < query->k; i++)
        match[i] = ivext_embed(e, i);
    ivext_free(e);
    if(!is_root)
        motifq_free(query);
    double time = pop_time();
    fprintf(stdout, "extract done [%.2lf ms]\n", time);
    return 1;
}

/**************************************************************** The lister. */

#define M_QUERY       0
#define M_OPEN        1
#define M_CLOSE       2
#define M_REWIND_U    3
#define M_REWIND_L    4

index_t command_mnemonic(index_t command) 
{
    return command >> 60;   
}

index_t command_index(index_t command)
{
    return command & (~(0xFFUL<<60));
}

index_t to_command_idx(index_t mnemonic, index_t idx)
{
    assert(idx < (1UL << 60));
    return (mnemonic << 60)|idx;
}

index_t to_command(index_t mnemonic)
{
    return to_command_idx(mnemonic, 0UL);
}

typedef struct 
{
    index_t n;              // number of elements in universe
    index_t k;              // size of the sets to be listed
    index_t *u;             // upper bound as a bitmap
    index_t u_size;         // size of upper bound
    index_t *l;             // lower bound 
    index_t l_size;         // size of lower bound
    index_t *stack;         // a stack for maintaining state
    index_t stack_capacity; // ... the capacity of the stack    
    index_t top;            // index of stack top
    motifq_t *root;         // the root query
} lister_t;

void lister_push(lister_t *t, index_t word)
{
    assert(t->top + 1 < t->stack_capacity);
    t->stack[++t->top] = word;
}

index_t lister_pop(lister_t *t)
{
    return t->stack[t->top--];
}

index_t lister_have_work(lister_t *t)
{
    return t->top >= 0;
}

index_t lister_in_l(lister_t *t, index_t j)
{
    for(index_t i = 0; i < t->l_size; i++)
        if(t->l[i] == j)
            return 1;
    return 0;
}

void lister_push_l(lister_t *t, index_t j)
{
    assert(!lister_in_l(t, j) && t->l_size < t->k);
    t->l[t->l_size++] = j;
}

void lister_pop_l(lister_t *t)
{
    assert(t->l_size > 0);
    t->l_size--;
}

void lister_reset(lister_t *t)
{
    t->l_size = 0;
    t->top = -1;
    lister_push(t, to_command(M_QUERY));
    for(index_t i = 0; i < t->n; i++)
        bitset(t->u, i, 1);
    t->u_size = t->n;
}

lister_t *lister_alloc(index_t n, index_t k, motifq_t *root)
{
    assert(n >= 1 && n < (1UL << 60) && k >= 1 && k <= n);
    lister_t *t = (lister_t *) MALLOC(sizeof(lister_t));
    t->n = n;
    t->k = k;
    t->u = alloc_idxtab((n+63)/64);
    t->l = alloc_idxtab(k);
    t->stack_capacity = n + k*(k+1+2*k) + 1;
    t->stack = alloc_idxtab(t->stack_capacity);
    lister_reset(t);
    t->root = root;
    if(t->root != (motifq_t *) 0) {
        assert(t->root->n == t->n);
        assert(t->root->k == t->k);
        assert(t->root->nl == 0);
    }
    return t;
}

void lister_free(lister_t *t)
{
    if(t->root != (motifq_t *) 0)
        motifq_free(t->root);
    FREE(t->u);
    FREE(t->l);
    FREE(t->stack);
    FREE(t);
}

void lister_get_proj_embed(lister_t *t, index_t **proj_out, index_t **embed_out)
{
    index_t n = t->n;
    index_t usize = t->u_size;

    index_t *embed = (index_t *) MALLOC(sizeof(index_t)*usize);
    index_t *proj  = (index_t *) MALLOC(sizeof(index_t)*n);

    // could parallelize this (needs parallel prefix sum)
    index_t run = 0;
    for(index_t i = 0; i < n; i++) {
        if(bitget(t->u, i)) {
            proj[i]    = run;
            embed[run] = i;
            run++;
        } else {
            proj[i] = PROJ_UNDEF;
        }
    }
    assert(run == usize);

    *proj_out  = proj;
    *embed_out = embed;
}

void lister_query_setup(lister_t *t, motifq_t **q_out, index_t **embed_out)
{
    index_t *proj;
    index_t *embed;

    // set up the projection with u and l
    lister_get_proj_embed(t, &proj, &embed);
    motifq_t *qq = motifq_project(t->root, 
                                  t->u_size, proj, embed, 
                                  t->l_size, t->l);
    FREE(proj);

    *q_out     = qq;
    *embed_out = embed;
}

index_t lister_extract(lister_t *t, index_t *s)
{
    // assumes t->u contains all elements of t->l
    // (otherwise query is trivial no)

    assert(t->root != (motifq_t *) 0);
    
    if(t->u_size == t->n) {
        // rush the root query without setting up a copy
        return extract_match(1, t->root, s);
    } else {
        // a first order of business is to set up the query 
        // based on the current t->l and t->u; this includes
        // also setting up the embedding back to the root,
        // in case we are lucky and actually discover a match
        motifq_t *qq; // will be released by extractor
        index_t *embed;
        lister_query_setup(t, &qq, &embed);
        
        // now execute the interval extractor ...
        index_t got_match = extract_match(0, qq, s);
        
        // ... and embed the match (if any) 
        if(got_match) {
            for(index_t i = 0; i < t->k; i++)
                s[i] = embed[s[i]];
        }
        FREE(embed);
        return got_match;
    }
}

index_t lister_run(lister_t *t, index_t *s)
{
    while(lister_have_work(t)) {
        index_t cmd = lister_pop(t);
        index_t mnem = command_mnemonic(cmd);
        index_t idx = command_index(cmd);
        switch(mnem) {
        case M_QUERY:
            if(t->k <= t->u_size && lister_extract(t, s)) {
                // we have discovered a match, which we need to
                // put on the stack to continue work when the user
                // requests this
                for(index_t i = 0; i < t->k; i++)
                    lister_push(t, s[i]);
                lister_push(t, to_command_idx(M_OPEN, t->k-1));
                // now report our discovery to user
                return 1;
            }
            break;
        case M_OPEN:
            {
                index_t *x = t->stack + t->top - t->k + 1;
                index_t k = 0;
                for(; k < idx; k++)
                    if(!lister_in_l(t, x[k]))
                        break;
                if(k == idx) {
                    // opening on last element of x not in l
                    // so we can dispense with x as long as we remember to 
                    // insert x[idx] back to u when rewinding
                    for(index_t j = 0; j < t->k; j++)
                        lister_pop(t); // axe x from stack
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        lister_push(t, to_command(M_QUERY));
                    }
                } else {
                    // have still other elements of x that we need to
                    // open on, so must keep x in stack 
                    // --
                    // invariant that controls stack size:
                    // each open increases l by at least one
                    lister_push(t, to_command_idx(M_CLOSE, idx));
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        // force x[0],x[1],...,x[idx-1] to l
                        index_t j = 0;
                        for(; j < idx; j++) {
                            if(!lister_in_l(t, x[j])) {
                                if(t->l_size >= t->k)
                                    break;
                                lister_push_l(t, x[j]);
                                lister_push(t, 
                                            to_command_idx(M_REWIND_L, x[j]));
                            }
                        }
                        if(j == idx)
                            lister_push(t, to_command(M_QUERY));
                    }
                }
            }
            break;
        case M_CLOSE:
            assert(idx > 0);
            lister_push(t, to_command_idx(M_OPEN, idx-1));
            break;
        case M_REWIND_U:
            bitset(t->u, idx, 1);
            t->u_size++;
            break;
        case M_REWIND_L:
            lister_pop_l(t);
            break;
        }
    }
    lister_push(t, to_command(M_QUERY));
    return 0;
}

/******************************************************** Root query builder. */

motifq_t *root_build(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n = g->num_vertices;
    index_t m = 2*g->num_edges;
    index_t *pos = alloc_idxtab(n);
    index_t *adj = alloc_idxtab(n+m);
    index_t ns = k;
    shade_map_t *shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);

    motifq_t *root = (motifq_t *) MALLOC(sizeof(motifq_t));
    root->is_stub = 0;
    root->n       = g->num_vertices;
    root->k       = k;
    root->pos     = pos;
    root->adj     = adj;
    root->nl      = 0;
    root->l       = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns      = ns;
    root->shade   = shade;

    push_time();
    fprintf(stdout, "root build ... ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < m; j++) {
            index_t u = e[j];
            if(start <= u && u <= stop)                
                pos[u]++; // I am responsible for u, record adjacency to u
        }
    }
#else
    for(index_t j = 0; j < m; j++)
        pos[e[j]]++;
#endif

    index_t run = prefixsum(n, pos, 1);
    assert(run == n+m);
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        adj[pos[u]] = 0;

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < m; j+=2) {
            index_t u0 = e[j+0];
            index_t u1 = e[j+1];
            if(start <= u0 && u0 <= stop) {
                // I am responsible for u0, record adjacency to u1
                index_t pu0 = pos[u0];
                adj[pu0 + 1 + adj[pu0]++] = u1;
            }
            if(start <= u1 && u1 <= stop) {
                // I am responsible for u1, record adjacency to u0
                index_t pu1 = pos[u1];
                adj[pu1 + 1 + adj[pu1]++] = u0;
            }
        }
    }
#else
    for(index_t j = 0; j < m; j+=2) {
        index_t u0 = e[j+0];
        index_t u1 = e[j+1];
        index_t p0 = pos[u0];
        index_t p1 = pos[u1];       
        adj[p1 + 1 + adj[p1]++] = u0;
        adj[p0 + 1 + adj[p0]++] = u1;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    adjsort(n, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
//        fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }
    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

/****************************************************** Input reader (ASCII). */

void skipws(FILE *in)
{
    int c;
    do {
        c = fgetc(in);
        if(c == '#') {
            do {
                c = fgetc(in);
            } while(c != EOF && c != '\n');
        }
    } while(c != EOF && isspace(c));
    if(c != EOF)
        ungetc(c, in);
}

#define CMD_NOP          0
#define CMD_TEST_UNIQUE  1
#define CMD_TEST_COUNT   2
#define CMD_RUN_ORACLE   3
#define CMD_LIST_FIRST   4
#define CMD_LIST_ALL     5

char *cmd_legend[] = { "no operation", "test unique", "test count", "run oracle", "list first", "list all" };

void reader_ascii(FILE *in, 
                  graph_t **g_out, index_t *k_out, index_t **kk_out, 
                  index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t n = 0;
    index_t m = 0;
    graph_t *g = (graph_t *) 0;
    index_t i, j, d, k;
    index_t *kk = (index_t *) 0;
    index_t cmd = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    skipws(in);
    while(!feof(in)) {
        skipws(in);
        int c = fgetc(in);
        switch(c) {
        case 'p':
            if(g != (graph_t *) 0)
                ERROR("duplicate parameter line");
            skipws(in);
            if(fscanf(in, "motif %ld %ld", &n, &m) != 2)
                ERROR("invalid parameter line");
            if(n <= 0 || m < 0) 
                ERROR("invalid input parameters (n = %ld, m = %ld)", n, m);
            g = graph_alloc(n);
            break;
        case 'e':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before edges");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &j) != 2)
                ERROR("invalid edge line");
            if(i < 1 || i > n || j < 1 || j > n)
                ERROR("invalid edge (i = %ld, j = %ld with n = %ld)", 
                      i, j, n);
            graph_add_edge(g, i-1, j-1);
            break;
        case 'n':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before vertex colors");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &d) != 2)
                ERROR("invalid color line");
            if(i < 1 || i > n || d < 1)
                ERROR("invalid color line (i = %ld, d = %ld with n = %ld)", 
                      i, d, n);
            graph_set_color(g, i-1, d-1);
            break;
        case 'k':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before motif");
            skipws(in);
            if(fscanf(in, "%ld", &k) != 1)
                ERROR("invalid motif line");
            if(k < 1 || k > n)
                ERROR("invalid motif line (k = %ld with n = %d)", k, n);
            kk = alloc_idxtab(k);
            for(index_t u = 0; u < k; u++) {
                skipws(in);
                if(fscanf(in, "%ld", &i) != 1)
                    ERROR("error parsing motif line");
                if(i < 1)
                    ERROR("invalid color on motif line (i = %ld)", i);
                kk[u] = i-1;
            }
            break;
        case 't':
            if(g == (graph_t *) 0 || kk == (index_t *) 0)
                ERROR("parameter and motif lines must be given before test");
            skipws(in);
            {
                char cmdstr[128];
                if(fscanf(in, "%100s", cmdstr) != 1)
                    ERROR("invalid test command");
                if(!strcmp(cmdstr, "unique")) {
                    cmd_args = alloc_idxtab(k);
                    for(index_t u = 0; u < k; u++) {
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 1 || i > n)
                            ERROR("invalid test line entry (i = %ld)", i);
                        cmd_args[u] = i-1;
                    }
                    heapsort_indext(k, cmd_args);
                    for(index_t u = 1; u < k; u++)
                        if(cmd_args[u-1] >= cmd_args[u])
                            ERROR("test line contains duplicate entries");
                    cmd = CMD_TEST_UNIQUE;
                } else {
                    if(!strcmp(cmdstr, "count")) {
                        cmd_args = alloc_idxtab(1);
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 0)
                            ERROR("count on test line cannot be negative");
                        cmd = CMD_TEST_COUNT;
                        cmd_args[0] = i;
                    } else {
                        ERROR("unrecognized test command \"%s\"", cmdstr);
                    }
                }
            }
            break;
        case EOF:
            break;
        default:
            ERROR("parse error");
        }
    }

    if(g == (graph_t *) 0)
        ERROR("no graph given in input");
    if(kk == (index_t *) 0)
        ERROR("no motif given in input");

    for(index_t i = 0; i < n; i++) {
        if(g->colors[i] == -1)
            ERROR("no color assigned to vertex i = %ld", i);
    }
    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");    
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}

/***************************************************** Input reader (binary). */

#define BIN_MAGIC 0x1234567890ABCDEFUL

void reader_bin(FILE *in, 
                graph_t **g_out, index_t *k_out, index_t **kk_out, 
                index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t magic = 0;
    index_t n = 0;
    index_t m = 0;
    graph_t *g = (graph_t *) 0;
    index_t k = 0;
    index_t has_target = 0;
    index_t *kk = (index_t *) 0;
    index_t cmd = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    
    if(fread(&magic, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(magic != BIN_MAGIC)
        ERROR("error reading input");
    if(fread(&n, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(fread(&m, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(n >= 0 && m >= 0 && m%2 == 0);
    g = graph_alloc(n);
    index_t *e = graph_edgebuf(g, m/2);
    if(fread(e, sizeof(index_t), m, in) != m)
        ERROR("error reading input");
    if(fread(g->colors, sizeof(index_t), n, in) != n)
        ERROR("error reading input");
    if(fread(&has_target, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(has_target == 0 || has_target == 1);
    if(has_target) {
        if(fread(&k, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");
        assert(k >= 0);
        kk = alloc_idxtab(k);
        if(fread(kk, sizeof(index_t), k, in) != k)
            ERROR("error reading input");         
        if(fread(&cmd, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");         
        switch(cmd) {
        case CMD_NOP:
            break;
        case CMD_TEST_UNIQUE:
            cmd_args = alloc_idxtab(k);
            if(fread(cmd_args, sizeof(index_t), k, in) != k)
                ERROR("error reading input");         
            shellsort(k, cmd_args);
            break;          
        case CMD_TEST_COUNT:
            cmd_args = alloc_idxtab(1);
            if(fread(cmd_args, sizeof(index_t), 1UL, in) != 1UL)
                ERROR("error reading input");                         
            break;          
        default:
            ERROR("invalid command in binary input stream");
            break;          
        }
    }

    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}


/******************************************************* Program entry point. */

int main(int argc, char **argv)
{
    GF_PRECOMPUTE;

    push_time();
    push_memtrack();
    
    index_t arg_cmd = CMD_NOP;
    index_t have_seed = 0;
    index_t seed = 123456789;
    for(index_t f = 1; f < argc; f++) {
        if(argv[f][0] == '-') {
            if(!strcmp(argv[f], "-bin")) {
                flag_bin_input = 1;
            }
            if(!strcmp(argv[f], "-ascii")) {
                flag_bin_input = 0;
            }
            if(!strcmp(argv[f], "-oracle")) {
                arg_cmd = CMD_RUN_ORACLE;
            }
            if(!strcmp(argv[f], "-first")) {
                arg_cmd = CMD_LIST_FIRST;
            }
            if(!strcmp(argv[f], "-all")) {
                arg_cmd = CMD_LIST_ALL;
            }
            if(!strcmp(argv[f], "-seed")) {
                if(f == argc - 1)
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1;
            }
        }
    }
    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    if(have_seed == 0) {
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }
    fprintf(stdout, "random seed = %ld\n", seed);
    
    srand(seed); 

    graph_t *g;
    index_t k;
    index_t *kk;
    index_t input_cmd;
    index_t *cmd_args;
    if(flag_bin_input) {
        reader_bin(stdin, &g, &k, &kk, &input_cmd, &cmd_args);
    } else {
        reader_ascii(stdin, &g, &k, &kk, &input_cmd, &cmd_args);
    }
    index_t cmd = input_cmd;  // by default execute command in input stream
    if(arg_cmd != CMD_NOP)
        cmd = arg_cmd;        // override command in input stream

    motifq_t *root = root_build(g, k, kk);
    graph_free(g);
    FREE(kk);

    fprintf(stdout, "command: %s\n", cmd_legend[cmd]);
    fflush(stdout);

    push_time();
    switch(cmd) {
    case CMD_NOP:
        motifq_free(root);
        break;
    case CMD_TEST_UNIQUE:
        {
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            while(lister_run(t, get)) {
                assert(ct == 0);
                fprintf(stdout, "found %ld: ", ct);
                for(index_t i = 0; i < k; i++)
                    fprintf(stdout, "%ld%s", get[i], i == k-1 ? "\n" : " ");
                for(index_t l = 0; l < k; l++)
                    assert(get[l] == cmd_args[l]);
                ct++;
            }
            assert(ct == 1);
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_LIST_FIRST:
    case CMD_LIST_ALL:
    case CMD_TEST_COUNT:
        {
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            while(lister_run(t, get)) {
                fprintf(stdout, "found %ld: ", ct);
                for(index_t i = 0; i < k; i++)
                    fprintf(stdout, "%ld%s", get[i], i == k-1 ? "\n" : " ");
                ct++;
                if(cmd == CMD_LIST_FIRST)
                    break;
            }
            if(cmd == CMD_TEST_COUNT) {
                fprintf(stdout, "count = %ld, target = %ld\n", ct, cmd_args[0]);
                assert(ct == cmd_args[0]);
            }
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_RUN_ORACLE:
        fprintf(stdout, "oracle: ");
        fflush(stdout);
        if(motifq_execute(root))
            fprintf(stdout, " -- true\n");
        else
            fprintf(stdout, " -- false\n");
        motifq_free(root);
        break;
    default:
        assert(0);
        break;
    }
    double time = pop_time();
    fprintf(stdout, "command done [%.2lf ms]\n", time);
    if(input_cmd != CMD_NOP)
        FREE(cmd_args);

    time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, 
            "build: %s, %s, %s, %ld x %s\n",
#ifdef BUILD_PARALLEL
            "multithreaded",
#else
            "single thread",
#endif
#ifdef BUILD_PREFETCH
            "prefetch",
#else
            "no prefetch",
#endif
            GENF_TYPE,
            LIMBS_IN_LINE,
            LIMB_TYPE);
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);

    return 0;
}
