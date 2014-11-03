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

/******************************************************* Common subroutines. */

typedef long int index_t;

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

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

#define MALLOC(id, x) malloc_wrapper(id, x)
#define FREE(id, x) free_wrapper(id, x)

index_t malloc_balance = 0;

void *malloc_wrapper(const char *id, size_t size)
{
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;
#ifdef MEM_INVENTORY
    fprintf(stdout, "alloc: %10s %7.3lf GiB\n", id, inGiB(size));
    fflush(stdout);
#endif
    return p;
}

void free_wrapper(const char *id, void *p)
{
    free(p);
    malloc_balance--;
#ifdef MEM_INVENTORY
    fprintf(stdout, "free: %10s\n", id);
    fflush(stdout);
#endif
}

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


/************************************************ Rudimentary graph builder. */

typedef struct 
{
    index_t num_vertices;
    index_t num_edges;
    index_t edge_capacity;
    index_t *edges;
    index_t *colors;
    index_t has_target;
    index_t motif_size;
    index_t *motif_counts;
    index_t match_count;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC("a", sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE("was", was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    index_t i;
    graph_t *g = (graph_t *) MALLOC("g", sizeof(graph_t));
    g->num_vertices = n;
    g->num_edges = 0;
    g->edge_capacity = 1000;
    g->edges = enlarge(2*g->edge_capacity, 0, (void *) 0);
    g->colors = (index_t *) MALLOC("g->colors", sizeof(index_t)*n);
    for(i = 0; i < n; i++)
        g->colors[i] = 0;
    g->has_target = 0;
    return g;
}

void graph_free(graph_t *g)
{
    if(g->has_target)
        FREE("g->motif_counts", g->motif_counts);
    FREE("g->edges", g->edges);
    FREE("g->colors", g->colors);
    FREE("g", g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v)
{
    assert(u >= 0 && 
           v >= 0 && 
           u != v && 
           u < g->num_vertices &&
           v < g->num_vertices);
    assert(g->has_target == 0);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(4*g->edge_capacity, 2*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 2*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
    shellsort(2, e);
}

void graph_set_color(graph_t *g, index_t u, index_t c)
{
    assert(u >= 0 && u < g->num_vertices && c >= 0);
    assert(g->has_target == 0);
    g->colors[u] = c;
}

index_t graph_max_color(graph_t *g)
{
    index_t i;
    index_t m = 0;
    for(i = 0; i < g->num_vertices; i++)
        if(m < g->colors[i])
            m = g->colors[i];
    return m;
}

void graph_set_target(graph_t *g, index_t k, index_t *f, index_t c) 
{
    g->has_target = 1;
    g->motif_size = k;
    g->match_count = c;
    index_t s = graph_max_color(g) + 1;
    g->motif_counts = (index_t *) MALLOC("g->motif_counts", sizeof(index_t)*s);
    for(index_t j = 0; j < s; j++)
        g->motif_counts[j] = f[j];
}

void graph_out_dimacs(FILE *out, graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t *e = g->edges;
    index_t *c = g->colors;
    fprintf(out, "p motif %ld %ld\n", (long) n, (long) m);
    for(index_t i = 0; i < m; i++)
        fprintf(out, "e %ld %ld\n", (long) e[2*i+0]+1, (long) e[2*i+1]+1);
    for(index_t i = 0; i < n; i++)
        fprintf(out, "n %ld %ld\n", (long) i+1, (long) c[i]+1);
    if(g->has_target) {
        fprintf(out, "k %ld", g->motif_size);
        index_t s = graph_max_color(g) + 1;
        for(index_t i = 0; i < s; i++) {
            for(index_t j = 0; j < g->motif_counts[i]; j++)
                fprintf(out, " %ld", i+1);
        }
        fprintf(out, "\n");
        fprintf(out, "t count %ld\n", g->match_count);
    }
}

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand()); // dirty hack
}

void randperm(index_t n, index_t *p)
{
    index_t i;
    for(i = 0; i < n; i++)
        p[i] = i;
    for(i = 0; i < n-1; i++) {
        index_t x = i+(irand()%(n-i));
        index_t t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

graph_t *graph_shuffle(graph_t *g, index_t seed)
{
    srand(seed);
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t *p = (index_t *) MALLOC("p", sizeof(index_t)*n);
    index_t *q = (index_t *) MALLOC("q", sizeof(index_t)*m);
    randperm(n, p);
    randperm(m, q);

    graph_t *s = graph_alloc(n);

    for(index_t j = 0; j < m; j++)
        graph_add_edge(s, p[g->edges[2*j+0]], p[g->edges[2*j+1]]);

    for(index_t i = 0; i < n; i++)
        graph_set_color(s, p[i], g->colors[i]);

    if(g->has_target)
        graph_set_target(s, g->motif_size, g->motif_counts, g->match_count);
    

    FREE("p", p);
    FREE("q", q);
    return s;
}


/************************************************** Test graph generator(s). */

/* An ugly hack to get test trees from Pruefer encoding. */

void baserep(index_t n, index_t t, index_t *b, index_t *d)
{
    index_t u;
    index_t q;
    index_t r;
    index_t i;

    u = 1;
    for(i = 1; i <= n; i++)
        u = u * b[i];
    if(t < 0 || t >= u)
        ERROR("out of bounds (n = %ld, t = %ld, u = %ld)", (long) n, (long) t, (long) u);
    u = u/b[n];
    q = t/u;
    r = t-q*u;
    d[n] = q;
    if(n > 1) {
        baserep(n-1, r, b, d);
    } else {
        if(r != 0)
            ERROR("improper termination");
    }
}

void urktree(index_t n, index_t *x, index_t *e)
{
    /* Unrank from Pruefer code to edge list. */

    index_t d[128];
    index_t i, j;
    index_t a, b;
    for(i = 1; i <= n; i++)
        d[i] = 1;
    for(i = 1; i <= n-2; i++)
        d[x[i]]++;
    for(i = 1; i <= n-2; i++) {
        for(j = 1; j <= n; j++)
            if(d[j] == 1)
                break;
        if(j > n)
            ERROR("cannot find vertex of degree 1");
        a = x[i];
        b = j;
        if(a > b) {
            j = a;
            a = b;
            b = j;
        }        
        e[2*i-1] = a;
        e[2*i] = b;
        d[a]--;
        d[b]--;
    }
    for(j = 1; j <= n; j++)
        if(d[j] == 1)
            break;
    if(j > n)
        ERROR("cannot find vertex of degree 1");
    a = j;
    for(j++; j <= n; j++)
        if(d[j] == 1)
            break;
    if(j > n)
        ERROR("cannot find vertex of degree 1");
    b = j;
    if(a > b) {
        j = a;
        a = b;
        b = j;
    }        
    e[2*(n-1)-1] = a;
    e[2*(n-1)] = b;
}

graph_t *pruefertree(index_t n, index_t t)
{
    index_t i;
    index_t x[1024];
    index_t e[1024];
    index_t b[1024];
    index_t d[1024];
    graph_t *g = graph_alloc(n);

    for(i = 1; i <= n-2; i++)
        b[i] = n;
    baserep(n-2, t-1, b, d);
    for(i = 1; i <= n-2; i++)
        x[i] = d[n-2+1-i] + 1;
    urktree(n, x, e);
    for(i = 1; i <= n-1; i++)
        graph_add_edge(g, e[2*i-1]-1, e[2*i]-1);
    return g;
}

index_t pruefermax(index_t n)
{
    index_t r = 1;
    for(index_t i = 0; i < n-2; i++)
        r*=n;
    return r;
}



/****************************************************** Pruefer ensemble. */

/* Contains d^{d-2} vertex-disjoint labeled trees, each tree on 
 * its unique set of d consecutive vertices. */

graph_t *pruefer_ensemble(index_t d)
{
    index_t p = pruefermax(d);
    graph_t *g = graph_alloc(p*d);
    for(index_t r = 1; r <= p; r++) {
        graph_t *t = pruefertree(d, r);
        for(index_t i = 0; i < d-1; i++)
            graph_add_edge(g, d*(r-1)+t->edges[2*i], d*(r-1)+t->edges[2*i+1]);
        graph_free(t);
    }
    return g;
}

graph_t *pruefer_ensemble_monochromatic(index_t d)
{
    graph_t *g = pruefer_ensemble(d);
    index_t p = pruefermax(d);
    for(index_t r = 1; r <= p; r++)
        for(index_t i = 0; i < d; i++)
            graph_set_color(g, d*(r-1)+i, 0);
    index_t f[128];
    for(index_t j = 0; j < d; j++)
        f[j] = (j == 0) ? d : 0;
    graph_set_target(g, d, f, p);
    return g;
}

graph_t *pruefer_ensemble_rainbow(index_t d)
{
    graph_t *g = pruefer_ensemble(d);
    index_t p = pruefermax(d);
    for(index_t r = 1; r <= p; r++)
        for(index_t i = 0; i < d; i++)
            graph_set_color(g, d*(r-1)+i, i);
    index_t f[128];
    for(index_t j = 0; j < d; j++)
        f[j] = 1;
    graph_set_target(g, d, f, p);
    return g;
}

graph_t *pruefer_ensemble_mixed(index_t d)
{
    graph_t *g = pruefer_ensemble(d);
    index_t p = pruefermax(d);
    for(index_t r = 1; r <= p; r++)
        for(index_t i = 0; i < d; i++)
            graph_set_color(g, d*(r-1)+i, i/((d+1)/2));
    index_t f[128];
    f[0] = (d+1)/2;
    f[1] = d-f[0];
    graph_set_target(g, d, f, p);
    return g;
}

/********************************************************* Complete graphs. */


graph_t *graph_complete(index_t n, index_t b)
{
    graph_t *g = graph_alloc(n);
    for(index_t i = 0; i < n; i++)
        for(index_t j = i+1; j < n; j++)
            graph_add_edge(g, i, j);
    for(index_t i = 0; i < n; i++)
        graph_set_color(g, i, i/b);
    return g;
}

index_t binom(index_t n, index_t k)
{
    index_t i;
    long long int r = 1;
    if(n < 0 || k < 0)
        return 0;
    for(i = 0; i < k; i++)
        r = r*(n-i);
    for(i = 0; i < k; i++)
        r = r/(k-i);
    return (index_t) r;
}


graph_t *graph_complete_mono(index_t k)
{
    index_t n = 10;
    assert(k >= 1 && k <= n);
    graph_t *g = graph_complete(n, n);
    index_t f[128];
    f[0] = k;
    index_t c = binom(n,k);
    graph_set_target(g, k, f, c);
    return g;
}

graph_t *graph_complete_rainbow(index_t k)
{
    index_t n = 3*k;
    assert(k >= 1 && k <= n && k <= 128);
    graph_t *g = graph_complete(n, n/k);
    index_t f[128];
    index_t c = 1;
    for(index_t i = 0; i < k; i++) {
        f[i] = 1;
        c *= 3;
    }
    graph_set_target(g, k, f, c);
    return g;
}

graph_t *graph_complete_mixed(index_t k)
{
    index_t q = k/2;
    index_t r = k-2*q;
    index_t n = 4*(q+r);
    assert(k > 0 && k < n);
    graph_t *g = graph_complete(n, 4);
    index_t f[128];

    index_t c = 1;
    for(index_t i = 0; i < q; i++) {
        f[i] = 2;
        c *= 6;
    }
    if(r == 1) {
        f[k/2] = 1;
        c *= 4;
    }
    graph_set_target(g, k, f, c);
    return g;
}


/****************************************************** Program entry point. */

int main(int argc, char **argv)
{

    srand(123);

    if(argc < 4) {
        fprintf(stdout, 
                "usage: %s <type> <order> <mono/rainbow/mixed> [shuffle <seed>]\n"
                "available types: pruefer  (order at least 3)\n"
                "                 complete (order at least 1)\n",
                argv[0]);
        return 0;
    }

    fprintf(stderr, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stderr, " %s", argv[f]);
    fprintf(stderr, "\n");    

    graph_t *g = (graph_t *) 0;

    index_t order = atol(argv[2]);
    char *type = argv[1];
    char *colortype = argv[3];
    if(!strcmp("pruefer", type)) {
        assert(order >= 3);     
        if(!strcmp("mono", colortype))
            g = pruefer_ensemble_monochromatic(order);
        if(!strcmp("rainbow", colortype))
            g = pruefer_ensemble_rainbow(order);
        if(!strcmp("mixed", colortype))
            g = pruefer_ensemble_mixed(order);
        if(g == (graph_t *) 0) {
            fprintf(stderr, "invalid coloring parameter: %s\n", colortype);
            return 1;
        }
    }
    if(!strcmp("complete", type)) {
        if(!strcmp("mono", colortype))
            g = graph_complete_mono(order);
        if(!strcmp("rainbow", colortype))
            g = graph_complete_rainbow(order);
        if(!strcmp("mixed", colortype))
            g = graph_complete_mixed(order);
        if(g == (graph_t *) 0) {
            fprintf(stderr, "invalid coloring parameter: %s\n", colortype);
            return 1;
        }
   }
    if(g == (graph_t *) 0) {
        fprintf(stderr, "unknown type: %s\n", type);
        return 1;
    }

    if(argc > 4) {
        if(strcmp("shuffle", argv[4])) {
            fprintf(stderr, "unknown parameter: %s\n", argv[4]);
            return 1;
        }
        if(argc < 6) {
            fprintf(stderr, "shuffle seed missing\n");
            return 1;
        } 
        index_t seed = atol(argv[5]);

        graph_t *sg = graph_shuffle(g, seed);
        graph_free(g);
        g = sg;
    }

    fprintf(stderr, "gen-count [%s,%s]: n = %ld, m = %ld, k = %ld\n", 
            type, 
            colortype,
            g->num_vertices,
            g->num_edges,
            g->has_target == 1 ? g->motif_size : -1);

    graph_out_dimacs(stdout, g);
    graph_free(g);

    return 0;
}
