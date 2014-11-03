#
# This file is part of an experimental software implementation of
# graph motif search utilizing the constrained multilinear 
# sieving framework; cf. 
#
#   A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri,
#   "Engineering motif search for large graphs",
#   ALENEX15 Meeting on Algorithm Engineering and Experiments,
#   5 January 2015, San Diego, CA.
#
# This experimental source code is supplied to accompany the 
# aforementioned paper. 
#
# The source code is configured for a gcc build to a native 
# microarchitecture that must support the AVX2 and PCLMULQDQ 
# instruction set extensions. Other builds are possible but 
# require manual configuration of 'Makefile' and 'builds.h'.
#
# The source code is subject to the following license.
#
# The MIT License (MIT)
#
# Copyright (c) 2014 A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

MAKE = make
CC = gcc
CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp

all: gen-count gen-unique lister

gen-count: gen-count.c
	$(CC) $(CFLAGS) -o gen-count gen-count.c

gen-unique: gen-unique.c ffprng.h
	$(CC) $(CFLAGS) -o gen-unique gen-unique.c -lm

lister: lister.c builds.h gf.h ffprng.h
	$(CC) $(CFLAGS) -DLISTER_DEFAULT -o lister lister.c

experiment-bin/Makefile: mkmk.pl
	mkdir experiment-bin
	perl mkmk.pl >experiment-bin/Makefile

expbin: experiment-bin/Makefile
	cd experiment-bin && $(MAKE) && cd ..

clean:	
	rm -f *.o *.a *~ gen-count gen-unique lister
	cd experiment-bin && $(MAKE) clean && cd ..
