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

@cfgs=("LISTER_1x64_UNROLL",
       "LISTER_4x64_UNROLL",
       "LISTER_8x64_UNROLL",
       "LISTER_1x64_PACK",
       "LISTER_4x64_PACK",
       "LISTER_8x64_PACK",
       "LISTER_1x8_EXPLOG",
       "LISTER_32x8_EXPLOG",
       "LISTER_32x8_SLICE",
       "LISTER_64x8_SLICE",
       "LISTER_32x8_PACK",
       "LISTER_64x8_PACK",
       "LISTER_32x8_MEMSAVE");

print "MAKE = make\n".
      "CC = gcc\n".
      "CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp\n";
print "\n";

print "root: all\n\n";

foreach $cfg (@cfgs) {
    for($par=0;$par<=1;$par++) {
	for($pre=0;$pre<=1;$pre++) {
	    $target = $cfg;
	    @dfs=();
	    push @dfs, ("-D$cfg");
	    if($par==1) {
		push @dfs, ("-DBUILD_PARALLEL");
		$target = $target . "_PAR";
	    }
	    if($pre==1) {
		push @dfs, ("-DBUILD_PREFETCH");
		$target = $target . "_PRE";
	    }
	    $defs = join(' ',@dfs);
	    print "$target: ../lister.c ../builds.h ../gf.h ../ffprng.h\n".
		  "\t\$(CC) \$(CFLAGS) $defs -o $target ../lister.c\n".
                  "\n";
	    push @targets, ($target);
	}
    }
}

print "all: ".join(' ',@targets)."\n\n";

print "clean:\n\trm -f ".join(' ',@targets)."\n\n";
