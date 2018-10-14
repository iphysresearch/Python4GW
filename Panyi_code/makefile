#$Id: makefile,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $

#include ../machines_data_base/for_compile/makefile.inc

.SUFFIXES: .o .f90 .C .for

.f90.o:
	$(f90) $(f90appflags) -c $< -o $@

.C.o:
	${CXX} $(CXXAPPFLAGS) -c $< $(filein) -o $@  $(GSLIB)

.for.o:
	$(f77) -c $< -o $@

# projects

Panyi: Panyimain.o Panyicomm.o Panyi.o
	g++ -o $@ $< Panyicomm.o Panyi.o -lgsl