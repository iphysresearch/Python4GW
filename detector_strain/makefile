#$Id: makefile,v 1.1 2013/08/20 11:49:54 zjcao Exp $

#filein  = -I/opt/local/include
filein  = -I/usr/include

CC  = cc

.SUFFIXES: .o .c

.c.o:
	${CC} -c $< $(filein) -o $@



# projects
MYdetector_strain: MYdetector_strain.o MYfuns.o MYgetopt.o 
	$(CC) -o $@ MYdetector_strain.o MYfuns.o MYgetopt.o 


clean:
	rm -f *.o MYdetector_strain