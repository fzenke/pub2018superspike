# shorthand for your MPI C++ compiler
CC = mpicxx
# Your Auryn install path
AURYNDIR = $(HOME)/auryn/
# Path to libauryn.a (see compiling Auryn)
BUILDDIR = $(AURYNDIR)/build/release/

CFLAGS= -ansi -Wall -pipe -O3 -DNDEBUG -ffast-math -funsafe-math-optimizations \
        -march=native -mtune=native -pedantic \
        -I/usr/local/include -I../src -I$(AURYNDIR)/src -I$(AURYNDIR)/dev/src -I..

LDFLAGS=$(BUILDDIR)/src/libauryn.a \
		-L/usr/local/lib -lboost_program_options -lboost_serialization -lboost_mpi -lboost_system -lboost_filesystem

# # debug
# BUILDDIR = $(AURYNDIR)/build/debug/src/
# CFLAGS= -ansi -Wall -pipe -g \
#         -march=native -mtune=native -pedantic \
#         -I$(AURYNDIR)/src -I$(AURYNDIR)/dev/src


OBJECTS=SuperSpikeConnection.o RFSuperSpikeConnection.o ErrorConnection.o
SIMULATIONS=sim_symfb sim_rfb

.SECONDARY:

all: $(SIMULATIONS) 

sim_%: sim_%.o $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $< $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

%.o : ../src/%.cpp
	$(CC) $(CFLAGS) -c $<

figs: 

plot_%.tex: plot_%.gnu 
	gnuplot $< 
	epstopdf epslatex-inc.eps
	mv epslatex.out $@

%.pdf: %.tex
	pdflatex $< 

clean: 
	rm -f *.o $(SIMULATIONS)
