CXX=g++
CPPFLAGS=
CXXFLAGS=-O3 -std=c++17 -fopenmp
LNKFLAGS=-lpthread -ltbb

# First target is the default target
all: problem1 problem2

# $@ is a variable that contains the target name.
problem1: problem1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<) $(LNKFLAGS)

problem2: problem2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<) $(LNKFLAGS)

# The target is not a filename
clean:
	rm *.out compile_commands.json
