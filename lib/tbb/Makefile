
CXX=/sw/bin/g++-fsf-4.8
CXXFLAGS = -I../../include -I/sw/include
LDFLAGS = -L/sw/lib -L../ $^ -lslack -ltbb

.cpp.o:
	$(CXX) -c -o $@ $(CXXFLAGS) $^

clean:
	rm test_tbb tdot.so
