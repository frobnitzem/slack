test_tbb: tbb.cpp memspace.cpp
	/sw/bin/g++-fsf-4.8 -o $@ -I../../include -I/sw/include -L/sw/lib -L../ $^ -ltce2 -ltbb

clean:
	rm test_tbb tdot.so
