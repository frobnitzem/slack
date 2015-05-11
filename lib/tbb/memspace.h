#include <cstdio>
#include <stdlib.h>

#include <tbb/mutex.h>
#include <tbb/concurrent_hash_map.h>

using namespace tbb;

// memory block info.
struct BlockInfo {
    int nref;
    size_t sz;
    int managed; // nonzero if we are responsible for free()
    void *info; // user-managed header info.
};

typedef concurrent_hash_map<void *, struct BlockInfo> MemTbl;

class MemSpace {
    public:
      //MemSpace();
      ~MemSpace();
      void *reserve(size_t sz, int refs, void *info);
      // get ref. to header info.
      void *get_info(void *x, int *nref, size_t *sz);
      void release(void *x);
      void *uniq(void *x, size_t sz);
      // If I have to, I guess.
      void insert_unmanaged(void *buf, size_t sz, void *info);
      // just remove with MemSpace::remove(buf)
    private:
      mutex lck; // for iterating over elements. (e.g. in reserve)
      MemTbl mem;
};

