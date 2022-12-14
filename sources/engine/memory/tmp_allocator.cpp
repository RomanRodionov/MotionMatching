#include "tmp_allocator.h"
#include <eastl/vector.h>
#include <common.h>
#include <assert.h>

#define TMP_ALLOCATOR_LOG 0

struct memory_slice
{
  char *memory;
  size_t size, used;
  memory_slice(size_t size):memory(new char[size]), size(size), used(0)
  {
    
  }
  memory_slice(memory_slice&&other)
  {
    memcpy(this, &other, sizeof(memory_slice));
    memset(&other, 0, sizeof(memory_slice));
  }
  memory_slice& operator=(memory_slice&&other)
  {
    memcpy(this, &other, sizeof(memory_slice));
    memset(&other, 0, sizeof(memory_slice));
    return *this;
  }
  ~memory_slice()
  {
    if (memory)
      delete[] memory;
  }
};


constexpr unsigned int DEFAULT_PAGE_SIZE = 10u << 10u;//10 kb
struct tmp_allocation_manager
{
  eastl::vector<memory_slice> memorySlices;
  int lastPage = 0;
  int allocationCount = 0, deallocationCount = 0;
  int frameId = 0;

  void* allocate(size_t n, size_t alignment)
  {
    allocationCount++;
    if (lastPage < (int)memorySlices.size())
    {
      if (!(memorySlices[lastPage].used + n + alignment < memorySlices[lastPage].size))
      {
        lastPage++;
        if (lastPage >= (int)memorySlices.size())
          memorySlices.emplace_back(DEFAULT_PAGE_SIZE);
      }
    }
    else
    {
      memorySlices.emplace_back(DEFAULT_PAGE_SIZE);
    }
    memory_slice &slice = memorySlices[lastPage];
    slice.used = (slice.used + alignment - 1) & ~(alignment - 1);
    void *mem = slice.memory + slice.used;
    slice.used += n;
    return mem;
  }
  
  void deallocate(void */* p */, size_t /* n */)
  {
    deallocationCount++;
  }

  void clear()
  {
    if (allocationCount == 0)
      return;
#if TMP_ALLOCATOR_LOG
    frameId++;
    if (frameId % 10 == 0)
    {
      debug_log("allocation per frame = %d, slices %d", allocationCount, memorySlices.size());
      for (const memory_slice &slice: memorySlices)
      {
        debug_log("memory %p, %d / %d", slice.memory, slice.used, slice.size);
      }
    }
#endif

    assert(allocationCount == deallocationCount);
    for (memory_slice &slice: memorySlices)
    {
      slice.used = 0;
    }
    lastPage = 0;
    allocationCount = deallocationCount = 0;
  }

} tmpAllocationManager;

tmp_allocator::tmp_allocator(const char* /* pName */)
{
}

void* tmp_allocator::allocate(size_t n, int /* flags */)
{
  if ((n >> 1u) < DEFAULT_PAGE_SIZE)
  {
    return tmpAllocationManager.allocate(n, 4);
  }
  else
  {
    debug_error("so huge allocation for tmp_allocator %d bytes. used default allocation", n);
    return new char[n];
  }
}

void* tmp_allocator::allocate(size_t n, size_t alignment, size_t /* offset */, int /* flags */)
{
  if ((n >> 1u) < DEFAULT_PAGE_SIZE)
  {
    return tmpAllocationManager.allocate(n, alignment);
  }
  else
  {
    debug_error("so huge allocation for tmp_allocator %d bytes. used default allocation", n);
    return new char[n];
  }
}

void tmp_allocator::deallocate(void* p, size_t n)
{
  if ((n >> 1u) < DEFAULT_PAGE_SIZE)
  {
    tmpAllocationManager.deallocate(p, n);
  }
  else
  {
    delete[] (char*)p;
  }
}

void clear_tmp_allocation()
{
  tmpAllocationManager.clear();
}