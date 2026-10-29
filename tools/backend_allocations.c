// DYLD-only allocation audit. Not injected during performance measurements.
#include <stddef.h>
#include <stdlib.h>
static _Thread_local int active;
static _Thread_local size_t calls;
void timestretch_qa_alloc_begin(void) {
  calls = 0;
  active = 1;
}
size_t timestretch_qa_alloc_end(void) {
  active = 0;
  return calls;
}
static void *probe_malloc(size_t n) {
  if (active)
    ++calls;
  return malloc(n);
}
static void *probe_calloc(size_t n, size_t s) {
  if (active)
    ++calls;
  return calloc(n, s);
}
static void *probe_realloc(void *p, size_t n) {
  if (active)
    ++calls;
  return realloc(p, n);
}
static int probe_memalign(void **p, size_t a, size_t n) {
  if (active)
    ++calls;
  return posix_memalign(p, a, n);
}
static void *probe_aligned(size_t a, size_t n) {
  if (active)
    ++calls;
  return aligned_alloc(a, n);
}
#define INTERPOSE(replacement, original)                                       \
  __attribute__((used)) static struct {                                        \
    const void *replacement;                                                   \
    const void *original;                                                      \
  } pair_##original __attribute__((section("__DATA,__interpose"))) = {         \
      (const void *)&replacement, (const void *)&original}
INTERPOSE(probe_malloc, malloc);
INTERPOSE(probe_calloc, calloc);
INTERPOSE(probe_realloc, realloc);
INTERPOSE(probe_memalign, posix_memalign);
INTERPOSE(probe_aligned, aligned_alloc);
