/* only for the macOS build check without an Apple SDK: what the Ooura FFT uses */
#ifndef NULL
#define NULL ((void *)0)
#endif
typedef unsigned long size_t;
void *malloc(size_t);
void free(void *);
void exit(int);
