/* only for the macOS build check without an Apple SDK: what the Ooura FFT uses (stderr as in Apple's libc) */
typedef struct __sFILE FILE;
extern FILE *__stderrp;
#define stderr __stderrp
int fprintf(FILE *, const char *, ...);
