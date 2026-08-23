#include <cblas.h>

void MY_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    openblas_set_num_threads(1);
    double alpha = 1.0;
    double beta = 1.0;

    // OpenBLAS CBLAS call
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}