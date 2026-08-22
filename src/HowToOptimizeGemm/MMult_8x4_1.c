/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

/* Block sizes */
#define mc 512
#define kc 512
#define nb 1600

#define min(i, j) ((i) < (j) ? (i) : (j))

/* Routine for computing C = A * B + C */

void AddDot8x4(int, double *, int, double *, int, double *, int);
void PackMatrixA(int, double *, int, double *);
void PackMatrixB(int, double *, int, double *);
void InnerKernel(int, int, int, double *, int, double *, int, double *, int, int);

void MY_MMult(int m, int n, int k, double *a, int lda,
              double *b, int ldb,
              double *c, int ldc)
{
  int i, p, pb, ib;

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */

  for (p = 0; p < k; p += kc)
  {
    pb = min(k - p, kc);
    for (i = 0; i < m; i += mc)
    {
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, i == 0);
    }
  }
}

void InnerKernel(int m, int n, int k, double *a, int lda,
                 double *b, int ldb,
                 double *c, int ldc, int first_time)
{
  int i, j;
  double
      packedA[m * k] __attribute__((aligned(64)));
  static double
      packedB[kc * nb] __attribute__((aligned(64))); /* Note: using a static buffer is not thread safe... */

  for (j = 0; j < n; j += 4)
  { /* Loop over the columns of C, unrolled by 4 */
    if (first_time)
      PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]);
    for (i = 0; i < m; i += 8)
    { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
   one routine (four inner products) */
      if (j == 0)
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      // Prefetch the start of the NEXT block of A before calling the microkernel
      // if (i + 4 < m) {
      //     __builtin_prefetch(&packedA[(i + 4) * k], 0, 0);
      // }
      AddDot8x4(k, &packedA[i * k], 8, &packedB[j * k], k, &C(i, j), ldc);
    }
  }
}

void PackMatrixA(int k, double *a, int lda, double *a_to)
{
  int j;

  for (j = 0; j < k; j++)
  { /* loop over columns of A */
    double
        *a_ij_pntr = &A(0, j);

    *a_to = *a_ij_pntr;
    *(a_to + 1) = *(a_ij_pntr + 1);
    *(a_to + 2) = *(a_ij_pntr + 2);
    *(a_to + 3) = *(a_ij_pntr + 3);

    a_to += 4;
  }
}

void PackMatrixB(int k, double *b, int ldb, double *b_to)
{
  int i;
  double
      *b_i0_pntr = &B(0, 0),
      *b_i1_pntr = &B(0, 1),
      *b_i2_pntr = &B(0, 2), *b_i3_pntr = &B(0, 3);

  for (i = 0; i < k; i++)
  { /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

// #include "sse2neon.h" // SSE2NEON
#include <arm_neon.h> // ARM NEON intrinsics

#include <arm_neon.h> // ARM NEON intrinsics

typedef union
{
  float64x2_t v;
  double d[2];
} v2df_t;

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int p;
  
  // 16 Accumulator registers for the 8x4 tile of C
  v2df_t c_00_c_10, c_01_c_11, c_02_c_12, c_03_c_13;
  v2df_t c_20_c_30, c_21_c_31, c_22_c_32, c_23_c_33;
  v2df_t c_40_c_50, c_41_c_51, c_42_c_52, c_43_c_53;
  v2df_t c_60_c_70, c_61_c_71, c_62_c_72, c_63_c_73;

  // 4 Registers for Matrix A (1 column of 8 doubles)
  v2df_t a_0p_a_1p, a_2p_a_3p, a_4p_a_5p, a_6p_a_7p;

  // 4 Registers for Matrix B (1 row of 4 doubles, duplicated)
  v2df_t b_p0, b_p1, b_p2, b_p3;

  // Initialize accumulators to zero
  c_00_c_10.v = vdupq_n_f64(0.0); c_01_c_11.v = vdupq_n_f64(0.0);
  c_02_c_12.v = vdupq_n_f64(0.0); c_03_c_13.v = vdupq_n_f64(0.0);
  c_20_c_30.v = vdupq_n_f64(0.0); c_21_c_31.v = vdupq_n_f64(0.0);
  c_22_c_32.v = vdupq_n_f64(0.0); c_23_c_33.v = vdupq_n_f64(0.0);
  c_40_c_50.v = vdupq_n_f64(0.0); c_41_c_51.v = vdupq_n_f64(0.0);
  c_42_c_52.v = vdupq_n_f64(0.0); c_43_c_53.v = vdupq_n_f64(0.0);
  c_60_c_70.v = vdupq_n_f64(0.0); c_61_c_71.v = vdupq_n_f64(0.0);
  c_62_c_72.v = vdupq_n_f64(0.0); c_63_c_73.v = vdupq_n_f64(0.0);

  // The K loop
  for (p = 0; p < k; p+=2)
  {
    // Load 8 contiguous elements from packed A
    a_0p_a_1p.v = vld1q_f64(a);
    a_2p_a_3p.v = vld1q_f64(a + 2);
    a_4p_a_5p.v = vld1q_f64(a + 4);
    a_6p_a_7p.v = vld1q_f64(a + 6);

    // Load and duplicate 4 elements from packed B
    b_p0.v = vld1q_dup_f64(b);
    b_p1.v = vld1q_dup_f64(b + 1);
    b_p2.v = vld1q_dup_f64(b + 2);
    b_p3.v = vld1q_dup_f64(b + 3);

    // 32 FMAs to compute the outer product
    c_00_c_10.v = vfmaq_f64(c_00_c_10.v, a_0p_a_1p.v, b_p0.v);
    c_20_c_30.v = vfmaq_f64(c_20_c_30.v, a_2p_a_3p.v, b_p0.v);
    c_40_c_50.v = vfmaq_f64(c_40_c_50.v, a_4p_a_5p.v, b_p0.v);
    c_60_c_70.v = vfmaq_f64(c_60_c_70.v, a_6p_a_7p.v, b_p0.v);

    c_01_c_11.v = vfmaq_f64(c_01_c_11.v, a_0p_a_1p.v, b_p1.v);
    c_21_c_31.v = vfmaq_f64(c_21_c_31.v, a_2p_a_3p.v, b_p1.v);
    c_41_c_51.v = vfmaq_f64(c_41_c_51.v, a_4p_a_5p.v, b_p1.v);
    c_61_c_71.v = vfmaq_f64(c_61_c_71.v, a_6p_a_7p.v, b_p1.v);

    c_02_c_12.v = vfmaq_f64(c_02_c_12.v, a_0p_a_1p.v, b_p2.v);
    c_22_c_32.v = vfmaq_f64(c_22_c_32.v, a_2p_a_3p.v, b_p2.v);
    c_42_c_52.v = vfmaq_f64(c_42_c_52.v, a_4p_a_5p.v, b_p2.v);
    c_62_c_72.v = vfmaq_f64(c_62_c_72.v, a_6p_a_7p.v, b_p2.v);

    c_03_c_13.v = vfmaq_f64(c_03_c_13.v, a_0p_a_1p.v, b_p3.v);
    c_23_c_33.v = vfmaq_f64(c_23_c_33.v, a_2p_a_3p.v, b_p3.v);
    c_43_c_53.v = vfmaq_f64(c_43_c_53.v, a_4p_a_5p.v, b_p3.v);
    c_63_c_73.v = vfmaq_f64(c_63_c_73.v, a_6p_a_7p.v, b_p3.v);

    // Load 8 contiguous elements from packed A
    a_0p_a_1p.v = vld1q_f64(a + 8);
    a_2p_a_3p.v = vld1q_f64(a + 10);
    a_4p_a_5p.v = vld1q_f64(a + 12);
    a_6p_a_7p.v = vld1q_f64(a + 14);

    // Load and duplicate 4 elements from packed B
    b_p0.v = vld1q_dup_f64(b + 4);
    b_p1.v = vld1q_dup_f64(b + 5);
    b_p2.v = vld1q_dup_f64(b + 6);
    b_p3.v = vld1q_dup_f64(b + 7);

    // 32 FMAs to compute the outer product
    c_00_c_10.v = vfmaq_f64(c_00_c_10.v, a_0p_a_1p.v, b_p0.v);
    c_20_c_30.v = vfmaq_f64(c_20_c_30.v, a_2p_a_3p.v, b_p0.v);
    c_40_c_50.v = vfmaq_f64(c_40_c_50.v, a_4p_a_5p.v, b_p0.v);
    c_60_c_70.v = vfmaq_f64(c_60_c_70.v, a_6p_a_7p.v, b_p0.v);

    c_01_c_11.v = vfmaq_f64(c_01_c_11.v, a_0p_a_1p.v, b_p1.v);
    c_21_c_31.v = vfmaq_f64(c_21_c_31.v, a_2p_a_3p.v, b_p1.v);
    c_41_c_51.v = vfmaq_f64(c_41_c_51.v, a_4p_a_5p.v, b_p1.v);
    c_61_c_71.v = vfmaq_f64(c_61_c_71.v, a_6p_a_7p.v, b_p1.v);

    c_02_c_12.v = vfmaq_f64(c_02_c_12.v, a_0p_a_1p.v, b_p2.v);
    c_22_c_32.v = vfmaq_f64(c_22_c_32.v, a_2p_a_3p.v, b_p2.v);
    c_42_c_52.v = vfmaq_f64(c_42_c_52.v, a_4p_a_5p.v, b_p2.v);
    c_62_c_72.v = vfmaq_f64(c_62_c_72.v, a_6p_a_7p.v, b_p2.v);

    c_03_c_13.v = vfmaq_f64(c_03_c_13.v, a_0p_a_1p.v, b_p3.v);
    c_23_c_33.v = vfmaq_f64(c_23_c_33.v, a_2p_a_3p.v, b_p3.v);
    c_43_c_53.v = vfmaq_f64(c_43_c_53.v, a_4p_a_5p.v, b_p3.v);
    c_63_c_73.v = vfmaq_f64(c_63_c_73.v, a_6p_a_7p.v, b_p3.v);


    a += 16; // Advance packed A by 16 (2 * 8)
    b += 8; // Advance packed B by 8 (2 * 4)
  }

  // Write back to C (Rows 0-3)
  C(0, 0) += c_00_c_10.d[0]; C(0, 1) += c_01_c_11.d[0]; C(0, 2) += c_02_c_12.d[0]; C(0, 3) += c_03_c_13.d[0];
  C(1, 0) += c_00_c_10.d[1]; C(1, 1) += c_01_c_11.d[1]; C(1, 2) += c_02_c_12.d[1]; C(1, 3) += c_03_c_13.d[1];
  C(2, 0) += c_20_c_30.d[0]; C(2, 1) += c_21_c_31.d[0]; C(2, 2) += c_22_c_32.d[0]; C(2, 3) += c_23_c_33.d[0];
  C(3, 0) += c_20_c_30.d[1]; C(3, 1) += c_21_c_31.d[1]; C(3, 2) += c_22_c_32.d[1]; C(3, 3) += c_23_c_33.d[1];

  // Write back to C (Rows 4-7)
  C(4, 0) += c_40_c_50.d[0]; C(4, 1) += c_41_c_51.d[0]; C(4, 2) += c_42_c_52.d[0]; C(4, 3) += c_43_c_53.d[0];
  C(5, 0) += c_40_c_50.d[1]; C(5, 1) += c_41_c_51.d[1]; C(5, 2) += c_42_c_52.d[1]; C(5, 3) += c_43_c_53.d[1];
  C(6, 0) += c_60_c_70.d[0]; C(6, 1) += c_61_c_71.d[0]; C(6, 2) += c_62_c_72.d[0]; C(6, 3) += c_63_c_73.d[0];
  C(7, 0) += c_60_c_70.d[1]; C(7, 1) += c_61_c_71.d[1]; C(7, 2) += c_62_c_72.d[1]; C(7, 3) += c_63_c_73.d[1];
}