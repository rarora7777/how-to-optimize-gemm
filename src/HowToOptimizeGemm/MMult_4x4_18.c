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

void AddDot4x4(int, double *, int, double *, int, double *, int);
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
      packedA[m * k];
  static double
      packedB[kc * nb]; /* Note: using a static buffer is not thread safe... */

  for (j = 0; j < n; j += 4)
  { /* Loop over the columns of C, unrolled by 4 */
    if (first_time)
      PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]);
    for (i = 0; i < m; i += 4)
    { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
   one routine (four inner products) */
      if (j == 0)
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      // Prefetch the start of the NEXT block of A before calling the microkernel
      // if (i + 4 < m) {
      //     __builtin_prefetch(&packedA[(i + 4) * k], 0, 0);
      // }
      AddDot4x4(k, &packedA[i * k], 4, &packedB[j * k], k, &C(i, j), ldc);
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

typedef union
{
  float64x2_t v;
  double d[2];
} v2df_t;

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )

     in the original matrix C

     And now we use vector registers and instructions */

  int p;
  v2df_t
      // Accumulators for even steps of K (p)
      c0_00_c_10, c0_01_c_11, c0_02_c_12, c0_03_c_13,
      c0_20_c_30, c0_21_c_31, c0_22_c_32, c0_23_c_33,
      // Accumulators for odd steps of K (p+1)
      c1_00_c_10, c1_01_c_11, c1_02_c_12, c1_03_c_13,
      c1_20_c_30, c1_21_c_31, c1_22_c_32, c1_23_c_33;

  v2df_t 
      a_0p_a_1p_vreg, a_2p_a_3p_vreg,
      b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

  c0_00_c_10.v = vdupq_n_f64(0.0);
  c0_01_c_11.v = vdupq_n_f64(0.0);
  c0_02_c_12.v = vdupq_n_f64(0.0);
  c0_03_c_13.v = vdupq_n_f64(0.0);
  c0_20_c_30.v = vdupq_n_f64(0.0);
  c0_21_c_31.v = vdupq_n_f64(0.0);
  c0_22_c_32.v = vdupq_n_f64(0.0);
  c0_23_c_33.v = vdupq_n_f64(0.0);
  c1_00_c_10.v = vdupq_n_f64(0.0);
  c1_01_c_11.v = vdupq_n_f64(0.0);
  c1_02_c_12.v = vdupq_n_f64(0.0);
  c1_03_c_13.v = vdupq_n_f64(0.0);
  c1_20_c_30.v = vdupq_n_f64(0.0);
  c1_21_c_31.v = vdupq_n_f64(0.0);
  c1_22_c_32.v = vdupq_n_f64(0.0);
  c1_23_c_33.v = vdupq_n_f64(0.0);

  // __builtin_prefetch(&C(0, 0), 1, 0);
  // __builtin_prefetch(&C(1, 0), 1, 0);
  // __builtin_prefetch(&C(2, 0), 1, 0);
  // __builtin_prefetch(&C(3, 0), 1, 0);
  // __builtin_prefetch(&C(0, 1), 1, 0);
  // __builtin_prefetch(&C(1, 1), 1, 0);
  // __builtin_prefetch(&C(2, 1), 1, 0);
  // __builtin_prefetch(&C(3, 1), 1, 0);
  // __builtin_prefetch(&C(0, 2), 1, 0);
  // __builtin_prefetch(&C(1, 2), 1, 0);
  // __builtin_prefetch(&C(2, 2), 1, 0);
  // __builtin_prefetch(&C(3, 2), 1, 0);
  // __builtin_prefetch(&C(0, 3), 1, 0);
  // __builtin_prefetch(&C(1, 3), 1, 0);
  // __builtin_prefetch(&C(2, 3), 1, 0);
  // __builtin_prefetch(&C(3, 3), 1, 0);

  for (p = 0; p < k; p+=2)
  {
    // __builtin_prefetch(a + 64, 0, 1); // Prefetch next 2 packed columns of A
    // __builtin_prefetch(b + 64, 0, 1); // Prefetch next 2 packed rows of B
    a_0p_a_1p_vreg.v = vld1q_f64(a);
    b_p0_vreg.v = vld1q_dup_f64(b);       /* load and duplicate */
    c0_00_c_10.v = vfmaq_f64(c0_00_c_10.v, a_0p_a_1p_vreg.v, b_p0_vreg.v);
    a_2p_a_3p_vreg.v = vld1q_f64(a + 2);
    c0_20_c_30.v = vfmaq_f64(c0_20_c_30.v, a_2p_a_3p_vreg.v, b_p0_vreg.v);
    b_p1_vreg.v = vld1q_dup_f64(b + 1); /* load and duplicate */
    c0_01_c_11.v = vfmaq_f64(c0_01_c_11.v, a_0p_a_1p_vreg.v, b_p1_vreg.v);
    c0_21_c_31.v = vfmaq_f64(c0_21_c_31.v, a_2p_a_3p_vreg.v, b_p1_vreg.v);
    b_p2_vreg.v = vld1q_dup_f64(b + 2); /* load and duplicate */
    c0_02_c_12.v = vfmaq_f64(c0_02_c_12.v, a_0p_a_1p_vreg.v, b_p2_vreg.v);
    c0_22_c_32.v = vfmaq_f64(c0_22_c_32.v, a_2p_a_3p_vreg.v, b_p2_vreg.v);
    b_p3_vreg.v = vld1q_dup_f64(b + 3); /* load and duplicate */
    c0_03_c_13.v = vfmaq_f64(c0_03_c_13.v, a_0p_a_1p_vreg.v, b_p3_vreg.v);
    c0_23_c_33.v = vfmaq_f64(c0_23_c_33.v, a_2p_a_3p_vreg.v, b_p3_vreg.v);
    

    /* --- Iteration 2 (p+1) --- */
    
    // Compute the next 8 FMAs
    // By the time the CPU scheduler gets here, the 4-cycle latency 
    // from Iteration 1 is finished, and these run without stalling!
    a_0p_a_1p_vreg.v = vld1q_f64(a + 4);
    b_p0_vreg.v = vld1q_dup_f64(b + 4);
    c1_00_c_10.v = vfmaq_f64(c1_00_c_10.v, a_0p_a_1p_vreg.v, b_p0_vreg.v);
    a_2p_a_3p_vreg.v = vld1q_f64(a + 6);
    c1_20_c_30.v = vfmaq_f64(c1_20_c_30.v, a_2p_a_3p_vreg.v, b_p0_vreg.v);
    b_p1_vreg.v = vld1q_dup_f64(b + 5);
    c1_01_c_11.v = vfmaq_f64(c1_01_c_11.v, a_0p_a_1p_vreg.v, b_p1_vreg.v);
    c1_21_c_31.v = vfmaq_f64(c1_21_c_31.v, a_2p_a_3p_vreg.v, b_p1_vreg.v);
    b_p2_vreg.v = vld1q_dup_f64(b + 6);
    c1_02_c_12.v = vfmaq_f64(c1_02_c_12.v, a_0p_a_1p_vreg.v, b_p2_vreg.v);
    c1_22_c_32.v = vfmaq_f64(c1_22_c_32.v, a_2p_a_3p_vreg.v, b_p2_vreg.v);
    b_p3_vreg.v = vld1q_dup_f64(b + 7);
    c1_03_c_13.v = vfmaq_f64(c1_03_c_13.v, a_0p_a_1p_vreg.v, b_p3_vreg.v);
    c1_23_c_33.v = vfmaq_f64(c1_23_c_33.v, a_2p_a_3p_vreg.v, b_p3_vreg.v);

    a += 8; // Advance A by 2 packed columns (2 * 4)
    b += 8; // Advance B by 2 packed rows (2 * 4)
  }

  C(0, 0) += c0_00_c_10.d[0] + c1_00_c_10.d[0];
  C(0, 1) += c0_01_c_11.d[0] + c1_01_c_11.d[0];
  C(0, 2) += c0_02_c_12.d[0] + c1_02_c_12.d[0];
  C(0, 3) += c0_03_c_13.d[0] + c1_03_c_13.d[0];

  C(1, 0) += c0_00_c_10.d[1] + c1_00_c_10.d[1];
  C(1, 1) += c0_01_c_11.d[1] + c1_01_c_11.d[1];
  C(1, 2) += c0_02_c_12.d[1] + c1_02_c_12.d[1];
  C(1, 3) += c0_03_c_13.d[1] + c1_03_c_13.d[1];

  C(2, 0) += c0_20_c_30.d[0] + c1_20_c_30.d[0];
  C(2, 1) += c0_21_c_31.d[0] + c1_21_c_31.d[0];
  C(2, 2) += c0_22_c_32.d[0] + c1_22_c_32.d[0];
  C(2, 3) += c0_23_c_33.d[0] + c1_23_c_33.d[0];

  C(3, 0) += c0_20_c_30.d[1] + c1_20_c_30.d[1];
  C(3, 1) += c0_21_c_31.d[1] + c1_21_c_31.d[1];
  C(3, 2) += c0_22_c_32.d[1] + c1_22_c_32.d[1];
  C(3, 3) += c0_23_c_33.d[1] + c1_23_c_33.d[1];
}
