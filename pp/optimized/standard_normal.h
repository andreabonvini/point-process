
//
//  standard_normal.h
//  regr_likel
//
//  Created by Andrea Bonvini on 31/03/21.
//

#ifndef standard_normal_h
#define standard_normal_h


#endif /* standard_normal_h */

#include <math.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpfr.h>



void mpfr_norm_cdf(mpfr_t rop, mpfr_t op){
    // norm_cdf(x) = 1/2 (1 + erf(x/sqrt(2))
    // https://en.wikipedia.org/wiki/Normal_distribution
    // Set rop to the value of the norm cdf on op.
    mpfr_set_d(rop, (double) sqrt(2.0),MPFR_RNDN); // sqrt(2)
    mpfr_div(rop, op, rop,MPFR_RNDN); // x / sqrt(2)
    mpfr_erf (rop, rop, MPFR_RNDN);// erf( x / sqrt(2))
    mpfr_add_d(rop, rop, (double) 1.0, MPFR_RNDN); // 1.0 + erf( x / sqrt(2))
    mpfr_mul_d(rop, rop, (double) 0.5,MPFR_RNDN); //  1/2 (1.0 + erf(x/sqrt(2))
}


void mpfr_norm_pdf(mpfr_t rop, mpfr_t op){
    // Set rop to the value of the norm pdf on op
    mpfr_pow_ui(rop, op, (unsigned long) 2, MPFR_RNDN);      // rop =                            op^2
    mpfr_neg(rop, rop, MPFR_RNDN);                           // rop =                          - op^2
    mpfr_mul_d(rop, rop, (double) 0.5, MPFR_RNDN);           // rop =                    - 0.5 * op^2
    mpfr_exp(rop, rop,MPFR_RNDN);                            // rop =                exp(- 0.5 * op^2)
    mpfr_div_d(rop, rop, (double) sqrt(2*M_PI), MPFR_RNDN);  // rop = 1/sqrt(2*pi) * exp(- 0.5 * op^2)
}

void mpfr_norm_logpdf(mpfr_t rop, mpfr_t op){
    // Set rop to the value of the norm logpdf on op
    mpfr_pow_ui(rop, op, (unsigned long) 2, MPFR_RNDN);          // rop =         op^2
    mpfr_div_d(rop, rop, (double) 2.0, MPFR_RNDN);               // rop =   1/2 * op^2
    mpfr_neg(rop, rop, MPFR_RNDN);                               // rop = - 1/2 * op^2
    mpfr_sub_d(rop, rop, (double) 0.5 * log(2*M_PI), MPFR_RNDN); // rop = - 1/2 * op^2 - 1/2 log(2*pi)
}


double norm_logcdf(double x){
    // Probabably that's kinda computationally expensive
    mpfr_t mpfr_x;
    mpfr_init2(mpfr_x,128);
    mpfr_set_d(mpfr_x,x,MPFR_RNDN);
    mpfr_neg(mpfr_x,mpfr_x,MPFR_RNDN);
    mpfr_mul_d(mpfr_x,mpfr_x, (double) M_SQRT1_2, MPFR_RNDN);
    mpfr_erfc(mpfr_x,mpfr_x,MPFR_RNDN);
    mpfr_mul_d(mpfr_x,mpfr_x,(double) 0.5, MPFR_RNDN);
    mpfr_log(mpfr_x,mpfr_x,MPFR_RNDN);
    double res = mpfr_get_d(mpfr_x,MPFR_RNDN);
    mpfr_clear(mpfr_x);
    return res;
}

