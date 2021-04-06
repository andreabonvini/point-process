//
//  inverse_gaussian.h
//  regr_likel
//
//  Created by Andrea Bonvini on 31/03/21.
//

#ifndef inverse_gaussian_h
#define inverse_gaussian_h


#endif /* inverse_gaussian_h */

#include <math.h>
#include <nlopt.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpfr.h>
#include "miscellania.h"
#include "standard_normal.h"

struct ig_data
{
    double * xn; // [N_SAMPLES][AR_ORDER +1];
    double * eta;// [N_SAMPLES];
    double * wn; // [N_SAMPLES];
    double * xt; //[AR_ORDER + 1];
    double wt;
    int AR_ORDER;
    int N_SAMPLES;
    int right_censoring;
};

double compute_igcdf(double wt, double rc_mu, double k){

    if (wt == 0.0){
        return 0.0;
        }

    mpfr_t igcdf;
    mpfr_t kwt;
    mpfr_t sqrt_kwt;
    mpfr_t minus_sqrt_kwt;
    mpfr_t wtrcmu;
    mpfr_t two_wt_sqrt_kwt;
    mpfr_t wtrcmu_minus_one;
    mpfr_t wtrcmu_plus_one;
    mpfr_t k_rcmu;
    mpfr_t sqrt_kwt_wtrcmu_minus_one;
    mpfr_t minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t exp_argument;
    mpfr_t exp_result;
    mpfr_t norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t two_k_over_rcmu; // 2 * k / rc_mu
    mpfr_t two_over_rcmu; // 2 / rc_mu
    mpfr_t norm_cdf_sqrt_kwt_wtrcmu_minus_one;
    mpfr_t minus_sqrt_kwt_wt_over_rcmu_squared; // (-sqrt(k / wt) * wt / rc_mu ** 2)
    mpfr_t minus_two_k_over_rcmu_squared; // (-2 * k / rc_mu**2)
    mpfr_t mpfr_k;
    mpfr_t mpfr_rc_mu;
    mpfr_t mpfr_wt;

    unsigned long precision = 256;

    mpfr_init2(igcdf, precision);
    mpfr_init2(kwt,precision);
    mpfr_init2(sqrt_kwt, precision);
    mpfr_init2(minus_sqrt_kwt, precision);
    mpfr_init2(wtrcmu, precision);
    mpfr_init2(two_wt_sqrt_kwt, precision);
    mpfr_init2(wtrcmu_minus_one, precision);
    mpfr_init2(wtrcmu_plus_one, precision);
    mpfr_init2(k_rcmu, precision);
    mpfr_init2(sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(exp_argument, precision);
    mpfr_init2(exp_result, precision);
    mpfr_init2(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(two_k_over_rcmu, precision);
    mpfr_init2(two_over_rcmu, precision);
    mpfr_init2(norm_cdf_sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(minus_sqrt_kwt_wt_over_rcmu_squared, precision);
    mpfr_init2(minus_two_k_over_rcmu_squared, precision);
    mpfr_init2(mpfr_k, precision);
    mpfr_init2(mpfr_rc_mu, precision);
    mpfr_init2(mpfr_wt, precision);

    mpfr_set_d(mpfr_k, k,MPFR_RNDD);
    mpfr_set_d(mpfr_rc_mu, (double) rc_mu, MPFR_RNDD);
    mpfr_set_d(mpfr_wt, wt, MPFR_RNDD);

    // two_over_rcmu = 2 / rc_mu
    mpfr_set_d(two_over_rcmu, (double) 2.0, MPFR_RNDN);
    mpfr_div(two_over_rcmu, two_over_rcmu, mpfr_rc_mu, MPFR_RNDN);

    // k_rcmu = k / rc_mu
    mpfr_div(k_rcmu,mpfr_k,mpfr_rc_mu,MPFR_RNDN);

    // two_k_over_rcmu = 2 * k / rc_mu
    mpfr_mul_d(two_k_over_rcmu, k_rcmu, (double) 2.0, MPFR_RNDN);

    // minus_two_k_over_rcmu_squared = (-2 * k / rc_mu**2)
    mpfr_neg(minus_two_k_over_rcmu_squared, two_k_over_rcmu, MPFR_RNDN);
    mpfr_div(minus_two_k_over_rcmu_squared, minus_two_k_over_rcmu_squared, mpfr_rc_mu, MPFR_RNDN);

    // kwt = k / wt
    mpfr_div(kwt,mpfr_k,mpfr_wt,MPFR_RNDN);

    // sqrt_kwt = sqrt(k/wt)
    mpfr_sqrt(sqrt_kwt,kwt,MPFR_RNDN);

    // two_wt_sqrt_kwt = 2 *wt * sqrt(k/wt)
    mpfr_mul_d(two_wt_sqrt_kwt, mpfr_wt, (double) 2.0, MPFR_RNDN);
    mpfr_mul(two_wt_sqrt_kwt,two_wt_sqrt_kwt, sqrt_kwt, MPFR_RNDN);

    // minus_sqrt_kwt = -sqrt(k/wt)
    mpfr_neg(minus_sqrt_kwt, sqrt_kwt, MPFR_RNDN);

    // wtrcmu = wt / rc_mu
    mpfr_div(wtrcmu, mpfr_wt, mpfr_rc_mu, MPFR_RNDN);

    // wtrcmu_minus_one = wt / rc_mu - 1
    mpfr_sub_d(wtrcmu_minus_one, wtrcmu, 1.0, MPFR_RNDN);

    // wtrcmu_plus_one = wt / rc_mu + 1
    mpfr_add_d(wtrcmu_plus_one, wtrcmu, 1.0, MPFR_RNDN);

    // sqrt_kwt_wtrcmu_minus_one = sqrt(k/wt) * (wt/rc_mu - )
    mpfr_mul(sqrt_kwt_wtrcmu_minus_one, sqrt_kwt, wtrcmu_minus_one,MPFR_RNDN);

    // minus_sqrt_kwt_wtrcmu_plus_one = -sqrt(k/wt) * (wt/rc_mu + 1)
    mpfr_mul(minus_sqrt_kwt_wtrcmu_plus_one, minus_sqrt_kwt, wtrcmu_plus_one,MPFR_RNDN);

    // minus_sqrt_kwt_wt_over_rcmu_squared = -sqrt(k/wwt) * wt / mu**2
    mpfr_mul(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt, mpfr_wt, MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);

    // --------------------------------------- Compute Inverse Gaussian CDF  ----------------------------------------

    // igCDF(t) = norm.cdf(sqrt(k / t) * (t / mu - 1)) + np.exp( (2 * k / mu) + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1)) )

    double double_minus_sqrt_kwt_wtrcmu_plus_one = mpfr_get_d(minus_sqrt_kwt_wtrcmu_plus_one,MPFR_RNDN);
    mpfr_set_d(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, norm_logcdf(double_minus_sqrt_kwt_wtrcmu_plus_one), MPFR_RNDN);

    // exp_argument = (2 * k / rc_mu) + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_mul_d(exp_argument,k_rcmu, (double) 2.0,MPFR_RNDN);
    mpfr_add(exp_argument, exp_argument, norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, MPFR_RNDN);

    // exp_result = e^(exp_argument)
    mpfr_exp(exp_result, exp_argument, MPFR_RNDN);

    // norm_cdf_sqrt_kwt_wtrcmu_minus_one = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1))
    mpfr_norm_cdf(norm_cdf_sqrt_kwt_wtrcmu_minus_one, sqrt_kwt_wtrcmu_minus_one);

    // igcdf = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1)) + exp_result
    mpfr_add(igcdf, norm_cdf_sqrt_kwt_wtrcmu_minus_one, exp_result,MPFR_RNDN);
    // -------------------------------------- Inverse Gaussian CDF Computed ----------------------------------------

    double res = mpfr_get_d(igcdf, MPFR_RNDN);

    // Deallocate variables
    mpfr_clear(igcdf);
    mpfr_clear(kwt);
    mpfr_clear(sqrt_kwt);
    mpfr_clear(minus_sqrt_kwt);
    mpfr_clear(wtrcmu);
    mpfr_clear(two_wt_sqrt_kwt);
    mpfr_clear(wtrcmu_minus_one);
    mpfr_clear(wtrcmu_plus_one);
    mpfr_clear(k_rcmu);
    mpfr_clear(sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(exp_argument);
    mpfr_clear(exp_result);
    mpfr_clear(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(two_k_over_rcmu);
    mpfr_clear(two_over_rcmu);
    mpfr_clear(norm_cdf_sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(minus_sqrt_kwt_wt_over_rcmu_squared);
    mpfr_clear(minus_two_k_over_rcmu_squared);
    mpfr_clear(mpfr_k);
    mpfr_clear(mpfr_rc_mu);
    mpfr_clear(mpfr_wt);

    return res;
}
// ------------

double compute_lambda(double wt, double rc_mu, double k){

    if (wt == 0.0){
        return 0.0;
        }

    mpfr_t igpdf;
    mpfr_t igcdf;
    mpfr_t lambda;
    mpfr_t kwt;
    mpfr_t sqrt_kwt;
    mpfr_t minus_sqrt_kwt;
    mpfr_t wtrcmu;
    mpfr_t two_wt_sqrt_kwt;
    mpfr_t wtrcmu_minus_one;
    mpfr_t wtrcmu_plus_one;
    mpfr_t k_rcmu;
    mpfr_t sqrt_kwt_wtrcmu_minus_one;
    mpfr_t minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t exp_argument;
    mpfr_t exp_result;
    mpfr_t norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t two_k_over_rcmu; // 2 * k / rc_mu
    mpfr_t two_over_rcmu; // 2 / rc_mu
    mpfr_t norm_cdf_sqrt_kwt_wtrcmu_minus_one;
    mpfr_t minus_sqrt_kwt_wt_over_rcmu_squared; // (-sqrt(k / wt) * wt / rc_mu ** 2)
    mpfr_t minus_two_k_over_rcmu_squared; // (-2 * k / rc_mu**2)
    mpfr_t mpfr_k;
    mpfr_t mpfr_rc_mu;
    mpfr_t mpfr_wt;
    mpfr_t sqrt_term;

    unsigned long precision = 256;

    mpfr_init2(igpdf, precision);
    mpfr_init2(igcdf, precision);
    mpfr_init2(lambda, precision);
    mpfr_init2(kwt,precision);
    mpfr_init2(sqrt_kwt, precision);
    mpfr_init2(minus_sqrt_kwt, precision);
    mpfr_init2(wtrcmu, precision);
    mpfr_init2(two_wt_sqrt_kwt, precision);
    mpfr_init2(wtrcmu_minus_one, precision);
    mpfr_init2(wtrcmu_plus_one, precision);
    mpfr_init2(k_rcmu, precision);
    mpfr_init2(sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(exp_argument, precision);
    mpfr_init2(exp_result, precision);
    mpfr_init2(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(two_k_over_rcmu, precision);
    mpfr_init2(two_over_rcmu, precision);
    mpfr_init2(norm_cdf_sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(minus_sqrt_kwt_wt_over_rcmu_squared, precision);
    mpfr_init2(minus_two_k_over_rcmu_squared, precision);
    mpfr_init2(mpfr_k, precision);
    mpfr_init2(mpfr_rc_mu, precision);
    mpfr_init2(mpfr_wt, precision);
    mpfr_init2(sqrt_term, precision);

    mpfr_set_d(mpfr_k, k,MPFR_RNDD);
    mpfr_set_d(mpfr_rc_mu, (double) rc_mu, MPFR_RNDD);
    mpfr_set_d(mpfr_wt, wt, MPFR_RNDD);

    // two_over_rcmu = 2 / rc_mu
    mpfr_set_d(two_over_rcmu, (double) 2.0, MPFR_RNDN);
    mpfr_div(two_over_rcmu, two_over_rcmu, mpfr_rc_mu, MPFR_RNDN);

    // k_rcmu = k / rc_mu
    mpfr_div(k_rcmu,mpfr_k,mpfr_rc_mu,MPFR_RNDN);

    // two_k_over_rcmu = 2 * k / rc_mu
    mpfr_mul_d(two_k_over_rcmu, k_rcmu, (double) 2.0, MPFR_RNDN);

    // minus_two_k_over_rcmu_squared = (-2 * k / rc_mu**2)
    mpfr_neg(minus_two_k_over_rcmu_squared, two_k_over_rcmu, MPFR_RNDN);
    mpfr_div(minus_two_k_over_rcmu_squared, minus_two_k_over_rcmu_squared, mpfr_rc_mu, MPFR_RNDN);

    // kwt = k / wt
    mpfr_div(kwt,mpfr_k,mpfr_wt,MPFR_RNDN);

    // sqrt_kwt = sqrt(k/wt)
    mpfr_sqrt(sqrt_kwt,kwt,MPFR_RNDN);

    // two_wt_sqrt_kwt = 2 *wt * sqrt(k/wt)
    mpfr_mul_d(two_wt_sqrt_kwt, mpfr_wt, (double) 2.0, MPFR_RNDN);
    mpfr_mul(two_wt_sqrt_kwt,two_wt_sqrt_kwt, sqrt_kwt, MPFR_RNDN);

    // minus_sqrt_kwt = -sqrt(k/wt)
    mpfr_neg(minus_sqrt_kwt, sqrt_kwt, MPFR_RNDN);

    // wtrcmu = wt / rc_mu
    mpfr_div(wtrcmu, mpfr_wt, mpfr_rc_mu, MPFR_RNDN);

    // wtrcmu_minus_one = wt / rc_mu - 1
    mpfr_sub_d(wtrcmu_minus_one, wtrcmu, 1.0, MPFR_RNDN);

    // wtrcmu_plus_one = wt / rc_mu + 1
    mpfr_add_d(wtrcmu_plus_one, wtrcmu, 1.0, MPFR_RNDN);

    // sqrt_kwt_wtrcmu_minus_one = sqrt(k/wt) * (wt/rc_mu - )
    mpfr_mul(sqrt_kwt_wtrcmu_minus_one, sqrt_kwt, wtrcmu_minus_one,MPFR_RNDN);

    // minus_sqrt_kwt_wtrcmu_plus_one = -sqrt(k/wt) * (wt/rc_mu + 1)
    mpfr_mul(minus_sqrt_kwt_wtrcmu_plus_one, minus_sqrt_kwt, wtrcmu_plus_one,MPFR_RNDN);

    // minus_sqrt_kwt_wt_over_rcmu_squared = -sqrt(k/wwt) * wt / mu**2
    mpfr_mul(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt, mpfr_wt, MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);

    // --------------------------------------- Compute Inverse Gaussian CDF  ----------------------------------------

    // igCDF(t) = norm.cdf(sqrt(k / t) * (t / mu - 1)) + np.exp( (2 * k / mu) + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1)) )

    double double_minus_sqrt_kwt_wtrcmu_plus_one = mpfr_get_d(minus_sqrt_kwt_wtrcmu_plus_one,MPFR_RNDN);
    mpfr_set_d(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, norm_logcdf(double_minus_sqrt_kwt_wtrcmu_plus_one), MPFR_RNDN);

    // exp_argument = (2 * k / rc_mu) + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_mul_d(exp_argument,k_rcmu, (double) 2.0,MPFR_RNDN);
    mpfr_add(exp_argument, exp_argument, norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, MPFR_RNDN);

    // exp_result = e^(exp_argument)
    mpfr_exp(exp_result, exp_argument, MPFR_RNDN);

    // norm_cdf_sqrt_kwt_wtrcmu_minus_one = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1))
    mpfr_norm_cdf(norm_cdf_sqrt_kwt_wtrcmu_minus_one, sqrt_kwt_wtrcmu_minus_one);

    // igcdf = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1)) + exp_result
    mpfr_add(igcdf, norm_cdf_sqrt_kwt_wtrcmu_minus_one, exp_result,MPFR_RNDN);
    // -------------------------------------- Inverse Gaussian CDF Computed ----------------------------------------

    // -------------------------------------- Compute Inverse Gaussian PDF -----------------------------------------

    // igPDF(t) = sqrt(k / (2 * pi * t ** 3)) * exp( (-k * (t - mu) ** 2) / (2 * mu ** 2 * t) )

    // sqrt_term = sqrt(k / (2 * pi * t ** 3))
    mpfr_pow_ui(sqrt_term, mpfr_wt, (unsigned long) 3, MPFR_RNDN);
    mpfr_mul_d(sqrt_term, sqrt_term, (double) 2 * M_PI, MPFR_RNDN);
    mpfr_div(sqrt_term, mpfr_k, sqrt_term,  MPFR_RNDN);
    mpfr_sqrt(sqrt_term, sqrt_term, MPFR_RNDN);

    // igpdf = sqrt_term * exp( (-k * (t - mu) ** 2) / (2 * mu ** 2 * t) )
    mpfr_sub(igpdf, mpfr_wt, mpfr_rc_mu, MPFR_RNDN); // (t-mu)
    mpfr_pow_ui(igpdf, igpdf, (unsigned long) 2, MPFR_RNDN); // (t - mu)**2
    mpfr_mul(igpdf, igpdf, mpfr_k, MPFR_RNDN); // k * (t - mu)**2
    mpfr_neg(igpdf, igpdf, MPFR_RNDN); // - k * (t - mu)**2
    mpfr_div_d(igpdf, igpdf, (double) 2.0, MPFR_RNDN); // (-k * (t - mu) ** 2) / (2)
    mpfr_div(igpdf, igpdf, mpfr_rc_mu, MPFR_RNDN); // (-k * (t - mu) ** 2) / (2 * mu)
    mpfr_div(igpdf, igpdf, mpfr_rc_mu, MPFR_RNDN); // (-k * (t - mu) ** 2) / (2 * mu ** 2)
    mpfr_div(igpdf, igpdf, mpfr_wt, MPFR_RNDN); // (-k * (t - mu) ** 2) / (2 * mu ** 2 * t)
    mpfr_exp(igpdf, igpdf, MPFR_RNDN); // exp((-k * (t - mu) ** 2) / (2 * mu ** 2 * t))
    mpfr_mul(igpdf, igpdf, sqrt_term, MPFR_RNDN); // sqrt_term * exp( (-k * (t - mu) ** 2) / (2 * mu ** 2 * t) )
    // -------------------------------------- Inverse Gaussian PDF Computed ----------------------------------------

    // -------------------------------------- Compute Hazard Rate (Lambda) -----------------------------------------

    // lambda = igpdf / (1 - igcdf)
    mpfr_neg(lambda, igcdf, MPFR_RNDN); // - igcdf
    mpfr_add_d(lambda, lambda , (double) 1.0, MPFR_RNDN); // 1 - igcdf
    mpfr_div(lambda, igpdf, lambda, MPFR_RNDN); // igpdf / (1- igcdf)
    // -------------------------------------- Hazard Rate (Lambda) Computed -----------------------------------------

    // Cast to double
    double res = mpfr_get_d(lambda, MPFR_RNDN);

    // Deallocate variables
    mpfr_clear(igpdf);
    mpfr_clear(igcdf);
    mpfr_clear(lambda);
    mpfr_clear(kwt);
    mpfr_clear(sqrt_kwt);
    mpfr_clear(minus_sqrt_kwt);
    mpfr_clear(wtrcmu);
    mpfr_clear(two_wt_sqrt_kwt);
    mpfr_clear(wtrcmu_minus_one);
    mpfr_clear(wtrcmu_plus_one);
    mpfr_clear(k_rcmu);
    mpfr_clear(sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(exp_argument);
    mpfr_clear(exp_result);
    mpfr_clear(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(two_k_over_rcmu);
    mpfr_clear(two_over_rcmu);
    mpfr_clear(norm_cdf_sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(minus_sqrt_kwt_wt_over_rcmu_squared);
    mpfr_clear(minus_two_k_over_rcmu_squared);
    mpfr_clear(mpfr_k);
    mpfr_clear(mpfr_rc_mu);
    mpfr_clear(mpfr_wt);
    mpfr_clear(sqrt_term);

    return res;
}

void ig_gradient(int N_SAMPLES, int AR_ORDER, double xn[N_SAMPLES][AR_ORDER+1], double * mus, double * wn, double k, double * eta, double * grad){

    double tmp2_n[N_SAMPLES];
    double tmp3_n[N_SAMPLES];
    double tmp4_n[N_SAMPLES];
    double tmp5_n[N_SAMPLES];
    double tmp6_n[N_SAMPLES];
    double tmp7_n[N_SAMPLES];
    double tmp8_n[N_SAMPLES];
    double k_grad;
    double theta_grad[AR_ORDER + 1];

    // Firstly we have to derive k_grad as
    // k_grad = 1/2 * dotprod(v1,v2)
    // v1 = eta
    // v2 = -1 / k + (wn - mus) ** 2 / (mus ** 2 * wn)
    vector_plus_vector(wn, mus, N_SAMPLES, tmp3_n, 0); // tmp3_n = wn - mus
    exp_vector(tmp3_n, 2.0, N_SAMPLES, tmp5_n); // tmp5_n = tmp3_n**2
    exp_vector(mus, 2.0, N_SAMPLES, tmp4_n); // tmp4_n = mus**2
    elementwise_product(tmp4_n, wn, N_SAMPLES, tmp6_n); // tmp6_n = tmp4_n * wn
    elementwise_ratio(tmp5_n, tmp6_n, N_SAMPLES, tmp7_n); // tmp7_n = tmp5_n/tmp6_n
    vector_plus_scalar(-1/k , tmp7_n, N_SAMPLES, tmp8_n);
    k_grad = 0.5 * vectorvector_product(eta,tmp8_n, N_SAMPLES);

    // In order to compute theta_grad we have smth like...
    // tmp =  -1 * k * eta * (wn - mus) / mus ** 3
    // theta_grad = dotprod(xn.T, tmp)
    exp_vector(mus, 3.0, N_SAMPLES, tmp2_n); // tmp2_n = mus**3
    vector_plus_vector(wn, mus, N_SAMPLES, tmp3_n, 0);  // tmp3_n = wn - mus
    vector_times_scalar(-k, eta, N_SAMPLES, tmp4_n); // tmp4_n = -1 * k * eta
    elementwise_product(tmp4_n, tmp3_n, N_SAMPLES, tmp5_n);  // tmp5_n = tmp4_n * tmp3_n
    elementwise_ratio(tmp5_n, tmp2_n, N_SAMPLES, tmp6_n); // tmp6_n = tmp5_n/tmp2_n = (-1 * k * eta * (wn-mus)) / mus**3
    matrixvector_product(N_SAMPLES, AR_ORDER + 1, xn, tmp6_n, theta_grad, 1); // thetagrad = dotprod(xn.T,tmp6_n)


    // Let's populate the gradient vector
    grad[0] = k_grad;
    for (int i = 1; i < AR_ORDER + 2; i++){
        grad[i] = theta_grad[i-1];
    }
}

void mpfr_ig_gradient_rc(int AR_ORDER, double k, double wt, double rc_mu, double rc_eta, double * xt, mpfr_t igcdf, double * rc_gradient){

    // Note that we pass the igcdf pointer as argument since we want to save it for later
    int precision = 256;

    double rc_dk;
    double rc_dtheta[AR_ORDER + 1];

    mpfr_t mul1;
    mpfr_t add1;
    mpfr_t add2;
    mpfr_t add3;
    mpfr_t kwt;
    mpfr_t sqrt_kwt;
    mpfr_t minus_sqrt_kwt;
    mpfr_t wtrcmu;
    mpfr_t two_wt_sqrt_kwt;
    mpfr_t wtrcmu_minus_one;
    mpfr_t wtrcmu_plus_one;
    mpfr_t k_rcmu;
    mpfr_t sqrt_kwt_wtrcmu_minus_one;
    mpfr_t minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t exp_argument;
    mpfr_t exp_result;
    mpfr_t tmp_exp_logcdf; // np.exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
    mpfr_t tmp_exp_logpdf; // np.exp(2 * k / rc_mu + norm.logpdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
    mpfr_t norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t norm_pdf_sqrt_kwt_wtrcmu_minus_one;
    mpfr_t norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one;
    mpfr_t two_k_over_rcmu; // 2 * k / rc_mu
    mpfr_t two_over_rcmu; // 2 / rc_mu
    mpfr_t norm_cdf_sqrt_kwt_wtrcmu_minus_one;
    mpfr_t mpfr_rc_dk;
    mpfr_t mpfr_rc_dmu;
    mpfr_t minus_sqrt_kwt_wt_over_rcmu_squared; // (-sqrt(k / wt) * wt / rc_mu ** 2)
    mpfr_t minus_two_k_over_rcmu_squared; // (-2 * k / rc_mu**2)
    mpfr_t mpfr_k;
    mpfr_t mpfr_rc_mu;
    mpfr_t mpfr_rc_eta;
    mpfr_t mpfr_wt;

    mpfr_init2(mul1, precision);
    mpfr_init2(add1, precision);
    mpfr_init2(add2, precision);
    mpfr_init2(add3, precision);
    mpfr_init2(kwt,precision);
    mpfr_init2(sqrt_kwt, precision);
    mpfr_init2(minus_sqrt_kwt, precision);
    mpfr_init2(wtrcmu, precision);
    mpfr_init2(two_wt_sqrt_kwt, precision);
    mpfr_init2(wtrcmu_minus_one, precision);
    mpfr_init2(wtrcmu_plus_one, precision);
    mpfr_init2(k_rcmu, precision);
    mpfr_init2(sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(exp_argument, precision);
    mpfr_init2(exp_result, precision);
    mpfr_init2(tmp_exp_logcdf, precision);
    mpfr_init2(tmp_exp_logpdf, precision);
    mpfr_init2(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(norm_pdf_sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one, precision);
    mpfr_init2(two_k_over_rcmu, precision);
    mpfr_init2(two_over_rcmu, precision);
    mpfr_init2(norm_cdf_sqrt_kwt_wtrcmu_minus_one, precision);
    mpfr_init2(mpfr_rc_dk, precision);
    mpfr_init2(mpfr_rc_dmu, precision);
    mpfr_init2(minus_sqrt_kwt_wt_over_rcmu_squared,precision);
    mpfr_init2(minus_two_k_over_rcmu_squared,precision);
    mpfr_init2(mpfr_k,precision);
    mpfr_init2(mpfr_rc_mu,precision);
    mpfr_init2(mpfr_rc_eta,precision);
    mpfr_init2(mpfr_wt,precision);

    mpfr_set_d(mpfr_k, k,MPFR_RNDD); // mpfr_k = k
    mpfr_set_d(mpfr_rc_mu, rc_mu, MPFR_RNDD); // mpfr_rc_mu = rc_mu
    mpfr_set_d(mpfr_rc_eta, rc_eta, MPFR_RNDD); // mpfr_rc_eta = rc_eta
    mpfr_set_d(mpfr_wt, wt, MPFR_RNDD); // mpfr_wt = wt

    // two_over_rcmu = 2 / rc_mu
    mpfr_set_d(two_over_rcmu, (double) 2.0, MPFR_RNDN);
    mpfr_div(two_over_rcmu, two_over_rcmu, mpfr_rc_mu, MPFR_RNDN);
    // k_rcmu = k / rc_mu
    mpfr_div(k_rcmu,mpfr_k,mpfr_rc_mu,MPFR_RNDN);
    // two_k_over_rcmu = 2 * k / rc_mu
    mpfr_mul_d(two_k_over_rcmu, k_rcmu, (double) 2.0, MPFR_RNDN);
    // minus_two_k_over_rcmu_squared = (-2 * k / rc_mu**2)
    mpfr_neg(minus_two_k_over_rcmu_squared, two_k_over_rcmu, MPFR_RNDN);
    mpfr_div(minus_two_k_over_rcmu_squared, minus_two_k_over_rcmu_squared, mpfr_rc_mu, MPFR_RNDN);
    // kwt = k / wt
    mpfr_div(kwt,mpfr_k,mpfr_wt,MPFR_RNDN);
    // sqrt_kwt = sqrt(k/wt)
    mpfr_sqrt(sqrt_kwt,kwt,MPFR_RNDN);
    // two_wt_sqrt_kwt = 2 *wt * sqrt(k/wt)
    mpfr_mul_d(two_wt_sqrt_kwt, mpfr_wt, (double) 2.0, MPFR_RNDN);
    mpfr_mul(two_wt_sqrt_kwt, two_wt_sqrt_kwt, sqrt_kwt, MPFR_RNDN);
    // minus_sqrt_kwt = -sqrt(k/wt)
    mpfr_neg(minus_sqrt_kwt, sqrt_kwt, MPFR_RNDN);
    // wtrcmu = wt / rc_mu
    mpfr_div(wtrcmu, mpfr_wt, mpfr_rc_mu, MPFR_RNDN);
    // wtrcmu_minus_one = wt / rc_mu - 1
    mpfr_sub_d(wtrcmu_minus_one, wtrcmu, 1.0, MPFR_RNDN);
    // wtrcmu_plus_one = wt / rc_mu + 1
    mpfr_add_d(wtrcmu_plus_one, wtrcmu, 1.0, MPFR_RNDN);
    // sqrt_kwt_wtrcmu_minus_one = sqrt(k/wt) * (wt/rc_mu - 1)
    mpfr_mul(sqrt_kwt_wtrcmu_minus_one, sqrt_kwt, wtrcmu_minus_one,MPFR_RNDN);
    // minus_sqrt_kwt_wtrcmu_plus_one = -sqrt(k/wt) * (wt/rc_mu + 1)
    mpfr_mul(minus_sqrt_kwt_wtrcmu_plus_one, minus_sqrt_kwt, wtrcmu_plus_one,MPFR_RNDN);
    // minus_sqrt_kwt_wt_over_rcmu_squared = -sqrt(k/wwt) * wt / mu**2
    mpfr_mul(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt, mpfr_wt, MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);
    mpfr_div(minus_sqrt_kwt_wt_over_rcmu_squared, minus_sqrt_kwt_wt_over_rcmu_squared, mpfr_rc_mu,MPFR_RNDN);

    // --------------------------------------- Compute Inverse Gaussian CDF  ----------------------------------------

    // igCDF(t) = norm.cdf(sqrt(k / t) * (t / mu - 1)) + np.exp( (2 * k / mu) + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1)) )

    double double_minus_sqrt_kwt_wtrcmu_plus_one = mpfr_get_d(minus_sqrt_kwt_wtrcmu_plus_one,MPFR_RNDN);
    mpfr_set_d(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, norm_logcdf(double_minus_sqrt_kwt_wtrcmu_plus_one), MPFR_RNDN);

    // exp_argument = (2 * k / rc_mu) + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_mul_d(exp_argument,k_rcmu, (double) 2.0,MPFR_RNDN);
    mpfr_add(exp_argument, exp_argument, norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, MPFR_RNDN);

    // exp_result = e^(exp_argument)
    mpfr_exp(exp_result, exp_argument, MPFR_RNDN);

    // norm_cdf_sqrt_kwt_wtrcmu_minus_one = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1))
    mpfr_norm_cdf(norm_cdf_sqrt_kwt_wtrcmu_minus_one, sqrt_kwt_wtrcmu_minus_one);

    // igcdf = norm.cdf(sqrt(k/wt) * (wt/rc_mu - 1)) + exp_result
    mpfr_add(igcdf, norm_cdf_sqrt_kwt_wtrcmu_minus_one, exp_result, MPFR_RNDN);
    // -------------------------------------- Inverse Gaussian CDF Computed ----------------------------------------

    // Compute the right censoring derivative for k...

    // mul1 = rc_eta / (1 - igcdf)
    mpfr_neg(mul1, igcdf, MPFR_RNDN); // - igcdf
    // mpfr_printf("-igcdf: %.5Rf\n", mul1);
    mpfr_add_d(mul1, mul1, (double) 1.0, MPFR_RNDN); // 1 - igcdf
    mpfr_div(mul1, mpfr_rc_eta, mul1, MPFR_RNDN);// rc_eta / (1 - igcdf)

    // add1 = (wt / rc_mu - 1) / (2 * wt * sqrt(k / wt)) * norm.pdf(sqrt(k / wt) * (wt / rc_mu - 1))
    mpfr_norm_pdf(norm_pdf_sqrt_kwt_wtrcmu_minus_one, sqrt_kwt_wtrcmu_minus_one); // norm_pdf_sqrt_kwt_wtrcmu_minus_one = nordm.pdf(sqrt(k / wt) * (wt / rc_mu - 1))
    mpfr_div(add1, wtrcmu_minus_one, two_wt_sqrt_kwt, MPFR_RNDN);        // add1 = (wt / rc_mu - 1) / (2 * wt * sqrt(k / wt))
    mpfr_mul(add1, add1, norm_pdf_sqrt_kwt_wtrcmu_minus_one, MPFR_RNDN); // add1 = (wt / rc_mu - 1) / (2 * wt * sqrt(k / wt)) * norm.pdf(sqrt(k / wt) * (wt / rc_mu - 1))

    // tmp_exp_logcdf = exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
    // add2 = tmp_exp_logcdf * 2 / rc_mu
    mpfr_add(add2, norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one, two_k_over_rcmu, MPFR_RNDN); // add2           =     2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1))
    mpfr_exp(tmp_exp_logcdf, add2, MPFR_RNDN);                                              // tmp_exp_logcdf = exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
    mpfr_mul(add2, two_over_rcmu, tmp_exp_logcdf, MPFR_RNDN);                               // add2           = exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1))) * 2 / rc_mu


    // tmp_exp_logpdf = exp(2 * k / rc_mu + norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1)))
    // add3 = tmp_exp_logpdf * (-1 / (2 * wt * sqrt(k / wt)) * (wt / rc_mu + 1))
    mpfr_norm_logpdf(norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one, minus_sqrt_kwt_wtrcmu_plus_one);  // norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one = norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1)))
    mpfr_add(add3, norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one, two_k_over_rcmu, MPFR_RNDN);        // add3           =       norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))) + 2k/rc_mu
    mpfr_exp(tmp_exp_logpdf, add3, MPFR_RNDN);                                                     // tmp_exp_logpdf =   exp(norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))) + 2k/rc_mu)
    mpfr_mul(add3, tmp_exp_logpdf, wtrcmu_plus_one, MPFR_RNDN);                                    // add3           =   exp(norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))) + 2k/rc_mu) * (wt/rcmu + 1)
    mpfr_div(add3, add3, two_wt_sqrt_kwt, MPFR_RNDN);                                              // add3           =   exp(norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))) + 2k/rc_mu) * (wt/rcmu + 1) / (2 * wt * sqrt(k / wt))
    mpfr_neg(add3, add3, MPFR_RNDN);                                                               // add3           = - exp(norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))) + 2k/rc_mu) * (wt/rcmu + 1) / (2 * wt * sqrt(k / wt))

    // rc_dk = mul1 * (add1 + add2 + add3);
    mpfr_set_d(mpfr_rc_dk, (double) 0.0, MPFR_RNDN);   // mpfr_rc_dk = 0.0
    mpfr_add(mpfr_rc_dk, mpfr_rc_dk, add1, MPFR_RNDN); // mpfr_rc_dk = add1
    mpfr_add(mpfr_rc_dk, mpfr_rc_dk, add2, MPFR_RNDN); // mpfr_rc_dk = add1 + add2
    mpfr_add(mpfr_rc_dk, mpfr_rc_dk, add3, MPFR_RNDN); // mpfr_rc_dk = add1 + add2 + add3
    mpfr_mul(mpfr_rc_dk, mpfr_rc_dk, mul1, MPFR_RNDN); // mpfr_rc_dk =(add1 + add2 + add3) + mul1

    // Cast rc_dk to double.
    rc_dk = mpfr_get_d(mpfr_rc_dk, MPFR_RNDN);

    // Compute right censoring derivative for mu...

    // mul1 = rc_eta / (1 - igcdf) (same as before)

    // add1 = norm.pdf(sqrt(k / wt) * (wt / rc_mu - 1)) * (-sqrt(k / wt) * wt / rc_mu ** 2)
    mpfr_mul(add1, norm_pdf_sqrt_kwt_wtrcmu_minus_one, minus_sqrt_kwt_wt_over_rcmu_squared, MPFR_RNDN);

    // tmp_exp_logcdf = np.exp(2 * k / rc_mu + norm.logcdf(-sqrt(k / wt) * (wt / rc_mu + 1)))
    // add2 = (-2 * k / rc_mu ** 2) * tmp_exp_logcdf
    mpfr_mul(add2, minus_two_k_over_rcmu_squared, tmp_exp_logcdf, MPFR_RNDN);

    // tmp_exp_logpdf =  exp(norm.logpdf(-sqrt(k / wt) * (wt / rc_mu + 1))  + 2k/rc_mu)
    // add3 = tmp_exp_logpdf * (sqrt(k / wt) * wt / rc_mu ** 2)
    mpfr_neg(add3, minus_sqrt_kwt_wt_over_rcmu_squared, MPFR_RNDN); // add3 = sqrt(k / wt) * wt / rc_mu ** 2
    mpfr_mul(add3, add3, tmp_exp_logpdf, MPFR_RNDN);                // add3 =(sqrt(k / wt) * wt / rc_mu ** 2) * tmp_exp_logpdf


    // rc_dmu = mul1 * (add1 + add2 + add3)
    mpfr_set_d(mpfr_rc_dmu, (double) 0.0, MPFR_RNDN);    // mpfr_rc_dmu =  0.0
    mpfr_add(mpfr_rc_dmu, mpfr_rc_dmu, add1, MPFR_RNDN); // mpfr_rc_dmu =  add1
    mpfr_add(mpfr_rc_dmu, mpfr_rc_dmu, add2, MPFR_RNDN); // mpfr_rc_dmu =  add1 + add2
    mpfr_add(mpfr_rc_dmu, mpfr_rc_dmu, add3, MPFR_RNDN); // mpfr_rc_dmu =  add1 + add2 + add3
    mpfr_mul(mpfr_rc_dmu, mpfr_rc_dmu, mul1, MPFR_RNDN); // mpfr_rc_dmu = (add1 + add2 + add3) * mul1

//    mpfr_printf("mul1: %.30Rf\n",mul1);
//    mpfr_printf("add1: %.30Rf\n",add1);
//    mpfr_printf("add2: %.30Rf\n",add2);
//    mpfr_printf("add3: %.30Rf\n\n",add3);
    // rc_dtheta = rc_dmu * xt
    vector_times_scalar(mpfr_get_d(mpfr_rc_dmu, MPFR_RNDN), xt, AR_ORDER + 1, rc_dtheta);

    rc_gradient[0] = rc_dk;
    for(int i = 1; i < AR_ORDER + 2; i++){
        rc_gradient[i] = rc_dtheta[i-1];
    }

    mpfr_clear(mul1);
    mpfr_clear(add1);
    mpfr_clear(add2);
    mpfr_clear(add3);
    mpfr_clear(kwt);
    mpfr_clear(sqrt_kwt);
    mpfr_clear(minus_sqrt_kwt);
    mpfr_clear(wtrcmu);
    mpfr_clear(two_wt_sqrt_kwt);
    mpfr_clear(wtrcmu_minus_one);
    mpfr_clear(wtrcmu_plus_one);
    mpfr_clear(k_rcmu);
    mpfr_clear(sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(exp_argument);
    mpfr_clear(exp_result);
    mpfr_clear(tmp_exp_logcdf);
    mpfr_clear(tmp_exp_logpdf);
    mpfr_clear(norm_logcdf_minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(norm_pdf_sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(norm_logpdf_minus_sqrt_kwt_wtrcmu_plus_one);
    mpfr_clear(two_k_over_rcmu);
    mpfr_clear(two_over_rcmu);
    mpfr_clear(norm_cdf_sqrt_kwt_wtrcmu_minus_one);
    mpfr_clear(mpfr_rc_dk);
    mpfr_clear(mpfr_rc_dmu);
    mpfr_clear(minus_sqrt_kwt_wt_over_rcmu_squared);
    mpfr_clear(minus_two_k_over_rcmu_squared);
    mpfr_clear(mpfr_k);
    mpfr_clear(mpfr_rc_mu);
    mpfr_clear(mpfr_rc_eta);
    mpfr_clear(mpfr_wt);

}


double invgauss_negloglikel(int N_SAMPLES, double * mus, double k, double * wn, double *eta){
    // Compute the total negative log likelihood
    double tmp1_n[N_SAMPLES];
    double tmp2_n[N_SAMPLES];
    double tmp3_n[N_SAMPLES];
    double tmp4_n[N_SAMPLES];
    double tmp5_n[N_SAMPLES];
    double tmp6_n[N_SAMPLES];
    double tmp7_n[N_SAMPLES];
    double tmp8_n[N_SAMPLES];
    double negloglikel;

    // arg1 = k / (2 * np.pi * wn ** 3)
    exp_vector(wn, 3.0, N_SAMPLES, tmp2_n);
    vector_times_scalar(2 * M_PI, tmp2_n, N_SAMPLES, tmp3_n);
    scalar_divides_vector(k, tmp3_n, N_SAMPLES, tmp1_n); // tmp1_n = arg1 = k / (2 * np.pi * wn ** 3)

    // arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)
    vector_plus_vector(wn, mus, N_SAMPLES, tmp4_n, 0); // tmp4_n = wn-mus
    exp_vector(tmp4_n, 2.0, N_SAMPLES, tmp5_n); // tmp5_n = (wn-mus)**2
    vector_times_scalar(-k, tmp5_n, N_SAMPLES, tmp3_n); // tmp3_n = (-k * (wn - mus) ** 2)
    exp_vector(mus, 2.0, N_SAMPLES, tmp6_n); // tmp6_n = mus**2
    vector_times_scalar(2.0, tmp6_n, N_SAMPLES, tmp7_n); // tmp7_n = 2 * mus**2
    elementwise_product(wn, tmp7_n, N_SAMPLES, tmp8_n); // tmp8_n = (2 * mus ** 2 * wn)
    elementwise_ratio(tmp3_n, tmp8_n, N_SAMPLES, tmp2_n); // tmp2_n = arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)

    //  ps = sqrt(arg1) * exp(arg2) (probabilities)
    //  logps = log(ps) | log(ab) = log(a) + log(b) | logps = log(sqrt(arg1)) + arg2
    sqrt_vector(tmp1_n,N_SAMPLES,tmp3_n);  // tmp3_n = sqrt(tmp1_n ) = sqrt(arg1)
    log_vector(tmp3_n, N_SAMPLES, tmp4_n); // tmp4_n = log(tmp3_n) = log(sqrt(arg1))
    vector_plus_vector(tmp4_n, tmp2_n, N_SAMPLES, tmp5_n, 1);// tmp5_n = logps = log(sqrt(arg1)) + arg2

    // Of course logps should be a vector containing NEGATIVE values.
    negloglikel = - vectorvector_product(tmp5_n, eta, N_SAMPLES); // negloglikel = - dotprod(eta, logps)
    return negloglikel;
}

double mpfr_invgauss_negloglikel(int N_SAMPLES, double * mus, double k, double * wn, double *eta){
    // Compute the total negative log likelihood
    mpfr_t mpfr_mus[N_SAMPLES];
    mpfr_t mpfr_eta[N_SAMPLES];
    mpfr_t mpfr_wn[N_SAMPLES];
    mpfr_t tmp1_n[N_SAMPLES];
    mpfr_t tmp2_n[N_SAMPLES];
    mpfr_t tmp3_n[N_SAMPLES];
    mpfr_t tmp4_n[N_SAMPLES];
    mpfr_t tmp5_n[N_SAMPLES];
    mpfr_t tmp6_n[N_SAMPLES];
    mpfr_t tmp7_n[N_SAMPLES];
    mpfr_t tmp8_n[N_SAMPLES];
    mpfr_t mpfr_negloglikel;
    mpfr_t mpfr_k;
    mpfr_t neg_mpfr_k;
    mpfr_t tmp;
    double negloglikel;
    unsigned long precision = 96;
    
    for(int i = 0; i < N_SAMPLES; i++){
        mpfr_init2(tmp1_n[i],precision);
        mpfr_init2(tmp2_n[i],precision);
        mpfr_init2(tmp3_n[i],precision);
        mpfr_init2(tmp4_n[i],precision);
        mpfr_init2(tmp5_n[i],precision);
        mpfr_init2(tmp6_n[i],precision);
        mpfr_init2(tmp7_n[i],precision);
        mpfr_init2(tmp8_n[i],precision);
        mpfr_init2(mpfr_mus[i],precision);
        mpfr_init2(mpfr_eta[i],precision);
        mpfr_init2(mpfr_wn[i],precision);
        mpfr_set_d(mpfr_mus[i],mus[i],MPFR_RNDN);
        mpfr_set_d(mpfr_eta[i],eta[i],MPFR_RNDN);
        mpfr_set_d(mpfr_wn[i],wn[i],MPFR_RNDN);
    }
    
    mpfr_init2(mpfr_negloglikel,precision);
    mpfr_init2(mpfr_k,precision);
    mpfr_init2(neg_mpfr_k,precision);
    mpfr_init2(tmp,precision);
    
    mpfr_set_d(mpfr_k, k, MPFR_RNDN);
    mpfr_set_d(neg_mpfr_k,-k, MPFR_RNDN);
    
    // arg1 = k / (2 * np.pi * wn ** 3)
    mpfr_set_d(tmp, (double) 3.0 , MPFR_RNDN); // tmp = 3.0
    mpfr_pow_vector(mpfr_wn, tmp, N_SAMPLES, tmp2_n); // tmp2_n = wn^3
    mpfr_set_d(tmp, (double) 2.0 * M_PI, MPFR_RNDN); // tmp = 2.0 * pi
    mpfr_vector_times_scalar(tmp, tmp2_n, N_SAMPLES, tmp3_n); // tmp3_n = wn^3 * 2.0 * pi
    mpfr_scalar_divides_vector(mpfr_k, tmp3_n, N_SAMPLES, tmp1_n); // tmp1_n = arg1 = k / (2 * pi * wn^3)

    // arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)
    mpfr_vector_plus_vector(mpfr_wn, mpfr_mus, N_SAMPLES, tmp4_n, 0); // tmp4_n = wn - mus
    mpfr_set_d(tmp, (double) 2.0, MPFR_RNDN); // tmp = 2.0
    mpfr_pow_vector(tmp4_n, tmp, N_SAMPLES, tmp5_n); // tmp5_n = (wn-mus)**2
    mpfr_vector_times_scalar(neg_mpfr_k, tmp5_n, N_SAMPLES, tmp3_n); // tmp3_n = (-k * (wn - mus) ** 2)
    mpfr_pow_vector(mpfr_mus, tmp, N_SAMPLES, tmp6_n); // tmp6_n = mus**2
    mpfr_vector_times_scalar(tmp, tmp6_n, N_SAMPLES, tmp7_n); // tmp7_n = 2 * mus**2
    mpfr_elementwise_product(mpfr_wn, tmp7_n, N_SAMPLES, tmp8_n); // tmp8_n = (2 * mus ** 2 * wn)
    mpfr_elementwise_ratio(tmp3_n, tmp8_n, N_SAMPLES, tmp2_n); // tmp2_n = arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)

    //  ps = sqrt(arg1) * exp(arg2) (probabilities)
    //  logps = log(ps) | log(ab) = log(a) + log(b) | logps = log(sqrt(arg1)) + arg2

    mpfr_sqrt_vector(tmp1_n,N_SAMPLES,tmp3_n);  // tmp3_n = sqrt(tmp1_n) = sqrt(arg1)
    mpfr_log_vector(tmp3_n, N_SAMPLES, tmp4_n); // tmp4_n = log(tmp3_n) = log(sqrt(arg1))
    mpfr_vector_plus_vector(tmp4_n, tmp2_n, N_SAMPLES, tmp5_n, 1);// tmp5_n = logps = log(sqrt(arg1)) + arg2
    // Of course logps should be a vector containing NEGATIVE values.
    mpfr_dot_product(mpfr_eta,tmp5_n,N_SAMPLES,mpfr_negloglikel);
    mpfr_neg(mpfr_negloglikel,mpfr_negloglikel,MPFR_RNDN); // negloglikel = - dotprod(eta, logps)
    
    negloglikel = mpfr_get_d(mpfr_negloglikel, MPFR_RNDN);
    for(int i = 0; i < N_SAMPLES; i++){
        mpfr_clear(mpfr_mus[i]);
        mpfr_clear(mpfr_eta[i]);
        mpfr_clear(mpfr_wn[i]);
        mpfr_clear(tmp1_n[i]);
        mpfr_clear(tmp2_n[i]);
        mpfr_clear(tmp3_n[i]);
        mpfr_clear(tmp4_n[i]);
        mpfr_clear(tmp5_n[i]);
        mpfr_clear(tmp6_n[i]);
        mpfr_clear(tmp7_n[i]);
        mpfr_clear(tmp8_n[i]);
    }
    mpfr_clear(mpfr_k);
    mpfr_clear(mpfr_negloglikel);
    mpfr_clear(tmp);
    
    return negloglikel;
}


double nlopt_invgauss_negloglikel(unsigned int n, const double x[n], double grad[n], void * ig_data){
    struct ig_data * data;
    data = (struct ig_data *) ig_data;

    // Unpack variables
    int N_SAMPLES = data->N_SAMPLES;
    int AR_ORDER = data->AR_ORDER;
    int right_censoring = data->right_censoring;
    double xn[N_SAMPLES][AR_ORDER+1];
    for(int i = 0; i < N_SAMPLES ; i++){
        for(int j = 0; j < AR_ORDER + 1 ; j++){
            xn[i][j] = *(data->xn + i*(AR_ORDER + 1 ) + j);
            
        }
    }
    double * wn = data->wn;
    double * eta = data->eta;
    double * xt = data->xt;
    double wt = data->wt;
    double k = x[0];
    double theta[AR_ORDER + 1];
    double mus[N_SAMPLES];
    double negloglikel;
    double rc_mu;
    double rc_eta;

    for (int i = 0; i<AR_ORDER+1; i++){
        theta[i] = x[i+1];
    }

    // Compute mus
    matrixvector_product(N_SAMPLES, AR_ORDER + 1, xn, theta, mus, 0);

    mpfr_t igcdf;
    mpfr_init2(igcdf, 256);
    mpfr_set_d(igcdf,0.0, MPFR_RNDN);
    
    rc_mu = vectorvector_product(data->xt, theta, AR_ORDER + 1); // rc_mu = dotprod(xt, theta)
    // By now, the weight of the right-censoring part is equal to 1.0, the other weights decrease as time passes.
    // (they're always < 1.0)
    rc_eta = 1.0;

    if (grad) {
        ig_gradient(N_SAMPLES, AR_ORDER, xn, mus, wn, k, eta, grad);
        // Now we have to update the gradient vector with the right-censoring derivatives...
        double rc_grad[n];
        if (wt > 0.0 && right_censoring) {
            mpfr_ig_gradient_rc(AR_ORDER, k, wt, rc_mu, rc_eta, xt, igcdf, rc_grad);
            // Set the gradient...
            for(int i=0;i<AR_ORDER+2;i++){
                grad[i] = grad[i] + rc_grad[i];
            }
        }
    }
    
    negloglikel = invgauss_negloglikel(N_SAMPLES, mus, k, wn, eta);

    // mpfr_invgauss_negloglikel is extremely slow.
    // negloglikel = mpfr_invgauss_negloglikel(N_SAMPLES, mus, k, wn, eta);

    if (right_censoring){
        mpfr_t rc_term; // rc_term = log(1-igcdf)
        mpfr_init2(rc_term, 256);
        mpfr_neg(rc_term, igcdf, MPFR_RNDN);
        mpfr_add_d(rc_term, rc_term, (double) 1.0, MPFR_RNDN);
        mpfr_log(rc_term, rc_term, MPFR_RNDN);
        negloglikel = negloglikel - rc_eta * mpfr_get_d(rc_term,MPFR_RNDN);
    }
    mpfr_clear(igcdf);

    return negloglikel;
}


void test_mpfr_ig_gradient_rc(int AR_ORDER, double k, double wt, double rc_mu, double rc_eta, double * xt, double * rc_gradient){
    // wrapper for the mpfr_ig_gradient function for testing purposes in the Python library (I have no idea how to declare the igcdf pointer when creating the python binding)
    mpfr_t igcdf;
    mpfr_init2(igcdf, 256);
    mpfr_ig_gradient_rc(AR_ORDER, k, wt, rc_mu, rc_eta, xt, igcdf, rc_gradient);
    mpfr_clear(igcdf);
}
