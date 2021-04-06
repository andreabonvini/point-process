//
//  miscellania.h
//  regr_likel
//
//  Created by Andrea Bonvini on 31/03/21.
//

#ifndef miscellania_h
#define miscellania_h


#endif /* miscellania_h */

#include <math.h>
#include <gmp.h>
#include <mpfr.h>



// ----------------------------------------------------------------------------------------------------------------------------------
void elementwise_product(double * a, double * b, int dim, double * res){
    // This function returns in res the element wise product between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        res[i] = a[i]*b[i];
    }
}

void mpfr_elementwise_product(mpfr_t * a, mpfr_t * b, int dim, mpfr_t * res){
    // This function returns in res the element wise product between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        mpfr_mul(res[i],a[i],b[i],MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void elementwise_ratio(double * a, double * b, int dim, double * res){
    // This function returns in res the elementwise ratio between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        res[i] = a[i]/b[i];
    }
}
void mpfr_elementwise_ratio(mpfr_t * a, mpfr_t * b, int dim, mpfr_t * res){
    // This function returns in res the elementwise ratio between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        mpfr_div(res[i],a[i],b[i],MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
double vectorvector_product(double * a, double * b, int dim){
    // This function returns in res the dot product between a and b,
    // a and b must have the same dimension dim.
    double res = 0;
    for (int i = 0; i < dim; i++){
        res = res + a[i]*b[i];
    }
    return res;
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void matrixvector_product(int n_rows, int n_cols, double mat[n_rows][n_cols], double v[n_cols], double res[n_rows], int transpose){
    if (!transpose){
        for (int row=0; row < n_rows; row++){
            res[row] = 0;
            for(int col=0; col < n_cols; col++){
                res[row] = res[row] + mat[row][col] * v[col];
            }
        }
    }
    else{
        for (int col=0; col < n_cols; col++){
            res[col] = 0;
            for(int row=0; row < n_rows; row++){
                res[col] = res[col] + mat[row][col] * v[row];
            }
        }
    }
}

void mpfr_matrixvector_product(int n_rows, int n_cols, mpfr_t mat[n_rows][n_cols], mpfr_t v[n_cols], mpfr_t res[n_rows], int transpose){
    mpfr_t tmp;
    mpfr_init2(tmp,96);
    if (!transpose){
        for (int row=0; row < n_rows; row++){
            mpfr_set_d(res[row],0.0,MPFR_RNDN);
            for(int col=0; col < n_cols; col++){
                mpfr_mul(tmp,mat[row][col],v[col],MPFR_RNDN);
                mpfr_add(res[row], res[row], tmp, MPFR_RNDN);
            }
        }
    }
    else{
        for (int col=0; col < n_cols; col++){
            mpfr_set_d(res[col],0.0,MPFR_RNDN);
            for(int row=0; row < n_rows; row++){
                mpfr_mul(tmp,mat[row][col],v[row],MPFR_RNDN);
                mpfr_add(res[col], res[col], tmp, MPFR_RNDN);
            }
        }
    }
    mpfr_clear(tmp);
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void vector_plus_vector(double * a, double * b, int dim, double * c, int is_sum){
    // if sum == 1
    // elementwise sum between a and b
    // c = a - b
    // else
    // elementswise difference between a and b
    if (is_sum){
        for (int i = 0; i < dim; i++){
            c[i] = a[i] + b[i];
        }
    }
    else{
        for (int i = 0; i < dim; i++){
            c[i] = a[i] - b[i];
        }
    }
}

void mpfr_vector_plus_vector(mpfr_t * a, mpfr_t * b, int dim, mpfr_t * c, int is_sum){
    // if sum == 1
    // elementwise sum between a and b
    // c = a - b
    // else
    // elementswise difference between a and b
    if (is_sum){
        for (int i = 0; i < dim; i++){
            mpfr_add(c[i],a[i],b[i],MPFR_RNDN);
        }
    }
    else{
        for (int i = 0; i < dim; i++){
            mpfr_sub(c[i],a[i],b[i],MPFR_RNDN);
        }
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void vector_plus_scalar(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = vect[i] + scalar;
    }
}
void mpfr_vector_plus_scalar(double scalar, mpfr_t * vect, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_add_d(res[i],vect[i],scalar,MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void vector_times_scalar(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = vect[i] * scalar;
    }
}
void mpfr_vector_times_scalar(mpfr_t scalar, mpfr_t * vect, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_mul(res[i],vect[i],scalar,MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void scalar_divides_vector(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = scalar / vect[i];
    }
}
void mpfr_scalar_divides_vector(mpfr_t scalar, mpfr_t * vect, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_div(res[i],scalar,vect[i],MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void exp_vector(double * v, double exponent, int size, double * res){ // TODO CHANGE NAME IN pow_vector
    for(int i=0; i<size; i++){
        res[i] = pow(v[i],exponent);
    }
}
void mpfr_pow_vector(mpfr_t * v, mpfr_t exponent, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_pow(res[i], v[i], exponent, MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void sqrt_vector(double * v, int size, double * res){
    for(int i=0; i < size; i++){
        res[i] = sqrt(v[i]);
    }
}
void mpfr_sqrt_vector(mpfr_t * v, int size, mpfr_t * res){
    for(int i=0; i < size; i++){
        mpfr_sqrt(res[i], v[i], MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------------------------
void log_vector(double * v, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = log(v[i]);
    }
}
void mpfr_log_vector(mpfr_t * v, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_log(res[i], v[i], MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------

void mpfr_dot_product(mpfr_t * v1, mpfr_t * v2, int size, mpfr_t res){
    mpfr_set_d(res,0.0,MPFR_RNDN);
    mpfr_t tmp;
    mpfr_init2(tmp,96);
    for(int i=0; i<size; i++){
        mpfr_mul(tmp,v1[i],v2[i],MPFR_RNDN);
        mpfr_add(res,res,tmp,MPFR_RNDN);
    }
    mpfr_clear(tmp);
}

// ----------------------------------------------------------------------------------------------------------------------------------

void mpfr_exp_vector(mpfr_t * v, int size, mpfr_t * res){
    for(int i=0; i<size; i++){
        mpfr_exp(res[i],v[i], MPFR_RNDN);
    }
}
// ----------------------------------------------------------------------------------------------------------------------------------
