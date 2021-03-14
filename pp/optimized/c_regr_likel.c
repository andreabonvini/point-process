#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <nlopt.h>


void elementwise_product(double * a, double * b, int dim, double * res){
    // This function returns in res the elementwiseproduct between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        res[i] = a[i]*b[i];
    }
}


void elementwise_ratio(double * a, double * b, int dim, double * res){
    // This function returns in res the elementwise ratio between a and b,
    // a and b must have the same dimension dim.
    for(int i = 0; i < dim; i++){
        res[i] = a[i]/b[i];
    }
}


double vectorvector_product(double * a, double * b, int dim){
    // This function returns in res the elementwiseproduct between a and b,
    // a and b must have the same dimension dim.
    return cblas_ddot(dim,a,1,b,1);
}


void matrixvector_product(int n_rows, int n_cols, double mat[n_rows][n_cols], double v[n_cols], double res[n_rows], int transpose){
    // This function returns in res matrix-vector dot-product between mat and v,
    // we must have that n_cols represents both the number of columns of mat and the number of elements of v.
    if (!transpose){
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, 1.f, &mat[0][0], n_cols, v, 1, 0.f, res, 1);
    }
    else{
        cblas_dgemv(CblasRowMajor, CblasTrans, n_rows, n_cols, 1.f, &mat[0][0], n_cols, v, 1, 0.f, res, 1);
    }
}


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


void vector_plus_scalar(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = vect[i] + scalar;
    }
}

void vector_times_scalar(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = vect[i] * scalar;
    }
}


void scalar_divides_vector(double scalar, double * vect, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = scalar / vect[i];
    }
}


void exp_vector(double * v, double exponent, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = pow(v[i],exponent);
    }
}


void sqrt_vector(double * v, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = sqrt(v[i]);
    }
}


void log_vector(double * v, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = log(v[i]);
    }
}


void exp_2vector(double * v, double base, int size, double * res){
    for(int i=0; i<size; i++){
        res[i] = pow(base,v[i]);
    }
}


long double norm_cdf(long double x){
    return (long double) 0.5*(1.0 + erf( (float) x/sqrt(2.0)));
}


long double norm_pdf(long double x){
    return (long double) 1.0/(sqrt(2*M_PI))*exp(-1/2*pow(x,2));
}


long double norm_logpdf(long double x){
    // Analyitical solution of the logarithm of the normal distribution's pdf
    return (long double) (-0.5*(long double) (log(2*M_PI)) - (long double)(pow(x,2.0))/2.0);
}


long double norm_logcdf(long double x){
    // We approximate the logcdf as explained here https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0
    if (x < 0.0){
        return (long double)(-0.5*((long double)pow(x,2.0)) - 4.8 + 2509.0 * (x-13.0)/(((long double)(pow(x-40.0,2))*(x-5.0))));
    }
    else{
        return 0.0;
    }
}


struct my_func_data
{
    double * xn; // [N_SAMPLES][AR_ORDER +1];
    double * eta;// [N_SAMPLES];
    double * wn; // [N_SAMPLES];
    double * xt; //[AR_ORDER + 1];
    double wt;
    int AR_ORDER;
    int N_SAMPLES;
} my_func_data;


long double compute_igcdf(long double t, long double mu, long double k){
    long double sqrt_kwt = (long double) sqrt(k/t);
    long double wtrcmu_minus_one = (long double) t / mu - 1.0;
    long double wtrcmu_plus_one = (long double) t / mu + 1.0;
    long double sqrt_kwt_wtrcmu_minus_one = sqrt_kwt * wtrcmu_minus_one; // np.sqrt(k / wt) * (wt / rc_mu - 1)
    long double minus_sqrt_kwt_wtrcmu_plus_one = - sqrt_kwt * wtrcmu_plus_one; // -np.sqrt(k / wt) * (wt / rc_mu + 1)
    long double exp_argument = (2.0 * k / mu) + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one);
    long double exp_result = exp(exp_argument);
    long double ncdf = norm_cdf(sqrt_kwt_wtrcmu_minus_one);
    return ncdf + exp_result;
}

double nlopt_invgauss_negloglikel(unsigned int n, const double x[n], double grad[n], void * my_func_data){
    struct my_func_data * data;
    data = (struct my_func_data *) my_func_data;
    int N_SAMPLES = data->N_SAMPLES;
    int AR_ORDER = data->AR_ORDER;
    double xn[N_SAMPLES][AR_ORDER+1];
    for(int i = 0; i < N_SAMPLES ; i++){
        for(int j = 0; j < AR_ORDER + 1 ; j++){
            xn[i][j] = *(data->xn + i*(AR_ORDER + 1 ) + j);
        }
    }

    double k = x[0];
    double theta[AR_ORDER + 1];
    for (int i = 0; i<AR_ORDER+1; i++){
        theta[i] = x[i+1];
    }

    // Compute mus
    double mus[N_SAMPLES];
    matrixvector_product(N_SAMPLES, AR_ORDER + 1, xn, theta, mus, 0);

    double tmp1_n[N_SAMPLES];
    double tmp2_n[N_SAMPLES];
    double tmp3_n[N_SAMPLES];
    double tmp4_n[N_SAMPLES];
    double tmp5_n[N_SAMPLES];
    double tmp6_n[N_SAMPLES];
    double tmp7_n[N_SAMPLES];
    double tmp8_n[N_SAMPLES];
    double k_grad;
    double theta_grad[n-1];
    double negloglikel;
    long double rc_mu;
    long double rc_eta;
    double wt_thr = 0.2;
    double abs_thr = 0.5;
    long double igcdf = 0.0;

    long double mul1;
    long double add1;
    long double add2;
    long double add3;
    double rc_dk;
    double rc_dtheta[AR_ORDER + 1];

    rc_mu = (long double) vectorvector_product(data->xt, theta, AR_ORDER + 1); // rc_mu = dotprod(xt, theta)
    rc_eta = (long double) data->eta[N_SAMPLES - 1]; // The weight of the right-censoring part is equal to the one of the most recent sample.

    if (grad) {

        // Firstly we have to derive k_grad as
        // k_grad = 1/2 * dotprod(v1,v2)
        // v1 = eta.
        // v2 = -1 / k + (wn - mus) ** 2 / (mus ** 2 * wn)
        vector_plus_vector(data->wn, mus, N_SAMPLES, tmp3_n, 0); // tmp3_n = wn - mus
        exp_vector(tmp3_n, 2.0, N_SAMPLES, tmp5_n); // tmp5_n = tmp3_n**2
        exp_vector(mus, 2.0, N_SAMPLES, tmp4_n); // tmp4_n = mus**2
        elementwise_product(tmp4_n, data->wn, N_SAMPLES, tmp6_n); // tmp6_n = tmp4_n * wn
        elementwise_ratio(tmp5_n, tmp6_n, N_SAMPLES, tmp7_n); // tmp7_n = tmp5_n/tmp6_n
        vector_plus_scalar(-1/k , tmp7_n, N_SAMPLES, tmp8_n);
        k_grad = 0.5 * vectorvector_product(data->eta,tmp8_n, N_SAMPLES);

        // In order to compute theta_grad we have smth like...
        // tmp =  -1 * k * eta * (wn - mus) / mus ** 3
        // theta_grad = dotprod(xn.T, tmp)
        exp_vector(mus, 3.0, N_SAMPLES, tmp2_n); // tmp2_n = mus**3
        vector_plus_vector(data->wn, mus, N_SAMPLES, tmp3_n, 0);  // tmp3_n = wn - mus
        vector_times_scalar(-k, data->eta, N_SAMPLES, tmp4_n); // tmp4_n = -1 * k * eta
        elementwise_product(tmp4_n, tmp3_n, N_SAMPLES, tmp5_n);  // tmp5_n = tmp4_n * tmp3_n
        elementwise_ratio(tmp5_n, tmp2_n, N_SAMPLES, tmp6_n); // tmp6_n = tmp5_n/tmp2_n = (-1 * k * eta * (wn-mus)) / mus**3
        matrixvector_product(N_SAMPLES, AR_ORDER + 1, xn, tmp6_n, theta_grad, 1); // thetagrad = dotprod(xn.T,tmp6_n)


        // Let's populate the gradient vector
        grad[0] = k_grad;
        for (int i = 1; i < AR_ORDER + 2; i++){
            grad[i] = theta_grad[i-1];
        }

        // Now we have to update the gradient vector with the right-censoring derivatives...
        // In order to avoid numerical problems we compute the right-censoring part only if the current time bin is
        // between (rc_mu - abs_thr) and (rc_mu + abs_thr) where abs_thr = 0.5
        if (fabs(data->wt - (double) rc_mu) < abs_thr){
//        printf("VOLEEEEVI\n");

            long double sqrt_kwt;
            long double wtrcmu_minus_one;
            long double wtrcmu_plus_one;
            long double sqrt_kwt_wtrcmu_minus_one;
            long double minus_sqrt_kwt_wtrcmu_plus_one;
            long double exp_argument;
            long double exp_result;

            // We first retrieve some useful quantities, sorry for the awful names.
            sqrt_kwt = (long double) sqrt(k/((long double) (data->wt)));
            wtrcmu_minus_one = (long double) ((long double)(data->wt)) / rc_mu - 1.0;
            wtrcmu_plus_one = (long double) ((long double) (data->wt)) / rc_mu + 1.0;
            sqrt_kwt_wtrcmu_minus_one = sqrt_kwt * wtrcmu_minus_one; // np.sqrt(k / wt) * (wt / rc_mu - 1)
            minus_sqrt_kwt_wtrcmu_plus_one = - sqrt_kwt * wtrcmu_plus_one; // -np.sqrt(k / wt) * (wt / rc_mu + 1)
            exp_argument = (long double)(2.0 * k / rc_mu) + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one);
            exp_result = (long double) exp(exp_argument);
            igcdf = norm_cdf(sqrt_kwt_wtrcmu_minus_one) + exp_result;
//            printf("\ncurrent_time: %f\n",data->wt);

//            printf("igcdf: %Lf\n\n",igcdf);
            // Compute the right censoring derivative for k...
            mul1 = ((long double) (rc_eta)) / (1.0-igcdf); // mul1 = rc_eta / (1 - igcdf)
            // add1 = norm.pdf(np.sqrt(k / wt) * (wt / rc_mu - 1)) * (1 / (2 * wt * np.sqrt(k / wt)) * (wt / rc_mu - 1))
            add1 = norm_pdf(sqrt_kwt_wtrcmu_minus_one) * 1.0/(2.0*((long double)(data->wt))*sqrt_kwt)*(((long double)(data->wt))/rc_mu + 1.0);
            // add2 = 2 / rc_mu * np.exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
            add2 = 2.0/rc_mu * exp(2.0 * k / rc_mu + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one));
            // add3 = np.exp(2 * k / rc_mu + norm.logpdf(-np.sqrt(k / wt) * (wt / rc_mu + 1))) * (-1 / (2 * wt * np.sqrt(k / wt)) * (wt / rc_mu + 1))
            add3 =(long double) exp(2 * k / rc_mu + norm_logpdf(minus_sqrt_kwt_wtrcmu_plus_one));
            rc_dk = (double) mul1 * (add1 + add2 + add3);

            // Compute right censoring derivative for mu...
            // mul1 = rc_eta / (1 - igcdf_) (same as before)
            // add1 = norm.pdf(np.sqrt(k / wt) * (wt / rc_mu - 1)) * (-np.sqrt(k / wt) * wt / rc_mu ** 2)
            add1 = norm_pdf(sqrt_kwt_wtrcmu_minus_one) * (-sqrt_kwt*((long double)(data->wt))/((long double) pow(rc_mu,2)));
            // add2 = (-2 * k / rc_mu ** 2) * np.exp(2 * k / rc_mu + norm.logcdf(-np.sqrt(k / wt) * (wt / rc_mu + 1)))
            add2 = (-2.0*k/((long double) pow(rc_mu,2)))*(long double)(exp(2.0*k/rc_mu + norm_logcdf(minus_sqrt_kwt_wtrcmu_plus_one)));
            // add3 = np.exp(2 * k / rc_mu + norm.logpdf(-np.sqrt(k / wt) * (wt / rc_mu + 1))) * (np.sqrt(k / wt) * wt / rc_mu ** 2)
            add3 = (long double) (exp(2.0*k/rc_mu+norm_logpdf(minus_sqrt_kwt_wtrcmu_plus_one))*(sqrt_kwt*((long double)(data->wt))/((long double) pow(rc_mu,2.0))));
            // rc_dtheta = mul1 * (add1 + add2 + add3) * xt.T
            vector_times_scalar((double) (mul1*(add1+add2+add3)), data->xt, AR_ORDER + 1, rc_dtheta);
            k_grad = k_grad + rc_dk;
            for(int i=0;i<AR_ORDER+1;i++){
                theta_grad[i] = theta_grad[i] + rc_dtheta[i];
            }
        }
        // Set the gradient...
        grad[0] = k_grad;
        for(int i=1;i<AR_ORDER+2;i++){
            grad[i] = theta_grad[i-1];
        }
    }

    // Compute the total negative log likelihood

    // arg1 = k / (2 * np.pi * wn ** 3)
    exp_vector(data->wn, 3.0, N_SAMPLES, tmp2_n);
    vector_times_scalar(2 * M_PI, tmp2_n, N_SAMPLES, tmp3_n);
    scalar_divides_vector(k, tmp3_n, N_SAMPLES, tmp1_n); // tmp1_n = arg1 = k / (2 * np.pi * wn ** 3)

    // arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)
    vector_plus_vector(data->wn, mus, N_SAMPLES, tmp4_n, 0); //tmp4_n = wn-mus
    exp_vector(tmp4_n, 2.0, N_SAMPLES, tmp5_n); // tmp5_n = (wn-mus)**2
    vector_times_scalar(-k, tmp5_n, N_SAMPLES, tmp3_n); // tmp3_n = (-k * (wn - mus) ** 2)
    exp_vector(mus, 2.0, N_SAMPLES, tmp6_n); // tmp6_n = mus**2
    vector_times_scalar(2.0, tmp6_n, N_SAMPLES, tmp7_n); // tmp7_n = 2 * mus**2
    elementwise_product(data->wn, tmp7_n, N_SAMPLES, tmp8_n); // tmp8_n = (2 * mus ** 2 * wn)
    elementwise_ratio(tmp3_n, tmp8_n, N_SAMPLES, tmp2_n); // tmp2_n = arg2 = (-k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)
    //  ps = sqrt(arg1) * exp(arg2)
    //  logps = log(ps) | log(ab) = log(a) + log(b) | logps = log(sqrt(arg1)) + arg2
    sqrt_vector(tmp1_n,N_SAMPLES,tmp3_n);
    log_vector(tmp3_n, N_SAMPLES, tmp4_n);
    vector_plus_vector(tmp4_n, tmp2_n, N_SAMPLES, tmp5_n, 1);
    negloglikel = - vectorvector_product(tmp5_n, data->eta, N_SAMPLES); // negloglikel = -dotprod(eta, logps)
    // We finally add the right-censoring part, note that if igcdf == 0.0 we have log(1.0-0.0) = 0.0
    negloglikel = negloglikel - rc_eta * log(1 - igcdf);

    return negloglikel;
}


double * regr_likel(
            int AR_ORDER,
            int N_SAMPLES,
            int max_steps,
            double theta0[],
            double k0,
            double xn[][AR_ORDER + 1],
            double eta[],
            double wn[],
            double xt[],
            double wt
           ){
    double x[AR_ORDER + 2];
    double * parameters = (double *) malloc (sizeof(double) * (AR_ORDER + 2));
    x[0] = k0;
    for(int i=1; i<AR_ORDER+2;i++){
        x[i] = theta0[i-1];
    }

    struct my_func_data my_func_data;
    my_func_data.xn = &xn[0][0];
    my_func_data.eta = eta;
    my_func_data.wn = wn;
    my_func_data.xt = xt;
    my_func_data.wt = wt;
    my_func_data.AR_ORDER = AR_ORDER;
    my_func_data.N_SAMPLES = N_SAMPLES;

    nlopt_opt opt = nlopt_create(NLOPT_LD_TNEWTON, AR_ORDER+2);
    nlopt_set_min_objective(opt, nlopt_invgauss_negloglikel, &my_func_data);
    nlopt_set_maxeval(opt, max_steps);
    double minf;
    nlopt_optimize(opt, x, &minf);
    for(int i = 0; i < AR_ORDER + 2; i++){
        parameters[i] = x[i];
    }
    return parameters;
}