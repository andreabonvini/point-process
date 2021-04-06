#include <stdio.h>
#include <math.h>
#include <nlopt.h>
#include <stdlib.h>
#include "inverse_gaussian.h"



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
            double wt,
            int right_censoring,
            int do_global
           ){
    double x[AR_ORDER + 2];
    double * parameters = (double *) malloc (sizeof(double) * (AR_ORDER + 2));
    
    x[0] = k0;
    for(int i=1; i<AR_ORDER+2;i++){
        x[i] = theta0[i-1];
    }

    struct ig_data data;
    data.xn = &xn[0][0];
    data.eta = eta;
    data.wn = wn;
    data.xt = xt;
    data.wt = wt;
    data.AR_ORDER = AR_ORDER;
    data.N_SAMPLES = N_SAMPLES;
    data.right_censoring = right_censoring;
    
    double minf;

    // Set bounds.
    // ALERT: These bounds are problem-specific of course! The following bounds e.g. make sense
    // for fitting an inverse gaussian model to a series of inter-event intervals between
    // heart beats (PROBABLY they are valid for any physiological signal)
    double lb[AR_ORDER+2];;
    lb[0] = 0.0;
    lb[1] = 0.0;
    for (int i = 2; i < AR_ORDER+2;i++){
        lb[i] = - 1.5;
        }
    double ub[AR_ORDER+2];;
    ub[0] = 3000.0;
    ub[1] = + 5.0; //TODO this shouldn't be bounded!
    for (int i = 2; i < AR_ORDER+2;i++){
        ub[i] = + 1.5;
    }

    nlopt_result res;

    if (do_global) {
        int global_max_steps = 1000;
        // ------------------------------------------------------------- Global optimization ----------------------------------------------------
        nlopt_opt global_opt = nlopt_create(NLOPT_GN_DIRECT, AR_ORDER+2);  // NLOPT_GN_ORIG_DIRECT  0.030 | NLOPT_GN_ORIG_DIRECT_L 0.031  | NLOPT_GN_DIRECT_L = 0.032 | NLOPT_GN_DIRECT = 0.025
        nlopt_set_min_objective(global_opt, nlopt_invgauss_negloglikel, &data);
        nlopt_set_maxeval(global_opt, global_max_steps);
        nlopt_set_lower_bounds(global_opt, lb);
        nlopt_set_upper_bounds(global_opt, ub);
        res = nlopt_optimize(global_opt, x, &minf);
        if (res < 0) {
            printf("Global optimization: FAILED!\n");
        }
        nlopt_destroy(global_opt);
    }
    // ------------------------------------------------------------------ Local optimization ----------------------------------------------------
    nlopt_opt local_opt = nlopt_create(NLOPT_LD_TNEWTON, AR_ORDER+2);
    nlopt_set_min_objective(local_opt, nlopt_invgauss_negloglikel, &data);
    // nlopt_set_maxeval(local_opt, max_steps);
    double tol = 0.001;
    nlopt_set_ftol_abs(local_opt, tol);

    nlopt_set_lower_bounds(local_opt, lb);
    nlopt_set_upper_bounds(local_opt, ub);
    res = nlopt_optimize(local_opt, x, &minf);
    if (res < 0) {
        printf("Local optimization: FAILED!\n");
    }
    int n_evals = nlopt_get_numevals(local_opt);

//    if (right_censoring){
//        printf("RIGHT CENSORING n evaluations: %d\n", n_evals);
//    }
//    else{
//        printf("                n evaluations: %d\n", n_evals);
//    }


    nlopt_destroy(local_opt);

    for(int i = 0; i < AR_ORDER + 2; i++){
        parameters[i] = x[i];
    }
    return parameters;
}

