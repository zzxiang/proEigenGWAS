/** 
Haha, not all of this code is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
*/

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <math.h>
#include <boost/math/distributions/students_t.hpp>

#include "time.h"
#include <thread>
#include <chrono>

#include "global.h"
#include "genotype.h"
#include "mailman.h"
#include "helper.h"
#include "storage.h"
#include "Goptions.hpp"
#include "mailbox.h"
//#include "Gmailman.hpp"

/*
#if SSE_SUPPORT == 1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif
*/

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

//options command_line_opts;
extern Goptions goptions;
extern genotype g;
//MatrixXdr geno_matrix; //(p, n)

//Intermediate Variables
//
//How to batch columns:
//int blocksize;
extern double **partialsums;
extern double *sum_op;

// Intermediate computations in E-step.
// Size = 3^(log_3(n)) * k
extern double **yint_e;
// n X k
extern double ***y_e;

// Intermediate computations in M-step. 
// Size = nthreads X 3^(log_3(n)) * k
extern double **yint_m;
//  nthreads X log_3(n) X k
extern double ***y_m;

struct timespec t0;

MatrixXdr c; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr eveP; //(n,k) //projection matrix for EigenGWAS
std::vector<std::vector<double> > pheVal;

//int Max_it;
//int k;
//int p;
//int n; //p = Nsnp, n = Nind
//int k_orig;
//bool debug = false;
//bool check_accuracy = false;
//bool var_normalize = false;
//int accelerated_em = 0;
//double convergence_limit;
//bool memory_efficient = false;
//bool missing = false;
//bool fast_mode = true;
//bool text_version = false;
//int nthreads = 1;
//int seed;
//bool inbred = false;

//bool propc = false;
//bool eigengwas = false;
//bool rhe = false;
//int rhe_it = 10;
//bool enc = false;
//bool cld = false;
string phe_File = "";

//X*y
/*
void multiply_y_post_naive(MatrixXdr &op, int Nrows_op, MatrixXdr &res) {
	res = op * g.geno_matrix;
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op, MatrixXdr &res) {
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	for (int n_iter = 0; n_iter < Nindv; n_iter++) {
		for (int k_iter = 0; k_iter < Nrows_op; k_iter++) {
			double temp = 0;
			for (int p_iter = 0; p_iter < Nsnp; p_iter++)
				temp += op(k_iter, p_iter) * (g.get_geno(p_iter, n_iter, goptions.IsGenericVarNorm()));
			res(k_iter, n_iter) = temp;
		}
	}
}

void multiply_y_post_fast_thread(int begin, int end, MatrixXdr &op, int Ncol_op, double *yint_e, double **y_e, double *partialsums) {
	int blocksize = goptions.GetGenericMailmanBlockSize();
	for (int i = 0; i < g.Nindv; i++) {
		memset(y_e[i], 0, blocksize * sizeof(double));
	}

	for (int seg_iter = begin; seg_iter < end; seg_iter++) {
		mailman::fastmultiply_pre(g.segment_size_hori, g.Nindv, Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter], op, yint_e, partialsums, y_e);
	}
}
*/

/*
 * E-step: Compute X = D Y 
 * Y : p X n genotype matrix
 * D : k X p matrix: (C^T C)^{-1} C^{T}
 * X : k X n matrix
 *
 * op_orig : D
 * Nrows_op : k
 * res : X
 * subtract_means :
 */

/*
void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res, bool subtract_means) {

	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	MatrixXdr op;
	op = op_orig.transpose();

	if (goptions.IsGenericVarNorm() && goptions.IsGenericSubstractMean()) {
		for (int p_iter = 0; p_iter < Nsnp; p_iter++) {
			for (int k_iter = 0; k_iter < Nrows_op; k_iter++)
				op(p_iter, k_iter) = op(p_iter, k_iter) / (g.get_col_std(p_iter));
		}
	}

	#if DEBUG == 1
		if (debug) {
			print_time ();
			cout << "Starting mailman on postmultiply" << endl;
		}
	#endif

	int Ncol_op = Nrows_op;
	int nthreads = goptions.GetGenericThreads();

	nthreads = (nthreads > g.Nsegments_hori) ? g.Nsegments_hori : nthreads;

	std::thread th[nthreads];
	int perthread = g.Nsegments_hori/nthreads;
//	cout << "post: " << g.segment_size_hori << "\t" << g.Nsegments_hori << "\t" << nthreads << "\t" << perthread << endl;
	int t = 0;
	for (; t < nthreads - 1; t++) {
//		cout << "Launching " << t << endl;
		th[t] = std::thread(multiply_y_post_fast_thread, t * perthread, (t+1) * perthread, std::ref(op), Ncol_op, yint_e[t], y_e[t], partialsums[t]);
	}

//	cout << "Launching " << t << endl;
	th[t] = std::thread(multiply_y_post_fast_thread, t * perthread, g.Nsegments_hori - 1, std::ref(op), Ncol_op, yint_e[t], y_e[t], partialsums[t]);
	for (int t = 0; t < nthreads; t++) {
		th[t].join();
	}
//	cout << "Joined "<< endl;


//	int seg_iter;
//	for (seg_iter = 0; seg_iter < g.Nsegments_hori-1; seg_iter++) {
//		mailman::fastmultiply_pre(g.segment_size_hori, g.Nindv, Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter], op, yint_e, partialsums[0], y_e);
//	}


	for (int t = 1; t < nthreads; t++) {
		for (int n_iter = 0; n_iter < Nindv; n_iter++)
			for (int k_iter = 0; k_iter < Ncol_op; k_iter++)
				y_e[0][n_iter][k_iter] += y_e[t][n_iter][k_iter];
	}

	int last_seg_size = (g.Nsnp % g.segment_size_hori !=0) ? g.Nsnp % g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size, g.Nindv, Ncol_op, (g.Nsegments_hori - 1) * g.segment_size_hori, g.p[g.Nsegments_hori-1], op, yint_e[0], partialsums[0], y_e[0]);

	for (int n_iter = 0; n_iter < Nindv; n_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			res(k_iter, n_iter) = y_e[0][n_iter][k_iter];
			y_e[0][n_iter][k_iter] = 0;
		}
	}

	#if DEBUG == 1
		if (debug) {
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif

	if (!goptions.IsGenericSubstractMean())
		return;

	double *sums_elements = new double[Ncol_op];
 	memset(sums_elements, 0, Nrows_op * sizeof(int));

 	for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
 		double sum_to_calc = 0.0;
 		for (int p_iter = 0; p_iter < Nsnp; p_iter++)
 			sum_to_calc += g.get_col_mean(p_iter) * op(p_iter, k_iter);
 		sums_elements[k_iter] = sum_to_calc;
 	}

 	for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
 		for (int n_iter = 0; n_iter < Nindv; n_iter++)
 			res(k_iter, n_iter) = res(k_iter, n_iter) - sums_elements[k_iter];
 	}
}

void multiply_y_post(MatrixXdr &op, int Nrows_op, MatrixXdr &res, bool subtract_means) {
    if (goptions.IsGenericFastMode())
        multiply_y_post_fast(op, Nrows_op, res, subtract_means);
    else {
		if (goptions.IsGenericMemoryEfficient())
			multiply_y_post_naive_mem(op, Nrows_op, res);
		else
			multiply_y_post_naive(op, Nrows_op, res);
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op, MatrixXdr &res) {
	res = g.geno_matrix * op;
}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op, MatrixXdr &res) {
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	for (int p_iter = 0; p_iter < Nsnp; p_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			double temp = 0;
			for (int n_iter = 0; n_iter < Nindv; n_iter++)
				temp += g.get_geno(p_iter, n_iter, goptions.IsGenericVarNorm()) * op(n_iter, k_iter);
			res(p_iter, k_iter) = temp;
		}
	}
}

void multiply_y_pre_fast_thread(int begin, int end, MatrixXdr &op, int Ncol_op, double *yint_m, double **y_m, double *partialsums, MatrixXdr &res) {
	for (int seg_iter = begin; seg_iter < end; seg_iter++) {
		mailman::fastmultiply(g.segment_size_hori, g.Nindv, Ncol_op, g.p[seg_iter], op, yint_m, partialsums, y_m);
		int p_base = seg_iter * g.segment_size_hori;
		for (int p_iter = p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp); p_iter++) {
			for (int k_iter = 0; k_iter < Ncol_op; k_iter++)
				res(p_iter, k_iter) = y_m [p_iter - p_base][k_iter];
		}
	}
}
*/

/*
 * M-step: Compute C = Y E 
 * Y: p X n genotype matrix
 * E: n K k matrix: X^{T} (XX^{T})^{-1}
 * C = p X k matrix
 *
 * op: E
 * Ncol_op: k
 * res: C
 * subtract_means:
 */

/*
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op, MatrixXdr &res, bool subtract_means) {

	int Nsnp = g.Nsnp;

	for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
		sum_op[k_iter] = op.col(k_iter).sum();
	}

	#if DEBUG == 1
		if (debug) {
			print_time();
			cout << "Starting mailman on premultiply" << endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif

	//TODO: Memory Effecient SSE FastMultipy
	int nthreads = goptions.GetGenericThreads();
	nthreads = (nthreads > g.Nsegments_hori) ? g.Nsegments_hori : nthreads;

	std::thread th[nthreads];
	int perthread = g.Nsegments_hori / nthreads;

	int t = 0;
	for (; t < nthreads - 1; t++) {
//		cout << "Launching thread " << t << endl;
		th[t] = std::thread(multiply_y_pre_fast_thread, t * perthread, (t+1) * perthread, std::ref(op), Ncol_op, yint_m[t], y_m[t], partialsums[t], std::ref(res));
	}

	th[t] = std::thread(multiply_y_pre_fast_thread, t * perthread, g.Nsegments_hori - 1, std::ref(op), Ncol_op, yint_m[t], y_m[t], partialsums[t], std::ref(res));

	for (int t = 0; t < nthreads; t++) {
		th[t].join();
	}

//	for(int seg_iter = 0; seg_iter < g.Nsegments_hori - 1; seg_iter++){
//		mailman::fastmultiply ( g.segment_size_hori, g.Nindv, Ncol_op, g.p[seg_iter], op, yint_m, partialsums, y_m);
//		int p_base = seg_iter * g.segment_size_hori; 
//		for(int p_iter=p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp) ; p_iter++ ){
//			for(int k_iter = 0; k_iter < Ncol_op; k_iter++) 
//				res(p_iter, k_iter) = y_m [p_iter - p_base][k_iter];
//		}
//	}


	int last_seg_size = (g.Nsnp % g.segment_size_hori !=0) ? g.Nsnp % g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size, g.Nindv, Ncol_op, g.p[g.Nsegments_hori-1], op, yint_m[0], partialsums[0], y_m[0]);		
	int p_base = (g.Nsegments_hori - 1) * g.segment_size_hori;
	for (int p_iter = p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp); p_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++)
			res(p_iter, k_iter) = y_m[0][p_iter - p_base][k_iter];
	}

	#if DEBUG == 1
		if (debug) {
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif

	if (!goptions.IsGenericSubstractMean())
		return;

	for (int p_iter = 0; p_iter < Nsnp; p_iter++) {
 		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			res(p_iter, k_iter) = res(p_iter, k_iter) - (g.get_col_mean(p_iter) * sum_op[k_iter]);
			if (goptions.IsGenericVarNorm())
				res(p_iter, k_iter) = res(p_iter, k_iter) / (g.get_col_std(p_iter));		
 		}
 	}
}
*/

//y*X
/*
void multiply_y_pre(MatrixXdr &op, int Ncol_op, MatrixXdr &res, bool subtract_means) {
    if (goptions.IsGenericFastMode()) {
        multiply_y_pre_fast(op, Ncol_op, res, subtract_means);
	} else {
		if (goptions.IsGenericMemoryEfficient())
			multiply_y_pre_naive_mem(op, Ncol_op, res);
		else
			multiply_y_pre_naive(op, Ncol_op, res);
	}
}
*/

pair<double, double> get_error_norm(MatrixXdr &c) {

	int k = goptions.GetGenericMailmanBlockSize();
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	int k_orig = goptions.GetGenericEigenvecNumber();

	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(Nsnp, k);
	MatrixXdr q_t(k, Nsnp);
	q_t = Q.transpose();
	MatrixXdr b(k, Nindv);
	// Need this for subtracting the correct mean in case of missing data
	if (goptions.IsGenericMissing()) {
		multiply_y_post(q_t, k, b, false);
		// Just calculating b from seen data
		MatrixXdr M_temp(k, 1);
		M_temp = q_t * means;
		for (int j = 0; j < Nindv; j++) {
			MatrixXdr M_to_remove(k, 1);
			M_to_remove = MatrixXdr::Zero(k, 1);
			for (int i = 0; i < g.not_O_j[j].size(); i++) {
				int idx = g.not_O_j[j][i];
				M_to_remove = M_to_remove + (Q.row(idx).transpose() * g.get_col_mean(idx));
			}
			b.col(j) -= (M_temp - M_to_remove);
		}
	} else {
		multiply_y_post(q_t, k, b, true);
	}

	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l, d_l, v_l;
	if (goptions.IsGenericFastMode())
        u_l = b_svd.matrixU();
    else
        u_l = Q * b_svd.matrixU();
	v_l = b_svd.matrixV();
	d_l = MatrixXdr::Zero(k, k);
	for (int kk = 0; kk < k; kk++)
		d_l(kk, kk) = (b_svd.singularValues())(kk);

	MatrixXdr u_k, v_k, d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);
	d_k = MatrixXdr::Zero(k_orig, k_orig);
	for (int kk = 0; kk < k_orig; kk++)
		d_k(kk, kk) = (b_svd.singularValues())(kk);

	MatrixXdr b_l, b_k;
    b_l = u_l * d_l * (v_l.transpose());
    b_k = u_k * d_k * (v_k.transpose());

    if (goptions.IsGenericFastMode()) {
        double temp_k = b_k.cwiseProduct(b).sum();
        double temp_l = b_l.cwiseProduct(b).sum();
        double b_knorm = b_k.norm();
        double b_lnorm = b_l.norm();
        double norm_k = (b_knorm * b_knorm) - (2 * temp_k);
        double norm_l = (b_lnorm * b_lnorm) - (2 * temp_l);	
        return make_pair(norm_k, norm_l);
    } else {
        MatrixXdr e_l(Nsnp, Nindv);
        MatrixXdr e_k(Nsnp, Nindv);
        for (int p_iter = 0; p_iter < Nsnp; p_iter++) {
            for (int n_iter = 0; n_iter < Nindv; n_iter++) {
                e_l(p_iter, n_iter) = g.get_geno(p_iter, n_iter, goptions.IsGenericVarNorm()) - b_l(p_iter, n_iter);
                e_k(p_iter, n_iter) = g.get_geno(p_iter, n_iter, goptions.IsGenericVarNorm()) - b_k(p_iter, n_iter);
            }
        }

        double ek_norm = e_k.norm();
        double el_norm = e_l.norm();
        return make_pair(ek_norm, el_norm);
    }
}

/* Run one iteration of EM when genotypes are not missing
 * c_orig : p X k matrix
 * Output: c_new : p X k matrix
 */
MatrixXdr run_EM_not_missing(MatrixXdr &c_orig) {

	int k = goptions.GetGenericMailmanBlockSize();
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;

	#if DEBUG==1
		if (debug) {
			print_time();
			cout << "Enter: run_EM_not_missing" << endl;
		}
	#endif

 	// c_temp : k X p matrix: (C^T C)^{-1} C^{T}
	MatrixXdr c_temp(k, Nsnp);
	MatrixXdr c_new(Nsnp, k);
	c_temp = ((c_orig.transpose() * c_orig).inverse()) * (c_orig.transpose());

	#if DEBUG == 1
		if (debug) {
			print_timenl();
		}
	#endif

 	/*E-step: Compute X = Z G
 	* G: p X n genotype matrix
 	* Z: k X p matrix: (C^T C)^{-1} C^{T}
 	* X: k X n matrix
 	* x_fn: X
 	* c_temp: Z
 	*/
	MatrixXdr x_fn(k, Nindv);
	multiply_y_post(c_temp, k, x_fn, true);

	#if DEBUG == 1
		if (debug) {
			print_timenl();
		}
	#endif

	//x_temp: n X k matrix X^{T} (XX^{T})^{-1}
	MatrixXdr x_temp(Nindv, k);
	x_temp = (x_fn.transpose()) * ((x_fn*(x_fn.transpose())).inverse());

	/* M-step: X = G Z
 	* G: p X n genotype matrix
 	* Z: n X k matrix: X^{T}(XX^{T})^{-1}
 	* X = p X k matrix
 	* c_new: X
 	* x_temp: Z
 	*/
	multiply_y_pre(x_temp, k, c_new, true);

	#if DEBUG==1
		if(debug) {
			print_time();
			cout << "Exiting: run_EM_not_missing" << endl;
		}
	#endif
	return c_new;
}

MatrixXdr run_EM_missing(MatrixXdr &c_orig) {
	int k = goptions.GetGenericMailmanBlockSize();
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	MatrixXdr c_new(Nsnp, k);
	MatrixXdr mu(k, Nindv);

	// E step
	MatrixXdr c_temp(k, k);
	c_temp = c_orig.transpose() * c_orig;

	MatrixXdr T(k, Nindv);
	MatrixXdr c_fn;
	c_fn = c_orig.transpose();
	multiply_y_post(c_fn, k, T, false);

	MatrixXdr M_temp(k, 1);
	M_temp = c_orig.transpose() * means;

	for (int j = 0; j < Nindv; j++) {
		MatrixXdr D(k, k);
		MatrixXdr M_to_remove(k, 1);
		D = MatrixXdr::Zero(k, k);
		M_to_remove = MatrixXdr::Zero(k, 1);
		for (int i = 0; i < g.not_O_j[j].size(); i++) {
			int idx = g.not_O_j[j][i];
			D = D + (c_orig.row(idx).transpose() * c_orig.row(idx));
			M_to_remove = M_to_remove + (c_orig.row(idx).transpose() * g.get_col_mean(idx));
		}
		mu.col(j) = (c_temp - D).inverse() * (T.col(j) - M_temp + M_to_remove);
	}

	#if DEBUG == 1
		if (debug) {
			ofstream x_file;
//			x_file.open((string(command_line_opts.OUTPUT_PATH)+string("x_in_fn_vals.txt")).c_str());
			x_file.open((goptions.GetGenericOutFile() + string("x_in_fn_vals.txt")).c_str());
			x_file<<std::setprecision(15)<<mu<<endl;
			x_file.close();
		}
	#endif

	// M step
	MatrixXdr mu_temp(k, k);
	mu_temp = mu * mu.transpose();
	MatrixXdr T1(Nsnp, k);
	MatrixXdr mu_fn;
	mu_fn = mu.transpose();
	multiply_y_pre(mu_fn, k, T1, false);
	MatrixXdr mu_sum(k, 1);
	mu_sum = MatrixXdr::Zero(k, 1);
	mu_sum = mu.rowwise().sum();

	for (int i = 0; i < Nsnp; i++) {
		MatrixXdr D(k, k);
		MatrixXdr mu_to_remove(k, 1);
		D = MatrixXdr::Zero(k, k);
		mu_to_remove = MatrixXdr::Zero(k, 1);
		for (int j = 0; j < g.not_O_i[i].size(); j++) {
			int idx = g.not_O_i[i][j];
			D = D + (mu.col(idx) * mu.col(idx).transpose());
			mu_to_remove = mu_to_remove + (mu.col(idx));
		}
		c_new.row(i) = (((mu_temp-D).inverse()) * (T1.row(i).transpose() - (g.get_col_mean(i) * (mu_sum-mu_to_remove)))).transpose();
		double mean;
		mean = g.get_col_sum(i);
		mean = mean -  (c_orig.row(i) * (mu_sum-mu_to_remove))(0, 0);
		mean = mean * 1.0 / (Nindv - g.not_O_i[i].size());
		g.update_col_mean(i, mean);
	}

	// IMPORTANT: Update the value of means variable present locally, so that for next iteration, updated value of means is used.
	for (int i = 0; i < Nsnp; i++) {
		means(i, 0) = g.get_col_mean(i);
		// Also updating std, just for consistency, though, it is not used presently.
		stds(i, 0) = g.get_col_std(i);
	}

	return c_new;
}

MatrixXdr run_EM(MatrixXdr &c_orig) {
	if (goptions.IsGenericMissing()) {
		return run_EM_missing(c_orig);
	} else {
		return run_EM_not_missing(c_orig);
	}
}

void print_vals() {
	int k = goptions.GetGenericMailmanBlockSize();
	int Nsnp = g.Nsnp;
	int Nindv = g.Nindv;
	int k_orig = goptions.GetGenericEigenvecNumber();

	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(Nsnp, k);
	MatrixXdr q_t(k, Nsnp);
	q_t = Q.transpose();
	MatrixXdr b(k, Nindv);

	// Need this for subtracting the correct mean in case of missing data
	if (goptions.IsGenericMissing()) {
		multiply_y_post(q_t, k, b, false);
		// Just calculating b from seen data
		MatrixXdr M_temp(k, 1);
		M_temp = q_t * means;
		for (int j = 0; j < Nindv; j++) {
			MatrixXdr M_to_remove(k, 1);
			M_to_remove = MatrixXdr::Zero(k, 1);
			for (int i = 0; i < g.not_O_j[j].size(); i++) {
				int idx = g.not_O_j[j][i];
				M_to_remove = M_to_remove + (Q.row(idx).transpose() * g.get_col_mean(idx));
			}
			b.col(j) -= (M_temp - M_to_remove);
		}
	} else {
		multiply_y_post(q_t, k, b, true);
	}

	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l;
	u_l = b_svd.matrixU();
	MatrixXdr v_l;
	v_l = b_svd.matrixV();
	MatrixXdr u_k;
	MatrixXdr v_k, d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);

	ofstream evec_file;
	evec_file.open((goptions.GetGenericOutFile() + string("evecs.txt")).c_str());
	evec_file << std::setprecision(15) << Q * u_k << endl;
	evec_file.close();
	ofstream eval_file;
	eval_file.open((goptions.GetGenericOutFile() + string("evals.txt")).c_str());
	for(int kk = 0; kk < k_orig; kk++)
		eval_file << std::setprecision(15) << (b_svd.singularValues())(kk) * (b_svd.singularValues())(kk)/g.Nsnp<<endl;
	eval_file.close();

	eveP = v_l.leftCols(k_orig);
	ofstream proj_file;
	proj_file.open((goptions.GetGenericOutFile() + string("projections.txt")).c_str());
	proj_file << std::setprecision(15)<< v_k << endl;
	proj_file.close();

	if (goptions.IsGenericDebug()) {
		ofstream c_file;
		c_file.open((goptions.GetGenericOutFile()+string("cvals.txt")).c_str());
		c_file << std::setprecision(15) << c << endl;
		c_file.close();

		ofstream means_file;
		means_file.open((goptions.GetGenericOutFile()+string("means.txt")).c_str());
		means_file << std::setprecision(15) << means << endl;
		means_file.close();

		d_k = MatrixXdr::Zero(k_orig, k_orig);
		for (int kk =0; kk < k_orig; kk++)
			d_k(kk,kk)  =(b_svd.singularValues())(kk);
		MatrixXdr x_k;
		x_k = d_k * (v_k.transpose());
		ofstream x_file;
		x_file.open((goptions.GetGenericOutFile() + string("xvals.txt")).c_str());
		x_file << std::setprecision(15) << x_k.transpose() << endl;
		x_file.close();
	}
}

void EigenGWAS(MatrixXdr IndEigenVec) {

	clock_t eg_begin = clock();
	cout << "---------------EigenGWAS scan-------------" << endl;
	cout << "SNP number: " << g.Nsnp << endl;
	cout << "Sample size: " << g.Nindv << endl;

	cout << "Eigenvector number: " << IndEigenVec.cols() <<endl;

	MatrixXdr evePS(IndEigenVec.rows(), IndEigenVec.cols());
	cout << "Standardization eigenvec" << endl;
	for (int i = 0; i < evePS.cols(); i++) {
		double sum = 0, sd = 0, eSq = 0, m = 0;
		for (int j = 0; j < evePS.rows(); j++) {
			sum += IndEigenVec(j, i);
			eSq += IndEigenVec(j, i) * IndEigenVec(j, i);
		}
		m = sum / IndEigenVec.rows();
		sd = sqrt((eSq - m * m * IndEigenVec.rows()) / (IndEigenVec.rows() - 1));
		for (int j = 0; j < IndEigenVec.rows(); j++) {
			evePS(j, i) = (IndEigenVec(j, i) - m) / sd;
		}
	}

	//using mailman
	MatrixXdr EgBeta(g.Nsnp, IndEigenVec.cols());
	multiply_y_pre(evePS, evePS.cols(), EgBeta, true);

	EgBeta = EgBeta / g.Nindv;
	for (int i = 0; i < EgBeta.cols(); i++) {
		cout << EgBeta(0, i) <<endl;
	}

	using boost::math::students_t;
	students_t Tdist(g.Nindv - 1);

	for (int i = 0; i < EgBeta.cols(); i++) {
		cout << "Scanning eigenvector " << i + 1 << endl;
		ofstream e_file;
		e_file.open((goptions.GetGenericOutFile() + string("eg."+std::to_string(i+1)+".txt")).c_str());
		e_file << "CHR\tSNP\tPOS\tBP\tA1\tA2\tBeta\tSE\tT-stat\tP" << endl;
		for (int j = 0; j <EgBeta.rows(); j++) {
			double seB = sqrt((1 - pow(EgBeta(j, i), 2))/(g.Nindv - 1));
			double t_stat = EgBeta(j, i) / seB;
			cout << "MK: " << j << " infor=" << g.get_bim_info(j) << " egB=" << EgBeta(j, i) << " seB=" << seB << " t=" << t_stat << endl;
			double pt2tail = cdf(complement(Tdist, fabs(t_stat))) * 2;
			e_file << g.get_bim_info(j) << "\t" << std::setprecision(8) << EgBeta(j, i) << "\t" << seB << "\t" << t_stat << "\t" << pt2tail <<endl;
		}
		e_file.close();
	}

	clock_t eg_end = clock();
	double eg_time = double(eg_end - eg_begin) / CLOCKS_PER_SEC;
	cout << "EigenGWAS total time " << eg_time << endl;
}

void RHE_read_pheno() {
   	ifstream ifs(goptions.GetGenericPhenoFile().c_str(), ios::in);
	std::string temp;

	while (std::getline(ifs, temp)) {
    std::istringstream buffer(temp);
    std::vector<double> line((std::istream_iterator<double>(buffer)),
                             std::istream_iterator<double>());
    	pheVal.push_back(line);
	}

	// for (auto it = numbers.begin(); it != numbers.end(); it++) {
	// 	vector<double> n1 = *it;
	// 	for (auto it1 = n1.begin(); it1 != n1.end(); it1++) {
	// 		cout << (*it1) << " ";
	// 	}
	// 	cout << endl;
	// }
}

void RHE_reg(int seed, int iter, int phe_idx) {

	cout << "Randomized HE (mailman), iteration for " << iter << " times" << endl;
	clock_t heReg_begin = clock();

	MatrixXdr yval(g.Nindv, 1);
	int cnt = 0;
	for (int i = 0; i < yval.rows(); i++) {
		yval(i, 0) = pheVal[i][phe_idx];
	}
	cout<<yval(0, 0)<<" "<<yval(1, 0)<<endl;

	double LB = 0;
	MatrixXdr Bz(g.Nindv, iter);
	srand(goptions.GetGenericSeed());
	std::default_random_engine generator(goptions.GetGenericSeed());
	std::normal_distribution<double> norm_dist(0, 1.0);
	for (int i = 0; i < Bz.rows(); i++) {
		for (int j = 0; j < Bz.cols(); j++) {
			Bz(i, j) = norm_dist(generator);
		}
	}

	//geno_matrix * Bz; //(p x n) * (n x iter) = p x iter
	MatrixXdr T1(g.Nsnp, iter);
	cout << "Here T1=G^T * z" << endl;
	multiply_y_pre(Bz, iter, T1, true);
	MatrixXdr T1_transpose(iter, g.Nsnp);
	T1_transpose = T1.transpose(); //iter X p
	cout << "Here T1.transpose" << endl;

	MatrixXdr T2(iter, g.Nindv);
	cout << "Here T2=T1^T * G^T" << endl;
	cout << T1_transpose.rows() << " " << T1_transpose.cols() << " " << endl;
	multiply_y_post(T1_transpose, iter, T2, true);

	for (int i = 0; i < T2.rows(); i++) {
		for (int j = 0; j < T2.cols(); j++) {
			LB += T2(i, j) * T2(i, j);
		}
	}

	cout << "LB " << LB << endl;
	LB = LB / (1.0 * iter * g.Nsnp * g.Nsnp);
	cout << "LB2 " << LB << endl;
	double me = (LB - g.Nindv) / (1.0 * g.Nindv * g.Nindv);
	cout << "Me " << 1/me << endl;
	clock_t heReg_end = clock();

	double heReg_time = double(heReg_end - heReg_begin) / CLOCKS_PER_SEC;
	cout << "RHE time " << heReg_time << endl;
}

void ENC(int seed, int kval) {
	cout << "ENC generates " << kval << " tags for " << g.Nindv <<" samples" <<endl;
	clock_t ENC_begin = clock();

	srand(goptions.GetGenericSeed());
	std::default_random_engine generator(goptions.GetGenericSeed());
	std::normal_distribution<double> norm_dist(0, 1.0);

	MatrixXdr Bz(kval, g.Nsnp);
	for (int i = 0; i < Bz.rows(); i++)
		for (int j = 0; j < Bz.cols(); j++)
			Bz(i, j) = norm_dist(generator);

	MatrixXdr encG(kval, g.Nindv);
	multiply_y_post(Bz, kval, encG, true);

	ofstream e_file;
	e_file.open((goptions.GetGenericOutFile() + string(".enc.txt")).c_str());
	for (int i = 0; i < encG.cols(); i++) {
		for (int j = 0; j < encG.rows(); j++) {
			e_file<<encG(j, i);
			if (j != (encG.rows() - 1)) e_file << " ";
		}
		e_file << endl;
	}

	e_file.close();
	clock_t ENC_end = clock();
	double ENC_time = double(ENC_end - ENC_begin) / CLOCKS_PER_SEC;
	cout<< "Save encG to " <<(goptions.GetGenericOutFile() + string(".enc.txt")).c_str() <<endl;
	cout << "ENC time " << ENC_time << endl;
}

void CLD() {

}

void setMem() {
	int blocksize = goptions.GetGenericMailmanBlockSize();
	int hsegsize = g.segment_size_hori;	// = log_3(n)
	int hsize = pow(3, hsegsize);
//	int vsegsize = g.segment_size_ver;	// = log_3(p)
//	int vsize = pow(3, vsegsize);

	sum_op = new double[blocksize];
	partialsums = new double*[goptions.GetGenericThreads()];
	yint_m = new double*[goptions.GetGenericThreads()];
	yint_e = new double*[goptions.GetGenericThreads()];

	for (int t = 0; t < goptions.GetGenericThreads(); t++) {
		partialsums[t] = new double [blocksize];
		yint_m[t] = new double [hsize * blocksize];
		memset(yint_m[t], 0, hsize * blocksize * sizeof(double));
		yint_e[t] = new double [hsize * blocksize];
		memset(yint_e[t], 0, hsize * blocksize * sizeof(double));
	}

	y_e = new double**[goptions.GetGenericThreads()];
	y_m = new double**[goptions.GetGenericThreads()];
	for (int t = 0; t < goptions.GetGenericThreads(); t++) {
		y_e[t] = new double*[g.Nindv];
		for (int i = 0; i < g.Nindv; i++) {
			y_e[t][i] = new double[blocksize];
			memset(y_e[t][i], 0, blocksize * sizeof(double));
		}
		y_m[t] = new double*[hsegsize];
		for (int i = 0; i < hsegsize; i++) {
			y_m[t][i] = new double[blocksize];
			memset(y_m[t][i], 0, blocksize * sizeof(double));
		}
	}
}

void cleanMem() {
	int nthreads = goptions.GetGenericThreads();
	int hsegsize = g.segment_size_hori;	// = log_3(n)
	delete[] sum_op;
	for (int t = 0; t < nthreads; t++) {
		delete[] yint_e[t];
	}
	delete[] yint_e;

	for (int t = 0; t < nthreads; t++) {
		delete[] yint_m[t];
		delete[] partialsums[t];
	}
	delete[] yint_m;
	delete[] partialsums;

	for (int t = 0; t < nthreads; t++) {
		for (int i  = 0; i < hsegsize; i++)
			delete[] y_m[t][i];
		delete[] y_m[t];
	}
	delete[] y_m;

	for (int t = 0; t < nthreads; t++) {
		for (int i  = 0; i < g.Nindv; i++)
			delete[] y_e[t][i]; 
		delete[] y_e[t];
	}
	delete[] y_e;
}

void ProPC() {

	int k = goptions.GetGenericMailmanBlockSize();
	int Nsnp = g.Nsnp;
	clock_t pc_begin = clock();

	pair<double,double> prev_error = make_pair(0.0, 0.0);
	bool toStop = false;
	
	//	if (convergence_limit != -1)
	if (goptions.GetPropcConvergenceLimit() != -1)
		toStop = true;

	double prevnll = 0.0;

	c.resize(Nsnp, k);
	means.resize(Nsnp, 1);
	stds.resize(Nsnp, 1);
	for (int i = 0; i < Nsnp; i++) {
		means(i, 0) = g.get_col_mean(i);
		stds(i, 0) = g.get_col_std(i);
	}

	std::default_random_engine generator(goptions.GetGenericSeed());
	std::normal_distribution<double> norm_dist(0, 1.0);
	for (int i = 0; i < c.rows(); i++)
		for (int j = 0; j < c.cols(); j++)
			c(i, j) = norm_dist(generator);
	// Initial intermediate data structures
	// Operate in blocks to improve caching
	//

	ofstream c_file;
	if (goptions.IsGenericDebug()) {
		// c_file.open((string(command_line_opts.OUTPUT_PATH) + string("cvals_orig.txt")).c_str());
		c_file.open((goptions.GetGenericOutFile() + string("cvals_orig.txt")).c_str());
		c_file << std::setprecision(15) << c << endl;
		c_file.close();
		printf("Read Matrix\n");
	}

	cout << "Running on Dataset of " << g.Nsnp << " SNPs and " << g.Nindv << " Individuals" << endl;

	#if SSE_SUPPORT == 1
		if (fast_mode)
			cout<<"Using Optimized SSE FastMultiply"<<endl;
	#endif

	clock_t it_begin = clock();
	for (int i = 0; i < goptions.GetPropcMaxIteration(); i++) {
		MatrixXdr c1, c2, cint, r, v;
		double a, nll;
		if (goptions.IsGenericDebug()) {
			print_time ();
			cout << "*********** Begin epoch " << i << "***********" << endl;
		}

//		if (accelerated_em != 0) {
		if (goptions.GetPropcAcceleratedEM() != 0) {
			#if DEBUG == 1
			if (debug) {
				print_time();
				cout << "Before EM" << endl;
			}
			#endif
			c1 = run_EM(c);
			c2 = run_EM(c1);
			#if DEBUG == 1
			if (debug) {
				print_time();
				cout << "After EM but before acceleration" << endl;
			}
			#endif
			r = c1 - c;
			v = (c2 - c1) - r;
			a = -1.0 * r.norm() / (v.norm()) ;
			if (goptions.GetPropcAcceleratedEM() == 1) {
				if (a > -1) {
					a = -1;
					cint = c2;
				} else {
					cint = c - 2 * a * r + a * a * v;
					nll = get_error_norm(cint).second;
					if ( i > 0 ) {
						while ( nll>prevnll && a < -1) {
							a = 0.5 * (a-1);
							cint = c - 2 * a * r + (a * a * v);
							nll = get_error_norm(cint).second;
						}
					}
				}
				c = cint;
			} else if (goptions.GetPropcAcceleratedEM() == 2) {
				cint = c - 2 * a * r + a * a * v;
				c = cint;
				// c = run_EM(cint);
			}
		} else {
			c = run_EM(c);
		}

		if (goptions.GetPropcAcceleratedEM() == 1 || goptions.IsPropcAccuracy() || toStop) {
			pair<double, double> e = get_error_norm(c);
				prevnll = e.second;
				if (goptions.IsPropcAccuracy())
					cout << "Iteration " << i+1 << "  " << std::setprecision(15) << e.first << "  " << e.second << endl;
				if (abs(e.first - prev_error.first) <= goptions.GetPropcConvergenceLimit()) {
					cout << "Breaking after " << i+1 << " iterations" << endl;
					break;
				}
				prev_error = e;
		}

		if (goptions.IsGenericDebug()) {
				print_time();
				cout << "*********** End epoch " << i << "***********" << endl;
		}
	}

	clock_t it_end = clock();

	print_vals();

	clock_t pc_end = clock();
	double avg_it_time = double(it_end - it_begin) / (goptions.GetPropcMaxIteration() * 1.0 * CLOCKS_PER_SEC);
	double total_time = double(pc_end - pc_begin) / CLOCKS_PER_SEC;
	cout << "\nAVG Iteration Time: " << avg_it_time << "\nTotal runtime: " << total_time << endl;
}

int main(int argc, char const *argv[]) {

	try {
    	goptions.ParseOptions(argc, argv);
//      PrintOptions(goptions);
	}
	catch (OptionsExitsProgram){}

	auto start = std::chrono::system_clock::now();
	clock_t io_begin = clock();
    clock_gettime(CLOCK_REALTIME, &t0);

	// parse_args(argc, argv);

	//TODO: Memory Effecient Version of Mailman

//	memory_efficient = goptions.IsGenericMemoryEfficient();
//	text_version = goptions.IsGenericTextMode();
//	fast_mode = goptions.IsGenericFastMode();
//	missing = goptions.IsGenericMissing();
//	debug = goptions.IsGenericDebug();
//	nthreads = goptions.GetGenericThreads();
//	seed = goptions.GetGenericSeed();
//	k_orig = goptions.GetGenericEigenvecNumber();
//	var_normalize = goptions.IsGenericVarNorm();

//	check_accuracy = goptions.IsPropcAccuracy();
//	MAX_ITER = goptions.GetPropcMaxIteration();
//	accelerated_em = goptions.GetPropcAcceleratedEM();
//	convergence_limit = goptions.GetPropcConvergenceLimit();

//	k = (int) ceil(goptions.GetGenericEigenvecNumber()/10.0) * 10;

//	propc = command_line_opts.propc;
//	propc = goptions.CheckPropcMasterOption();

//	eigengwas = command_line_opts.scan;
//	eigengwas = goptions.CheckEigenGWASMasterOption();

//	rhe_it = command_line_opts.rhe_it;
//	rhe_it = goptions.GetGenericIteration();

//	rhe = command_line_opts.rhe;
//	rhe = goptions.CheckRandHEMasterOption();

//	enc = command_line_opts.enc;
//	enc = goptions.CheckEncMasterOption();

//	cld = command_line_opts.cld;
//	cld = goptions.CheckCLDMasterOption();

//read data--universal step
//	g.set_poptype(inbred);

	srand(goptions.GetGenericSeed());

	g.read_geno(goptions.GetGenericGenoFile(),
	goptions.IsGenericTextMode(),
	goptions.IsGenericFastMode(),
	goptions.IsGenericMissing(),
	goptions.IsGenericMemoryEfficient(),
	goptions.IsGenericVarNorm());

//	g2.read_geno(goptions.GetGenericGenoFile(), text_version, fast_mode, missing, memory_efficient, var_normalize);

/*	
	if (text_version) {
		if (fast_mode)
			// g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH, missing);
			g.read_txt_mailman(goptions.GetGenericGenoFile(), missing);
		else
			// g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH, missing);
			g.read_txt_naive(goptions.GetGenericGenoFile(), missing);
	} else {
		clock_t plink_read_begin = clock();
		// g.read_plink(command_line_opts.GENOTYPE_FILE_PATH, missing, fast_mode);
		g.read_plink(goptions.GetGenericGenoFile(), missing, fast_mode);
		clock_t plink_read_end = clock();
		double plink_io_time = double(plink_read_end - plink_read_begin) / CLOCKS_PER_SEC;
		cout << "Reading plink data in " << plink_io_time << "s" <<endl;
	}
*/
	//TODO: Implement these codes.
/*	
	if (missing && !fast_mode) {
		cout << "Missing version works only with mailman i.e. fast mode\n EXITING..." << endl;
		exit(-1);
	}
	if (fast_mode && memory_efficient) {
		cout << "Memory effecient version for mailman EM not yet implemented" << endl;
		cout << "Ignoring Memory effecient Flag" << endl;
	}
	if (missing && var_normalize) {
		cout << "Missing version works only without variance normalization\n EXITING..." << endl;
		exit(-1);
	}
*/
//	command_line_opts.l = k - k_orig;
//	p = g.Nsnp;
//	n = g.Nindv;

//	Gmailman gman(opt2, g2);
//	Gmailman gman(goptions, g);
//	if (inbred) cout<< "Inbred mode is switched on." << endl;
	if (!goptions.IsGenericFastMode() && !goptions.IsGenericMemoryEfficient()) {
		cout << "Genotype standardization..." << endl;
//		geno_matrix.resize(g.Nindv, g.Nsnp);
//		g.generate_eigen_geno(geno_matrix, goptions.IsGenericVarNorm());
	}

	clock_t io_end = clock();
	double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
	cout << "IO Time: " << io_time << endl;

	if (goptions.CheckPropcMasterOption()) {
		setMem();
		ProPC();
		cleanMem();
	} else if (goptions.CheckEigenGWASMasterOption()) {
		setMem();
		ProPC();
		EigenGWAS(eveP);
		cleanMem();
	} else if (goptions.CheckRandHEMasterOption()) {
//		RHE_read_pheno(command_line_opts.PHENO_FILE);
//		setMem(command_line_opts.rhe_it);
//		RHE_reg(seed, command_line_opts.rhe_it, command_line_opts.pheno_num);
		RHE_read_pheno();
		setMem();
		RHE_reg(goptions.GetGenericSeed(), goptions.GetGenericIteration(), goptions.GetGenericPhenoNum()[0]);
		cleanMem();
	} else if (goptions.CheckEncMasterOption()) {
		setMem();

//		gmanENC(seed, goptions.GetEncK(), gman);
		ENC(goptions.GetGenericSeed(), goptions.GetEncK());
		cleanMem();
	} else if (goptions.CheckCLDMasterOption()) {
		CLD();
	}

	std::chrono::duration<double> wctduration = std::chrono::system_clock::now() - start;
	cout << "Wall clock time = " << wctduration.count() << endl;
	return 0;
}
