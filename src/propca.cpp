/** 
 All of this code is written by Aman Agrawal 
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

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
#include "helper.h"
#include "storage.h"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

//Intermediate Variables
//
// How to batch columns: 
int blocksize;
double **partialsums;
double *sum_op;		

// Intermediate computations in E-step. 
// Size = 3^(log_3(n)) * k
double **yint_e;
//  n X k
double ***y_e;

// Intermediate computations in M-step. 
// Size = nthreads X 3^(log_3(n)) * k
double **yint_m;
//  nthreads X log_3(n) X k
double ***y_m;

struct timespec t0;

clock_t total_begin = clock();

genotype g;
MatrixXdr geno_matrix; //(p,n)

int MAX_ITER;
int k, p, n; //p = Nsnp, n = Nind
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)

//projection matrix for EigenGWAS
//Size = sample_size, k--the number of eigenvector calculated
MatrixXdr eveP;

options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize = false;
int accelerated_em = 0;
double convergence_limit;
bool memory_efficient = false;
bool missing = false;
bool fast_mode = true;
bool text_version = false;
int nthreads = 1;

bool propc = false;

bool given_seed = false;
int seed;

bool scan = false;
bool inbred = false;

bool rhe = false;
int rhe_it = 10;

bool enc = false;
bool cld = false;

//X*y

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op, MatrixXdr &res) {
	res = op * geno_matrix;
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op, MatrixXdr &res) {
	for (int n_iter = 0; n_iter < n; n_iter++) {
		for (int k_iter = 0; k_iter < Nrows_op; k_iter++) {
			double temp = 0;
			for (int p_iter = 0; p_iter < p; p_iter++)
				temp += op(k_iter, p_iter) * (g.get_geno(p_iter, n_iter, var_normalize));
			res(k_iter, n_iter) = temp;
		}
	}
}

void multiply_y_post_fast_thread(int begin, int end, MatrixXdr &op, int Ncol_op, double *yint_e, double **y_e, double *partialsums) {
	for (int i = 0; i < g.Nindv; i++) {
		memset(y_e[i], 0, blocksize * sizeof(double));
	}

	for (int seg_iter = begin; seg_iter < end; seg_iter++) {
		mailman::fastmultiply_pre(g.segment_size_hori, g.Nindv, Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter], op, yint_e, partialsums, y_e);
	}
}

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
void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res, bool subtract_means) {

	MatrixXdr op;
	op = op_orig.transpose();

	if (var_normalize && subtract_means) {
		for (int p_iter = 0; p_iter < p; p_iter++) {
			for (int k_iter = 0; k_iter < Nrows_op; k_iter++)		
				op(p_iter, k_iter) = op(p_iter, k_iter) / (g.get_col_std(p_iter));		
		}
	}

	#if DEBUG == 1
		if(debug){
			print_time ();
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif

	int Ncol_op = Nrows_op;

	nthreads = (nthreads > g.Nsegments_hori) ? g.Nsegments_hori : nthreads;

	std::thread th[nthreads];
	int perthread = g.Nsegments_hori/nthreads;
//	cout << "post: " << g.segment_size_hori << "\t" << g.Nsegments_hori << "\t" << nthreads << "\t" << perthread << endl;
	int t = 0;
	for (; t < nthreads - 1; t++) {
//		cout << "Launching " << t << endl;
		th[t] = std::thread (multiply_y_post_fast_thread, t * perthread, (t+1) * perthread, std::ref(op), Ncol_op, yint_e[t], y_e[t], partialsums[t]);
	}

//	cout << "Launching " << t << endl;
	th[t] = std::thread(multiply_y_post_fast_thread, t * perthread, g.Nsegments_hori - 1, std::ref(op), Ncol_op, yint_e[t], y_e[t], partialsums[t]);
	for (int t = 0; t < nthreads; t++) {
		th[t].join();
	}
//	cout << "Joined "<< endl;

/*
	int seg_iter;
	for(seg_iter = 0; seg_iter < g.Nsegments_hori-1; seg_iter++){
		mailman::fastmultiply_pre (g.segment_size_hori, g.Nindv, Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter], op, yint_e, partialsums[0], y_e);
	}
*/

	for (int t = 1; t < nthreads; t++) {
		for (int n_iter = 0; n_iter < n; n_iter++) 
			for (int k_iter = 0; k_iter < Ncol_op; k_iter++)
				y_e[0][n_iter][k_iter] += y_e[t][n_iter][k_iter];
	}

	int last_seg_size = (g.Nsnp % g.segment_size_hori !=0) ? g.Nsnp % g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size, g.Nindv, Ncol_op, (g.Nsegments_hori-1) * g.segment_size_hori, g.p[g.Nsegments_hori-1], op, yint_e[0], partialsums[0], y_e[0]);

	for (int n_iter = 0; n_iter < n; n_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			res(k_iter,n_iter) = y_e[0][n_iter][k_iter];
			y_e[0][n_iter][k_iter] = 0;
		}
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif

	if (!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset(sums_elements, 0, Nrows_op * sizeof(int));

 	for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {		
 		double sum_to_calc = 0.0;
 		for (int p_iter = 0; p_iter < p; p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter) * op(p_iter, k_iter);		
 		sums_elements[k_iter] = sum_to_calc;
 	}
 	for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {		
 		for (int n_iter=0; n_iter<n; n_iter++)
 			res(k_iter,n_iter) = res(k_iter, n_iter) - sums_elements[k_iter];
 	}
}

void multiply_y_post(MatrixXdr &op, int Nrows_op, MatrixXdr &res, bool subtract_means) {
    if (fast_mode)
        multiply_y_post_fast(op, Nrows_op, res, subtract_means);
    else {
		if (memory_efficient)
			multiply_y_post_naive_mem(op, Nrows_op, res);
		else
			multiply_y_post_naive(op, Nrows_op, res);
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op, MatrixXdr &res) {
	res = geno_matrix * op;
}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op, MatrixXdr &res) {
	for (int p_iter = 0; p_iter < p; p_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			double temp = 0;
			for (int n_iter = 0; n_iter < n; n_iter++)
				temp += g.get_geno(p_iter, n_iter, var_normalize) * op(n_iter, k_iter);
			res(p_iter, k_iter) = temp;
		}
	}
}

void multiply_y_pre_fast_thread(int begin, int end, MatrixXdr &op, int Ncol_op, double *yint_m, double **y_m, double *partialsums, MatrixXdr &res) {
	for(int seg_iter = begin; seg_iter < end; seg_iter++) {
		mailman::fastmultiply(g.segment_size_hori, g.Nindv, Ncol_op, g.p[seg_iter], op, yint_m, partialsums, y_m);
		int p_base = seg_iter * g.segment_size_hori;
		for(int p_iter = p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp); p_iter++) {
			for(int k_iter = 0; k_iter < Ncol_op; k_iter++)
				res(p_iter, k_iter) = y_m [p_iter - p_base][k_iter];
		}
	}
}

/*
 * M-step: Compute C = Y E 
 * Y : p X n genotype matrix
 * E : n K k matrix: X^{T} (XX^{T})^{-1}
 * C = p X k matrix
 *
 * op : E
 * Ncol_op : k
 * res : C
 * subtract_means :
 */
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op, MatrixXdr &res, bool subtract_means) {

	cout<<"cgb1"<<endl;	
	for(int k_iter = 0; k_iter < Ncol_op; k_iter++) {
		sum_op[k_iter] = op.col(k_iter).sum();
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif

	//TODO: Memory Effecient SSE FastMultipy

	nthreads = (nthreads > g.Nsegments_hori) ? g.Nsegments_hori : nthreads;

	std::thread th[nthreads];
	int perthread = g.Nsegments_hori/nthreads;

	int t = 0;
	for (; t < nthreads - 1; t++) {
//		cout << "Launching thread " << t << endl;
		th[t] = std::thread(multiply_y_pre_fast_thread, t * perthread, (t+1) * perthread, std::ref(op), Ncol_op, yint_m[t], y_m[t], partialsums[t], std::ref(res));
	}
	cout<<"cgb2"<<endl;	

	th[t] = std::thread(multiply_y_pre_fast_thread, t * perthread, g.Nsegments_hori - 1, std::ref(op), Ncol_op, yint_m[t], y_m[t], partialsums[t], std::ref(res));
	cout<<"cgb3"<<endl;	

	for (int t = 0; t < nthreads; t++) {
		th[t].join();
	}

/*
	for(int seg_iter = 0; seg_iter < g.Nsegments_hori - 1; seg_iter++){
		mailman::fastmultiply ( g.segment_size_hori, g.Nindv, Ncol_op, g.p[seg_iter], op, yint_m, partialsums, y_m);
		int p_base = seg_iter * g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp) ; p_iter++ ){
			for(int k_iter = 0; k_iter < Ncol_op; k_iter++) 
				res(p_iter, k_iter) = y_m [p_iter - p_base][k_iter];
		}
	}
*/
	cout<<"cgb4"<<endl;	

	int last_seg_size = (g.Nsnp % g.segment_size_hori !=0 ) ? g.Nsnp % g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size, g.Nindv, Ncol_op, g.p[g.Nsegments_hori-1], op, yint_m[0], partialsums[0], y_m[0]);		
	int p_base = (g.Nsegments_hori - 1) * g.segment_size_hori;
	for (int p_iter = p_base; (p_iter < p_base + g.segment_size_hori) && (p_iter < g.Nsnp); p_iter++) {
		for (int k_iter = 0; k_iter < Ncol_op; k_iter++)
			res(p_iter, k_iter) = y_m[0][p_iter - p_base][k_iter];
	}

	#if DEBUG==1
		if(debug) {
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif

	if (!subtract_means)
		return;
	cout<<"cgb5"<<endl;	

	for (int p_iter = 0; p_iter < p; p_iter++) {
 		for (int k_iter = 0; k_iter < Ncol_op; k_iter++) {
			res(p_iter, k_iter) = res(p_iter, k_iter) - (g.get_col_mean(p_iter) * sum_op[k_iter]);
			if (var_normalize)
				res(p_iter, k_iter) = res(p_iter, k_iter)/(g.get_col_std(p_iter));		
 		}
 	}
	cout<<"cgb6"<<endl;	

}

//y*X
void multiply_y_pre(MatrixXdr &op, int Ncol_op, MatrixXdr &res, bool subtract_means) {
    if (fast_mode) {
        multiply_y_pre_fast(op, Ncol_op, res, subtract_means);
	} else {
		if (memory_efficient)
			multiply_y_pre_naive_mem(op, Ncol_op, res);
		else
			multiply_y_pre_naive(op, Ncol_op, res);
	}
}

pair<double, double> get_error_norm(MatrixXdr &c) {
	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p, k);
	MatrixXdr q_t(k, p);
	q_t = Q.transpose();
	MatrixXdr b(k, n);
	// Need this for subtracting the correct mean in case of missing data
	if (missing) {
		multiply_y_post(q_t, k, b, false);
		// Just calculating b from seen data
		MatrixXdr M_temp(k, 1);
		M_temp = q_t * means;
		for (int j=0; j<n; j++) {
			MatrixXdr M_to_remove(k,1);
			M_to_remove = MatrixXdr::Zero(k, 1);
			for (int i=0; i<g.not_O_j[j].size(); i++) {
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
	if (fast_mode)
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

    if (fast_mode) {
        double temp_k = b_k.cwiseProduct(b).sum();
        double temp_l = b_l.cwiseProduct(b).sum();
        double b_knorm = b_k.norm();
        double b_lnorm = b_l.norm();
        double norm_k = (b_knorm * b_knorm) - (2 * temp_k);
        double norm_l = (b_lnorm * b_lnorm) - (2 * temp_l);	
        return make_pair(norm_k, norm_l);
    } else {
        MatrixXdr e_l(p, n);
        MatrixXdr e_k(p, n);
        for (int p_iter = 0; p_iter < p; p_iter++) {
            for (int n_iter = 0; n_iter < n; n_iter++) {
                e_l(p_iter, n_iter) = g.get_geno(p_iter, n_iter, var_normalize) - b_l(p_iter, n_iter);
                e_k(p_iter, n_iter) = g.get_geno(p_iter, n_iter, var_normalize) - b_k(p_iter, n_iter);
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

	#if DEBUG==1
		if(debug){
			print_time ();
			cout << "Enter: run_EM_not_missing" << endl;
		}
	#endif

 	// c_temp : k X p matrix: (C^T C)^{-1} C^{T}
	MatrixXdr c_temp(k, p);
	MatrixXdr c_new(p, k);
	c_temp = ((c_orig.transpose() * c_orig).inverse()) * (c_orig.transpose());

	#if DEBUG==1
		if(debug){
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
	MatrixXdr x_fn(k, n);
	multiply_y_post(c_temp, k, x_fn, true);

	#if DEBUG == 1
		if (debug) {
			print_timenl();
		}
	#endif

	//x_temp: n X k matrix X^{T} (XX^{T})^{-1}
	MatrixXdr x_temp(n, k);
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
	
	MatrixXdr c_new(p, k);

	MatrixXdr mu(k, n);
	
	// E step
	MatrixXdr c_temp(k,k);
	c_temp = c_orig.transpose() * c_orig;

	MatrixXdr T(k,n);
	MatrixXdr c_fn;
	c_fn = c_orig.transpose();
	multiply_y_post(c_fn,k,T,false);

	MatrixXdr M_temp(k,1);
	M_temp = c_orig.transpose() *  means;
	
	for (int j = 0; j < n; j++) {
		MatrixXdr D(k,k);
		MatrixXdr M_to_remove(k,1);
		D = MatrixXdr::Zero(k,k);
		M_to_remove = MatrixXdr::Zero(k,1);
		for(int i=0;i<g.not_O_j[j].size();i++){
			int idx = g.not_O_j[j][i];
			D = D + (c_orig.row(idx).transpose() * c_orig.row(idx));
			M_to_remove = M_to_remove + (c_orig.row(idx).transpose()*g.get_col_mean(idx));
		}
		mu.col(j) = (c_temp-D).inverse() * ( T.col(j) - M_temp + M_to_remove);
	}

	#if DEBUG==1
		if(debug){
			ofstream x_file;
			x_file.open((string(command_line_opts.OUTPUT_PATH)+string("x_in_fn_vals.txt")).c_str());
			x_file<<std::setprecision(15)<<mu<<endl;
			x_file.close();
		}
	#endif


	// M step

	MatrixXdr mu_temp(k,k);
	mu_temp = mu * mu.transpose();
	MatrixXdr T1(p,k);
	MatrixXdr mu_fn;
	mu_fn = mu.transpose();
	multiply_y_pre(mu_fn,k,T1,false);
	MatrixXdr mu_sum(k,1);
	mu_sum = MatrixXdr::Zero(k,1);
	mu_sum = mu.rowwise().sum();

	for(int i=0;i<p;i++){
		MatrixXdr D(k,k);
		MatrixXdr mu_to_remove(k,1);
		D = MatrixXdr::Zero(k,k);
		mu_to_remove = MatrixXdr::Zero(k,1);
		for(int j=0;j<g.not_O_i[i].size();j++){
			int idx = g.not_O_i[i][j];
			D = D + (mu.col(idx) * mu.col(idx).transpose());
			mu_to_remove = mu_to_remove + (mu.col(idx));
		}
		c_new.row(i) = (((mu_temp-D).inverse()) * (T1.row(i).transpose() -  ( g.get_col_mean(i) * (mu_sum-mu_to_remove)))).transpose();
		double mean;
		mean = g.get_col_sum(i);
		mean = mean -  (c_orig.row(i)*(mu_sum-mu_to_remove))(0,0);
		mean = mean * 1.0 / (n-g.not_O_i[i].size());
		g.update_col_mean(i,mean);
	}

	// IMPORTANT: Update the value of means variable present locally, so that for next iteration, updated value of means is used.
	for (int i = 0; i < p; i++) {
		means(i,0) = g.get_col_mean(i);
		// Also updating std, just for consistency, though, it is not used presently.
		stds(i,0) = g.get_col_std(i);
	}

	return c_new;
}

MatrixXdr run_EM(MatrixXdr &c_orig) {

	if (missing) {
		return run_EM_missing(c_orig);
	}
	else {
		return run_EM_not_missing(c_orig);
	}
}

void print_vals() {

	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);

	// Need this for subtracting the correct mean in case of missing data
	if (missing) {
		multiply_y_post(q_t, k, b, false);
		// Just calculating b from seen data
		MatrixXdr M_temp(k,1);
		M_temp = q_t * means;
		for (int j=0; j<n; j++){
			MatrixXdr M_to_remove(k,1);
			M_to_remove = MatrixXdr::Zero(k,1);
			for (int i=0; i<g.not_O_j[j].size(); i++) {
				int idx = g.not_O_j[j][i];
				M_to_remove = M_to_remove + (Q.row(idx).transpose()*g.get_col_mean(idx));
			}
			b.col(j) -= (M_temp - M_to_remove);
		}
	}
	else{
		multiply_y_post(q_t, k, b, true);
	}

	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l; 
	u_l = b_svd.matrixU();
	MatrixXdr v_l;
	v_l = b_svd.matrixV();
	MatrixXdr u_k;
	MatrixXdr v_k,d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);

	ofstream evec_file;
	evec_file.open((string(command_line_opts.OUTPUT_PATH)+string("evecs.txt")).c_str());
	evec_file<< std::setprecision(15) << Q*u_k << endl;
	evec_file.close();
	ofstream eval_file;
	eval_file.open((string(command_line_opts.OUTPUT_PATH)+string("evals.txt")).c_str());
	for(int kk =0 ; kk < k_orig ; kk++)
		eval_file << std::setprecision(15)<< (b_svd.singularValues())(kk)*(b_svd.singularValues())(kk)/g.Nsnp<<endl;
	eval_file.close();

	eveP = v_l.leftCols(k_orig);
	ofstream proj_file;
	proj_file.open((string(command_line_opts.OUTPUT_PATH) + string("projections.txt")).c_str());
	proj_file << std::setprecision(15)<< v_k << endl;
	proj_file.close();

	if (debug) {
		ofstream c_file;
		c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals.txt")).c_str());
		c_file<<std::setprecision(15)<<c<<endl;
		c_file.close();

		ofstream means_file;
		means_file.open((string(command_line_opts.OUTPUT_PATH)+string("means.txt")).c_str());
		means_file<<std::setprecision(15)<<means<<endl;
		means_file.close();
		
		d_k = MatrixXdr::Zero(k_orig, k_orig);
		for(int kk =0 ; kk < k_orig ; kk++)
			d_k(kk,kk)  =(b_svd.singularValues())(kk);
		MatrixXdr x_k;
		x_k = d_k * (v_k.transpose());
		ofstream x_file;
		x_file.open((string(command_line_opts.OUTPUT_PATH) + string("xvals.txt")).c_str());
		x_file<<std::setprecision(15)<<x_k.transpose()<<endl;
		x_file.close();
	}
}

void EigenGWAS() {

	clock_t eg_begin = clock();
	cout << "---------------EigenGWAS scan-------------" << endl;
	cout << "SNP size: " << p << endl;
	cout << "Sample size: " << n << endl;
	cout << eveP.cols() << " eigen vectors" << endl;
	cout << eveP.rows() << " row numbers" << endl;

	using boost::math::students_t;
	students_t Tdist(n - 1);
	double pt2tail;

	MatrixXdr evePS(eveP.rows(), eveP.cols());
	//standardization
	for (int i = 0; i < eveP.cols(); i++) {
		double sum = 0;
		double sd = 0;
		double eSq = 0;

		for (int j = 0; j < eveP.rows(); j++) {
			sum += eveP(j, i);
			eSq += eveP(j, i) * eveP(j, i);
		}

		double m = sum/eveP.rows();
		sd = sqrt((eSq - m * m * eveP.rows())/(eveP.rows() - 1));
		for (int j = 0; j < eveP.rows(); j++) {
			evePS(j, i) = (eveP(j, i) - m)/sd;
		}
	}

//	cout<<" conventional X"<<endl;
	//calculate beta; possibly faster using mailman
//	MatrixXdr EgBetaC = geno_matrix * evePS / geno_matrix.cols();

	//using mailman
	MatrixXdr EgBeta(p, eveP.cols());
	multiply_y_pre(evePS, evePS.cols(), EgBeta, true);
	cout<<" conventional X"<<endl;

	EgBeta = EgBeta/n;
	for (int i = 0; i < EgBeta.cols(); i++) {
		cout << EgBeta(0, i) <<endl;
	}

	for (int i = 0; i < EgBeta.cols(); i++) {
		cout << "Scanning " << i+1 << " eigenvector..." <<endl;
		ofstream e_file;
		e_file.open((string(command_line_opts.OUTPUT_PATH) + string("eg."+std::to_string(i+1)+".txt")).c_str());
		e_file << "CHR\tSNP\tPOS\tBP\tA1\tA2\tBeta\tSE\tT-stat\tP" << endl;
		for (int j = 0; j <EgBeta.rows(); j++) {
			double egB = EgBeta(j, i);
			double seB = sqrt((1 - pow(EgBeta(j, i), 2))/(n - 1));
			double t_stat = egB/seB;
//			cout << "mk: " << j << " infor=" <<g.get_bim_info(j)<<" egB="<< egB<<" seB="<<seB<< " t="<<t_stat<<endl;
			double pt2tail = cdf(complement(Tdist, fabs(t_stat))) * 2;
			e_file<<g.get_bim_info(j)<<"\t"<<std::setprecision(8)<<egB<<"\t"<<seB<<"\t"<<t_stat<<"\t"<<pt2tail<<endl;
		}
		e_file.close();
		cout << "Finished " << i+1 << " eigenvector..." <<endl;
	}

	clock_t eg_end = clock();
	double eg_time = double(eg_end - eg_begin) / CLOCKS_PER_SEC;
	cout<<"RHE time "<< eg_time <<endl;
//		MatrixXdr EigenSe = 
}

void RandHE(int seed, int iter) {

	cout << "Randomized HE, iteration for " << iter << " times" << endl;

	clock_t he_begin = clock();

	double LB = 0;
	MatrixXdr Bz(geno_matrix.cols(), 1);
	srand(seed);
	std::default_random_engine generator(seed);
	std::normal_distribution<double> norm_dist(0, 1.0);

	for (int i = 0; i < iter; i++) {
		for (int j = 0; j < geno_matrix.cols(); j++) {
			Bz(j, 0) = norm_dist(generator);
		}
//		MatrixXdr T1 = geno_matrix * Bz; //(m x n) * (n x1) = mx1
//		MatrixXdr T2 = T1.transpose() * geno_matrix; // (1xm) * (mxn) = 1xn
//		MatrixXdr Lb = T2 * (T2.transpose());

		MatrixXdr T1 = geno_matrix * Bz; //(m x n) * (n x1) = mx1
		MatrixXdr T2 = geno_matrix.transpose() * T1; // (1xm) * (mxn) = 1xn
		MatrixXdr Lb = T2.transpose() * T2;
		LB += Lb(0, 0);
	}
	cout<< "LB " << LB <<endl;
	LB /= iter * geno_matrix.rows() * geno_matrix.rows();
	cout<< "LB2 " << LB <<endl;
	double me = (LB - geno_matrix.cols())/(geno_matrix.cols() * geno_matrix.cols());
	cout<< "Me " << 1/me <<endl;
	clock_t he_end = clock();

	double he_time = double(he_end - he_begin) / CLOCKS_PER_SEC;
	cout<<"RHE time "<< he_time <<endl;

}

void RandHE_fast(int seed, int iter) {

	cout << "Randomized HE (mailman), iteration for " << iter << " times" << endl;

	clock_t he_begin = clock();

	double LB = 0;
	MatrixXdr Bz(n, iter);
	srand(seed);
	std::default_random_engine generator(seed);
	std::normal_distribution<double> norm_dist(0, 1.0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < iter; j++) {
			Bz(i, j) = norm_dist(generator);
		}
	}

	//geno_matrix * Bz; //(p x n) * (n x iter) = p x iter
	MatrixXdr T1(p, iter);
	cout<<"Here T1=G^T * z"<<endl;
	multiply_y_pre(Bz, iter, T1, true);
	MatrixXdr T1_transpose(iter, p);
	T1_transpose = T1.transpose(); //iter X p
	cout << "Here T1.transpose" << endl;

	MatrixXdr T2(iter, n);
	cout << "Here T2=T1^T * G^T" << endl;
	cout << T1_transpose.rows() << " " << T1_transpose.cols() << " " << endl;
	multiply_y_post(T1_transpose, iter, T2, true);

	for (int i = 0; i < T2.rows(); i++) {
		for (int j = 0; j < T2.cols(); j++) {
			LB += T2(i, j) * T2(i, j);
		}
	}

	cout<< "LB " << LB <<endl;
	LB /= iter * p * p;
	cout<< "LB2 " << LB <<endl;
	double me = (LB - n)/(n * n);
	cout<< "Me " << 1/me <<endl;
	clock_t he_end = clock();

	double he_time = double(he_end - he_begin) / CLOCKS_PER_SEC;
	cout<<"RHE time "<< he_time <<endl;

}


void ENC() {

}

void CLD() {

}

int main(int argc, char const *argv[]) {

	auto start = std::chrono::system_clock::now();

	clock_t io_begin = clock();
    clock_gettime(CLOCK_REALTIME, &t0);

	pair<double,double> prev_error = make_pair(0.0, 0.0);
	double prevnll = 0.0;

	parse_args(argc, argv);

	//TODO: Memory Effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
    fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;
    MAX_ITER = command_line_opts.max_iterations;
	k_orig = command_line_opts.l;
	debug = command_line_opts.debugmode;
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = command_line_opts.var_normalize;
	accelerated_em = command_line_opts.accelerated_em;
	nthreads = command_line_opts.nthreads;
	given_seed = command_line_opts.given_seed;
	seed = given_seed ? std::abs(command_line_opts.seed) : (unsigned int) time(0);

	propc = command_line_opts.propc;

	scan = command_line_opts.scan;
	inbred = command_line_opts.inbred;

	rhe = command_line_opts.rhe;
	rhe_it = command_line_opts.rhe_it;

	enc = command_line_opts.enc;

	cld = command_line_opts.cld;

//read data--universal step
	g.set_poptype(inbred);
	if (text_version) {
		if (fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH, missing);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH, missing);
	} else {
		clock_t plink_read_begin = clock();
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH, missing, fast_mode);
		clock_t plink_read_end = clock();
		double plink_io_time = double(plink_read_end - plink_read_begin) / CLOCKS_PER_SEC;
		cout << "Reading plink data in " << plink_io_time << "s" <<endl;
	}

	//TODO: Implement these codes.
	if (missing && !fast_mode) {
		cout<<"Missing version works only with mailman i.e. fast mode\n EXITING..."<<endl;
		exit(-1);
	}
	if (fast_mode && memory_efficient) {
		cout<<"Memory effecient version for mailman EM not yet implemented"<<endl;
		cout<<"Ignoring Memory effecient Flag"<<endl;
	}
	if (missing && var_normalize) {
		cout<<"Missing version works only without variance normalization\n EXITING..."<<endl;
		exit(-1);
	}

	k = k_orig + command_line_opts.l;
	k = (int) ceil(k/10.0) * 10;
	command_line_opts.l = k - k_orig;
	p = g.Nsnp;
	n = g.Nindv;
	convergence_limit = command_line_opts.convergence_limit;
	bool toStop = false;
	if (convergence_limit != -1)
		toStop = true;

	if (command_line_opts.given_seed)
		srand(command_line_opts.seed);
	else
		srand((unsigned int) time(0));

	c.resize(p, k);
	x.resize(k, n);
	v.resize(p, k);
	means.resize(p, 1);
	stds.resize(p, 1);

	if (inbred) cout<<"Inbred mode is switched on."<<endl;
	if (!fast_mode && !memory_efficient) {
		cout<<"Genotype standardization..."<<endl;
		geno_matrix.resize(p, n);
		g.generate_eigen_geno(geno_matrix, var_normalize);
	}

	clock_t io_end = clock();


	blocksize = k;
	int hsegsize = g.segment_size_hori; 	// = log_3(n)
	int hsize = pow(3, hsegsize);
	int vsegsize = g.segment_size_ver; 		// = log_3(p)
	int vsize = pow(3, vsegsize);

	partialsums = new double* [nthreads];
	yint_m = new double* [nthreads];
	for (int t = 0; t < nthreads; t++) {
		partialsums[t] = new double [blocksize];
		yint_m[t] = new double [hsize*blocksize];
		memset(yint_m[t], 0, hsize*blocksize * sizeof(double));
	}

	sum_op = new double[blocksize];

	yint_e = new double*[nthreads];
	for (int t = 0; t < nthreads; t++) {
		yint_e[t] = new double [hsize*blocksize];
		memset(yint_e[t], 0, hsize * blocksize * sizeof(double));
	}

	y_e = new double**[nthreads];
	for (int t = 0; t < nthreads; t++) {
		y_e[t]  = new double*[g.Nindv];
		for (int i = 0; i < g.Nindv; i++) {
			y_e[t][i] = new double[blocksize];
			memset(y_e[t][i], 0, blocksize * sizeof(double));
		}
	}

	y_m = new double**[nthreads];
	for (int t = 0; t < nthreads; t++) {
		y_m[t] = new double*[hsegsize];
		for (int i = 0; i < hsegsize; i++)
			y_m[t][i] = new double[blocksize];
	}

	for (int i=0; i<p; i++) {
		means(i, 0) = g.get_col_mean(i);
		stds(i, 0) = g.get_col_std(i);
	}


	if (propc) {
		cout<<"propc"<<endl;
		//TODO: Initialization of c with gaussian distribution
		c = MatrixXdr::Random(p, k);
		if(given_seed) {
			srand(seed);
		} else {
			srand((unsigned int) time(0));
		}
		std::default_random_engine generator(seed);
		std::normal_distribution<double> norm_dist(0, 1.0);
		for (int i = 0; i < c.rows(); i++)
			for (int j = 0; j < c.cols(); j++)
				c(i, j) = norm_dist(generator);
		cout<<"propc1"<<endl;
		// Initial intermediate data structures
		// Operate in blocks to improve caching
		//

		ofstream c_file;
		if (debug) {
			c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals_orig.txt")).c_str());
			c_file<<std::setprecision(15)<<c<<endl;
			c_file.close();
			printf("Read Matrix\n");
		}

		cout<<"Running on Dataset of "<<g.Nsnp<<" SNPs and "<<g.Nindv<<" Individuals"<<endl;

		#if SSE_SUPPORT==1
			if(fast_mode)
				cout<<"Using Optimized SSE FastMultiply"<<endl;
		#endif

		clock_t it_begin = clock();
		for (int i=0; i<MAX_ITER; i++) {

			MatrixXdr c1, c2, cint, r, v;
			double a, nll;
			if (debug) {
				print_time ();
				cout << "*********** Begin epoch " << i << "***********" << endl;
			}
			if (accelerated_em != 0) {
				#if DEBUG==1
				if (debug) {
					print_time();
					cout << "Before EM" << endl;
				}
				#endif
				c1 = run_EM(c);
				c2 = run_EM(c1);
				#if DEBUG==1
				if (debug) {
					print_time();
					cout << "After EM but before acceleration" << endl;
				}
				#endif
				r = c1 - c;
				v = (c2-c1) - r;
				a = -1.0 * r.norm() / (v.norm()) ;
				if (accelerated_em == 1) {
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
				} else if (accelerated_em == 2) {
					cint = c - 2 * a * r + a * a * v;
					c = cint;
					// c = run_EM(cint);
				}
			} else {
				c = run_EM(c);
			}

			if (accelerated_em == 1 || check_accuracy || toStop) {
				pair<double, double> e = get_error_norm(c);
				prevnll = e.second;
				if (check_accuracy) 
					cout << "Iteration " << i+1 << "  " << std::setprecision(15) << e.first << "  " << e.second << endl;
				if (abs(e.first - prev_error.first) <= convergence_limit) {
					cout << "Breaking after " << i+1 << " iterations" << endl;
					break;
				}
				prev_error = e;
			}
			if (debug) {
				print_time();
				cout<< "*********** End epoch " << i << "***********" << endl;
			}
		}

		clock_t it_end = clock();

	    print_vals();

		clock_t total_end = clock();
		double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
		double avg_it_time = double(it_end - it_begin) / (MAX_ITER * 1.0 * CLOCKS_PER_SEC);
		double total_time = double(total_end - total_begin) / CLOCKS_PER_SEC;
		cout<<"IO Time: "<<io_time<<"\nAVG Iteration Time: "<<avg_it_time<<"\nTotal runtime: "<<total_time<<endl;

		std::chrono::duration<double> wctduration = std::chrono::system_clock::now() - start;
		cout<<"Wall clock time = "<<wctduration.count()<<endl;
	}
	if (propc && scan) {
		EigenGWAS();
	} else if (rhe) {
		RandHE_fast(seed, command_line_opts.rhe_it);
	} else if (enc) {
		ENC();
	} else if (cld) {
		CLD();
	}

	delete[] sum_op;
	for (int t = 0 ; t < nthreads ; t++){
		delete[] yint_e[t];
	}
	delete[] yint_e;

	for (int t = 0 ; t < nthreads ; t++){
		delete[] yint_m[t];
		delete[] partialsums[t];
	}
	delete[] yint_m;
	delete[] partialsums;

	for (int t = 0 ; t < nthreads ; t++){
		for (int i  = 0 ; i < hsegsize; i++)
			delete[] y_m [t][i];
		delete[] y_m[t];
	}
	delete[] y_m;

	for (int t = 0 ; t < nthreads ; t++){
		for (int i  = 0 ; i < g.Nindv; i++)
			delete[] y_e[t][i]; 
		delete[] y_e[t];
	}
	delete[] y_e;

	return 0;
}
