#ifndef MAILBOX_H
#define MAILBOX_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include "mailman.h"

#if SSE_SUPPORT == 1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

extern Goptions goptions;
extern genotype g;
extern double **partialsums;
extern double *sum_op;

extern double **yint_e;
extern double ***y_e;

extern double **yint_m;
extern double ***y_m;


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

/*
	int seg_iter;
	for(seg_iter = 0; seg_iter < g.Nsegments_hori-1; seg_iter++){
		mailman::fastmultiply_pre (g.segment_size_hori, g.Nindv, Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter], op, yint_e, partialsums[0], y_e);
	}
*/

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

//y*X
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

#endif
