#ifndef EIGENGWAS_HPP_
#define EIGENGWAS_HPP_

#include "time.h"

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <math.h>
#include <boost/math/distributions/students_t.hpp>
#include "genotype.h"
#include "Goptions.hpp"
#include "global.h"

using namespace std;

extern Goptions goptions;
extern genotype g;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

class EigenGWAS {
    public:
    EigenGWAS() {

    }

    void Scan(MatrixXdr IndEigenVec) {
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
				if (goptions.IsGenericDebug()) {
				    cout << "MK: " << j << " infor=" << g.get_bim_info(j) << " egB=" << EgBeta(j, i) << " seB=" << seB << " t=" << t_stat << endl;
				}
			    double pt2tail = cdf(complement(Tdist, fabs(t_stat))) * 2;
			    e_file << g.get_bim_info(j) << "\t" << std::setprecision(8) << EgBeta(j, i) << "\t" << seB << "\t" << t_stat << "\t" << pt2tail <<endl;
		    }
    		e_file.close();
	    }

    	clock_t eg_end = clock();
	    double eg_time = double(eg_end - eg_begin) / CLOCKS_PER_SEC;
	    cout << "EigenGWAS total time " << eg_time << endl;
    }
};

#endif


/*
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
*/
