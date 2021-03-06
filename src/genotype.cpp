#include <bits/stdc++.h>
#include "genotype.h"
#include "storage.h"
#include "time.h"
#include "Goptions.hpp"

using namespace std;

void genotype::init_means(bool is_missing) {
	allow_missing = is_missing;
	columnmeans.resize(Nsnp);
	for (int i = 0; i < Nsnp; i++) {
		double sum = columnsum[i] * 1.0;
		if (is_missing)
			columnmeans[i] = sum * 1.0 / (Nindv - (int) not_O_i[i].size());
		else
			columnmeans[i] = sum * 1.0 / Nindv;
	}
}

float genotype::get_observed_pj(const std::string &line) {
	int observed_sum = 0;
	int observed_ct = 0;
	for (int j = 0; j < line.size(); j++) {
		int val = int(line[j]-'0');
		if (val == 0 || val == 1 || val == 2) {
			observed_sum += val;
			observed_ct++;
		}
	}

	float p_j = observed_sum * 0.5/observed_ct;
	return p_j;
}

float genotype::get_observed_pj(const unsigned char* line) {
	int y[4];
	int observed_sum = 0;
	int observed_ct = 0;

	for (int k = 0; k < ncol; k++) {
		unsigned char c = line[k];
		y[0] = (c) & mask;
		y[1] = (c>>2) & mask;
		y[2] = (c>>4) & mask;
		y[3] = (c>>6) & mask;
		int j0 = k * unitsperword;
		int lmax = 4;
		if (k == ncol - 1) {
			lmax = Nindv % 4;
			lmax = (lmax==0) ? 4 : lmax;
		}
		for (int l = 0; l < lmax; l++) {
			int j = j0 + l;
//			int ver_seg_no = j/segment_size_ver;
			// Extract  PLINK coded genotype and convert into 0/1/2
			// PLINK coding:
			// 00->0
			// 01->missing
			// 10->1
			// 11->2
			int val = y[l];
			val--;
			if (val != 0) {
				val = (val < 0) ? 0 : val;
				observed_sum += val;
				observed_ct++;
			}
		}
	}
	return observed_sum * 0.5/observed_ct;
}

int simulate_geno_from_random(float p_j) {
	float rval = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float dist_pj[3] = {(1 - p_j) * (1 - p_j), 2 * p_j * (1 - p_j), p_j * p_j};
	if (rval < dist_pj[0])
		return 0;
	else if(rval >= dist_pj[0] && rval < (dist_pj[0] + dist_pj[1]))
		return 1;
	else
		return 2;
}

void genotype::read_geno(string geno_file, bool text_version, bool fast_mode, bool missing, bool mem_efficient, bool var_norm) {
	allow_missing = missing;
	memory_efficient = mem_efficient;
	if (text_version) {
		if (fast_mode)
			// g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH, missing);
			read_txt_mailman(geno_file, missing);
		else
			// g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH, missing);
			read_txt_naive(geno_file, missing);
		if (!fast_mode && !mem_efficient) {
			geno_matrix.resize(Nindv, Nsnp);
			generate_eigen_geno();
		}
	} else {
		clock_t plink_read_begin = clock();
		// g.read_plink(command_line_opts.GENOTYPE_FILE_PATH, missing, fast_mode);
		read_plink(geno_file, missing, fast_mode);
		clock_t plink_read_end = clock();
		double plink_io_time = double(plink_read_end - plink_read_begin) / CLOCKS_PER_SEC;
		cout << "Reading plink data in " << plink_io_time << "s" <<endl;
	}
}

// Functions to read text files
void genotype::read_txt_naive(std::string filename, bool allow_missing) {
	
	ifstream ifs (filename.c_str(), ios::in);                                       
	
	std::string line;
	std::getline(ifs, line);
    std::istringstream iss(line);
    if (!(iss >> Nsnp >> Nindv)) { 
		cout<<"ERROR: Header with number of SNPs and individuals not present"<<endl; 
		exit(-1);
	}

	if(allow_missing){
		not_O_i.resize(Nsnp);
		not_O_j.resize(Nindv);	
	}

	int i=0;

	vector <bool> m;
	vector <bool> l;
	while (std::getline(ifs, line)) {
		float p_j = get_observed_pj(line);
		int sum = 0;
		int SqSum = 0;
		for (int j = 0; j < line.size(); j++) {
			int val = int(line[j] - '0');
			if (val == 9 && !allow_missing) {
				val = simulate_geno_from_random(p_j);
			}

			if (val == 0) {
				l.push_back(false);
				m.push_back(false);
			} else if (val == 1) {
				sum += 1;
				SqSum += 1;
				l.push_back(true);
				m.push_back(false);
			} else if (val == 2) {
				sum += 2;
				SqSum += 4;
				l.push_back(false);
				m.push_back(true);
			} else if (val == 9 && allow_missing) {
				not_O_i[i].push_back(j);
				not_O_j[j].push_back(i);
				l.push_back(false);
				m.push_back(false);
			} else {
				cout << "Invalid entry in Genotype Matrix" << endl;
				exit(-1);
			}
		}
		i++;
		columnsum.push_back(sum);
		columnSqSum.push_back(SqSum);
		msb.push_back(m);
		lsb.push_back(l);
		m.clear();
		l.clear();
	}
	init_means(allow_missing);
}

void genotype::read_txt_mailman(std::string filename, bool allow_missing) {
   	ifstream ifs(filename.c_str(), ios::in);

	// Calculating the sizes and other stuff for genotype matrix
	std::string line;
	std::getline(ifs, line);
    std::istringstream iss(line);
    if (!(iss >> Nsnp >> Nindv)) {
		cout << "ERROR: Header with number of SNPs and individuals not present" << endl; 
		exit(-1);
	}

//	segment_size_ver = ceil(log(Nsnp) / log(genosize));//not useful, cgb
//	Nsegments_ver = ceil(Nindv*1.0 / (segment_size_ver*1.0));//not useful, cgb

	segment_size_hori = ceil(log(Nindv) / log(genosize));
	Nsegments_hori = ceil(Nsnp*1.0 / (segment_size_hori*1.0));
	p.resize(Nsegments_hori, std::vector<int>(Nindv));

	if (allow_missing) {
		not_O_i.resize(Nsnp);
		not_O_j.resize(Nindv);
	}

	int i = 0;
	while (std::getline(ifs, line)) {
		float p_j = get_observed_pj(line);
		int horiz_seg_no = i / segment_size_hori;
		int sum = 0;
		int SqSum = 0;
		int missingCnt = 0;
		for (int j = 0; j < line.size(); j++) {
			int val = int(line[j] - '0');

			if (val == 9) {//mising genotype is 9
				if (allow_missing) {
					not_O_i[i].push_back(j);
					not_O_j[j].push_back(i);
					val = 0;
				} else {
					val = simulate_geno_from_random(p_j);
				}
				missingCnt++;
			}

			if (val == 0 || val == 1 || val == 2) {
				sum += val;
				SqSum += val * val;
				p[horiz_seg_no][j] = (3 * p[horiz_seg_no][j]) + val;
			} else {
				cout << "ERROR: Invalid character in genotype file" << endl;
				exit(-1);
			}
		}
		i++;
		columnsum.push_back(sum);
		columnSqSum.push_back(SqSum);
	}
	init_means(allow_missing);
}

// Functions to read binary files
template<typename T>
static std::istream & binary_read(std::istream& stream, T& value) {
	return stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <class T>
inline void printvector(vector<T> &t, string delim = " ", bool newline = false) {
		for (int i = 0; i < t.size(); i++)
				cout << t[i] << delim;
        if (newline)
            cout << endl;
}

template <class T>
inline void printvectornl(vector<T> &t, string delim = " ") {
    printvector (t, delim, true);
}

void genotype::read_bim(std::string filename) {

	ifstream inp(filename);
	if (!inp.is_open()){
		cerr << "Error reading file "<< filename <<endl;
		exit(1);
	}
	string line;
	int j = 0 ;
	int linenum = 0 ;
	while(std::getline (inp, line)){
		char c = line[0];
		if (c=='#')
			continue;
		bimInfo.push_back(line);
		linenum ++;
	}
	inp.close();
}

string genotype::get_bim_info(int snpIdx) {
	if (snpIdx > bimInfo.size()) {
		cerr << "Error getting snp information:" << snpIdx << "out of range (" << bimInfo.size() <<")"<<endl;
	}
	return bimInfo[snpIdx];
}

int genotype::countlines(string filename) {
	ifstream inp(filename.c_str());
	if (!inp.is_open()){
		cerr << "Error reading file "<< filename <<endl;
		exit(1);
	}
	string line;
	int j = 0 ;
	int linenum = 0 ;
	while(std::getline (inp, line)){
		linenum ++;
		char c = line[0];
		if (c=='#')
			continue;
		istringstream ss (line);
		if (line.empty())
			continue;
		j++;
	}
	inp.close();
	return j;
}

//void genotype::set_poptype(bool isInbred) {
//	this->inbred = isInbred;
//}

void genotype::set_metadata() { 
	wordsize = sizeof(char) * 8;
	unitsize = 2;
	genosize = 3;
	unitsperword = wordsize/unitsize;
	mask = 0;
	for (int i = 0 ; i < unitsize; i++)
		mask = mask |(0x1<<i);
    nrow = Nsnp;
    ncol = ceil((Nindv * 1.0) / unitsperword);
}

void genotype::read_bed_mailman(string filename, bool allow_missing) {
	cout << "Decomposing plink-bed into Mailman X=U*P format" << endl;
   	ifstream ifs (filename.c_str(), ios::in|ios::binary);
   	char magic[3];
	set_metadata();

    gtype = new unsigned char[ncol];

   	binary_read(ifs, magic);

//	segment_size_hori = ceil(log(Nindv) / log(genosize));
	segment_size_hori = floor(log(Nindv) / log(genosize)) - 2;
	Nsegments_hori = ceil((Nsnp * 1.0) / (segment_size_hori * 1.0));
	p.resize(Nsegments_hori, std::vector<int>(Nindv));

	if (allow_missing) {
		not_O_i.resize(Nsnp);
		not_O_j.resize(Nindv);
	}

	columnsum.resize(Nsnp);
	columnSqSum.resize(Nsnp);

	// Note that the coding of 0 and 2 can get flipped relative to plink because plink uses allele frequency (minor)
	// allele to code a SNP as 0 or 1.
	// This flipping does not matter for results.
//	vector<int> v(Nindv);
	int y[4];
	for (int i = 0; i < Nsnp; i++) {
		int sum = 0;
		int SqSum = 0;
		int horiz_seg_no = i / segment_size_hori;
	   	ifs.read(reinterpret_cast<char*>(gtype), ncol*sizeof(unsigned char)); 
		float p_j = get_observed_pj(gtype);
    	for (int k = 0; k < ncol; k++) {
        	unsigned char c = gtype[k];
			// Extract PLINK genotypes
        	y[0] = (c) & mask;
        	y[1] = (c>>2) & mask;
        	y[2] = (c>>4) & mask;
        	y[3] = (c>>6) & mask;
			int j0 = k * unitsperword;
			// Handle number of individuals not being a multiple of 4
			int lmax = 4;
			if (k == ncol - 1) {
				lmax = Nindv % 4;
				lmax = (lmax==0) ? 4 : lmax;
			}
			for (int l = 0; l < lmax; l++) {
				int j = j0 + l;
//				int ver_seg_no = j/segment_size_ver; //ver_seg_no is not useful, cgb
				// Extract PLINK coded genotype and convert into 0/1/2
				// PLINK coding:
				// 00->0
				// 01->missing
				// 10->1
				// 11->2
				int val = y[l];
				if (val == 1 && !allow_missing) {
					val = simulate_geno_from_random(p_j);
					val++;
					val = (val==1) ? 0 : val;
				}
				if (val == 1 && allow_missing) {
					not_O_i[i].push_back(j);
					not_O_j[j].push_back(i);
				}
				val--;
				val = (val < 0 ) ? 0 : val;
				sum += val;
				SqSum += val * val;
				p[horiz_seg_no][j] = 3 * p[horiz_seg_no][j] + val;
			}
    	}
		columnsum[i] = sum;
		columnSqSum[i] = SqSum;
		sum = 0;
	}
	init_means(allow_missing);

	delete[] gtype;
}

void genotype::read_bed_naive(string filename, bool allow_missing) {

	ifstream ifs (filename.c_str(), ios::in|ios::binary);                                       
   	char magic[3];
	set_metadata();
    gtype =  new unsigned char[ncol];

   	binary_read(ifs,magic);

	msb.resize(Nsnp, std::vector<bool>(Nindv));
	lsb.resize(Nsnp, std::vector<bool>(Nindv));

	if (allow_missing) {
		not_O_i.resize(Nsnp);
		not_O_j.resize(Nindv);	
	}

	columnsum.resize(Nsnp);  
	columnSqSum.resize(Nsnp);

	// Note that the coding of 0 and 2 can get flipped relative to plink because plink uses allele frequency (minor)
	// allele to code a SNP as 0 or 1.
	// This flipping does not matter for results.
	vector<int> v (Nindv);
	int y[4];
	for (int i = 0 ; i < Nsnp; i++){
		int sum = 0;
		int SqSum = 0;
	   	ifs.read (reinterpret_cast<char*>(gtype), ncol*sizeof(unsigned char));  
		float p_j = get_observed_pj(gtype); 		    
    	for (int k = 0 ;k < ncol ; k++) {
        	unsigned char c = gtype [k];
			// Extract PLINK genotypes
        	y[0] = (c)&mask;
        	y[1] = (c>>2)&mask;
        	y[2] = (c>>4)&mask;
        	y[3] = (c>>6)&mask;
			int j0 = k * unitsperword;
			// Handle number of individuals not being a multiple of 4
			int lmax = 4;
			if (k == ncol - 1) {
				lmax = Nindv % 4;
				lmax = (lmax == 0) ? 4 : lmax;
			}	
			// Note  : Plink uses different values for coding genotypes
			// Note  : Does not work for missing values
			// To handle missing data it is recommended to write a separate function. This is easy to do.
			// This will avoid the performance hit of checking for and handling missing values
			for (int l = 0; l < lmax; l++) {
				int j = j0 + l;
				// Extract  PLINK coded genotype and convert into 0/1/2
				// PLINK coding: 
				// 00->0
				// 01->missing
				// 10->1
				// 11->2
				int val = y[l];
				if (val==1 && !allow_missing) {
					val = simulate_geno_from_random(p_j);
					val++;
					val = (val==1) ? 0 : val;
				}
				if (val==1 && allow_missing) {
					not_O_i[i].push_back(j);
					not_O_j[j].push_back(i);
				}
				val--;
				val = (val < 0 ) ? 0 : val;
				sum += val;
				SqSum += val * val;
				if (val == 0) {
					lsb[i][j] = false;
					msb[i][j] = false;
				} else if (val == 1) {
					lsb[i][j] = true;
					msb[i][j] = false;
				} else if (val == 2) {
					lsb[i][j] = false;
					msb[i][j] = true;
				} else {
					cout << "Invalid entry in Genotype Matrix" << endl;
					exit(-1);
				}
			}
    	}
		columnsum[i] = sum;
		columnSqSum[i] = SqSum;
	}
	init_means(allow_missing);
	delete[] gtype;
}


void genotype::read_bed(string filename, bool allow_missing, bool mailman_mode) {
	if (mailman_mode) {
		cout<< "Read plink bed data in mailman mode..." <<endl;
		read_bed_mailman(filename, allow_missing);
	} else {
		cout<< "Read plink bed data in naive (no mailman) mode..." <<endl;
		read_bed_naive(filename, allow_missing);
	}
}

void genotype::read_plink(std::string filenameprefix, bool allow_missing, bool mailman_mode) { 
	
	std::stringstream f1;
	f1 << filenameprefix << ".bim";
	Nsnp = countlines(f1.str());
	read_bim(f1.str());
	cout<<"Read " << Nsnp << " SNPs from '" << filenameprefix << ".bim'" <<endl;

	std::stringstream f2;
	f2 << filenameprefix << ".fam";
	Nindv = countlines(f2.str());
	cout<<"Read " << Nindv << " Individuals from '" << filenameprefix << ".fam'" <<endl;

	std::stringstream f3;
	f3 << filenameprefix << ".bed";
	read_bed(f3.str(), allow_missing, mailman_mode);
}

// Accessor Functions
double genotype::get_geno(int snpindex, int indvindex, bool var_normalize = false) {
	double m = msb[snpindex][indvindex];
	double l = lsb[snpindex][indvindex];
	double geno = (m * 2.0 + l) - get_col_mean(snpindex);
	if (var_normalize)
		return geno/get_col_std(snpindex);
	else
		return geno;
}

double genotype::get_col_mean(int snpindex) {
	double temp = columnmeans[snpindex];
	return temp;
}

double genotype::get_col_sum(int snpindex) {
	double temp = columnsum[snpindex];
	return temp;
}

double genotype::get_col_std(int snpindex) {
	double p_i = get_col_mean(snpindex) / 2;
	if (p_i == 0 || p_i == 1)
		return 1.0;
	double cnt = allow_missing ? (1.0 * (Nindv - (int) not_O_i[snpindex].size())) : (1.0 * Nindv);
	double std = sqrt((columnSqSum[snpindex] - columnmeans[snpindex] * columnmeans[snpindex] * cnt) / (cnt - 1));
//	double std0 = inbred ? sqrt(4 * p_i * (1 - p_i)) : sqrt(2 * p_i * (1 - p_i));
//	cout << std << " " << std0 << endl;
	return std;
}

void genotype::generate_eigen_geno() {
//void genotype::generate_eigen_geno(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &geno_matrix, bool var_normalize) {
	for (int i = 0; i < Nsnp; i++) {
		for (int j = 0; j < Nindv; j++) {
			double m = msb[i][j];
			double l = lsb[i][j];
			double geno = (m * 2.0 + l) - get_col_mean(i);
			if (var_normalize)
				geno_matrix(i, j) = geno / get_col_std(i);
			else
				geno_matrix(i, j) = geno;
		}
	}
}


void genotype::generate_eigen_geno(MatrixXdr &geno_matrix, bool var_normalize) {
//void genotype::generate_eigen_geno(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &geno_matrix, bool var_normalize) {
	for (int i = 0; i < Nsnp; i++) {
		for (int j = 0; j < Nindv; j++) {
			double m = msb[i][j];
			double l = lsb[i][j];
			double geno = (m * 2.0 + l) - get_col_mean(i);
			if (var_normalize)
				geno_matrix(i, j) = geno / get_col_std(i);
			else
				geno_matrix(i, j) = geno;
		}
	}
}

// Modifier Functions
void genotype::update_col_mean(int snpindex,double value) {
	columnmeans[snpindex] = value;
}

/* Redundant Function
void genotype::read_genotype_eff (std::string filename,bool allow_missing){
	FILE* fp;
	fp= fopen(filename.c_str(),"r");
	int j=0;
	int i=0;
	char ch;
	// Calculating the sizes and other stuff for genotype matrix
	int rd = fscanf(fp,"%d %d\n",&Nsnp,&Nindv);
	segment_size_hori = ceil(log(Nindv)/log(3));
	segment_size_ver = ceil(log(Nsnp)/log(3));
	Nsegments_hori = ceil(Nsnp*1.0/(segment_size_hori*1.0));
	Nsegments_ver = ceil(Nindv*1.0/(segment_size_ver*1.0));
	Nbits_hori = ceil(log2(pow(3,segment_size_hori)));
	Nbits_ver = ceil(log2(pow(3,segment_size_ver)));
	Nelements_hori = floor( (Nindv * Nbits_hori *1.0) / 32) + 1;
	Nelements_ver = floor( (Nsnp * Nbits_ver*1.0) / 32) + 1;
	cout<<Nbits_hori<<"  "<<Nbits_ver<<"  "<<Nelements_hori<<"  "<<Nelements_ver<<endl;
	p_eff.resize(Nsegments_hori,std::vector<unsigned>(Nelements_hori));
	q_eff.resize(Nsegments_ver,std::vector<unsigned>(Nelements_ver));
	int sum=0;
	if(allow_missing){
		not_O_i.resize(Nsnp);
		not_O_j.resize(Nindv);	
	}

    do{
		int rd = fscanf(fp,"%c",&ch);
		if(ch=='\n'){
			i++;
			columnsum.push_back(sum);
			sum=0;
			j=0;
		}
		else{
			int val = int(ch-'0');
			int horiz_seg_no = i/segment_size_hori ;
			int ver_seg_no = j/segment_size_ver ;
			if(val==0){
				int temp = 3* extract_from_arr(j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(temp,j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(3*extract_from_arr(i,Nbits_ver,q_eff[ver_seg_no]),i,Nbits_ver,q_eff[ver_seg_no]);
			}
			else if(val==1){
				sum+=1;
				int temp = 3* extract_from_arr(j,Nbits_hori,p_eff[horiz_seg_no]) + 1;
				add_to_arr(temp,j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(3*extract_from_arr(i,Nbits_ver,q_eff[ver_seg_no]) + 1,i,Nbits_ver,q_eff[ver_seg_no]);
			}
			else if(val==2){
				sum+=2;
				int temp = 3* extract_from_arr(j,Nbits_hori,p_eff[horiz_seg_no]) + 2;
				add_to_arr(temp,j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(3*extract_from_arr(i,Nbits_ver,q_eff[ver_seg_no]) + 2,i,Nbits_ver,q_eff[ver_seg_no]);
			}
			else if(val==9 && allow_missing){
				int temp = 3* extract_from_arr(j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(temp,j,Nbits_hori,p_eff[horiz_seg_no]);
				add_to_arr(3*extract_from_arr(i,Nbits_ver,q_eff[ver_seg_no]),i,Nbits_ver,q_eff[ver_seg_no]);
				not_O_i[i].push_back(j);
				not_O_j[j].push_back(i);				
			}
			else{
				cout<<"Invalid entry in Genotype Matrix"<<endl;
				cout<<"If there is Missing data, run with -miss flag"<<endl;
				exit(-1);
			}
			j++;
		}
	}while(!feof(fp));
	i--;
	init_means(allow_missing);	
}



void genotype::read_bim (string filename){
	ifstream inp(filename.c_str());
	if (!inp.is_open()){
		cerr << "Error reading file "<< filename <<endl;
		exit(1);
	}
	string line;
	int j = 0 ;
	int linenum = 0 ;
	while(std::getline (inp, line)){
		linenum ++;
		char c = line[0];
		if (c=='#')
			continue;
		istringstream ss (line);
		if (line.empty())
			continue;
		j++;
	}
	Nsnp = j;
	inp.close();
}

void genotype::read_fam (string filename){
	ifstream inp(filename.c_str());
	if (!inp.is_open()){
		cerr << "Error reading file "<< filename <<endl;
		exit(1);
	}
	string line;
	int j = 0 ;
	int linenum = 0 ;
	while(std::getline (inp, line)){
		linenum ++;
		char c = line[0];
		if (c=='#')
			continue;
		istringstream ss (line);
		if (line.empty())
			continue;
		j++;
	}
	Nindv = j;
	inp.close();
}

*/