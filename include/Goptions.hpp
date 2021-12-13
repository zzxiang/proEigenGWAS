#ifndef GOPTIONS_HPP_
#define GOPTIONS_HPP_

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <string>
#include <iostream>
#include <map>
#include <stdexcept>
#include <fstream>
using namespace std;

const std::string version("1.0");

class OptionsExitsProgram : public std::exception
{};

class Goptions {
public:
   // The constructor sets up all the various options that will be parsed
   Goptions() {
      SetOptions();
   }

   // Parse options runs through the heirarchy doing all the parsing
   void ParseOptions(int argc, char const *argv[]) {
      ParseCommandLine(argc, argv);
      NotDefined();
      CheckForHelp();
      CheckForVersion();
   }

   // Below is the interface to access the data, once ParseOptions has been run
   std::string Path() {
      return results["path"].as<std::string>();
   }

   std::string Verbosity() {
      return results["verbosity"].as<std::string>();
   }

   std::vector<std::string> IncludePath() {
      if (results.count("include-path")) {
         return results["include-path"].as<std::vector<std::string> >();
      }
      return std::vector<std::string>();
   }

   std::string MasterFile() {
      if (results.count("master-file")) {
         return results["master-file"].as<std::string>();
      }
      return "";
   }

   std::vector<std::string> Files() {
      if (results.count("file")) {
         return results["file"].as<std::vector<std::string> >();
      }
      return std::vector<std::string>();
   }

   bool GUI() {
      if (results["run-gui"].as<bool>()) {
         return true;
      }
      return false;
   }

//generic
   string GetGenericGenoFile() {
      return results["bfile"].as<string>();
   }

   string GetGenericOutFile() {
      return results["out"].as<string>();
   }

   string GetGenericPhenoFile() {
      return results["pheno"].as<string>();
   }

   vector<int> GetGenericPhenoNum() {
      return results["pheno-num"].as<vector<int>>();
   }

   bool IsGenericMemoryEfficient() {
      return results["mem"].as<bool>();
   }

   bool IsGenericTextMode() {
      return results["txt"].as<bool>();
   }

   bool IsGenericSubstractMean() {
      return !results["miss"].as<bool>();
//      return results["substract-mean"].as<bool>();
   }

   bool IsGenericNoMailman() {
      return results["no-mailman"].as<bool>();
   }

   bool IsGenericFastMode() {
      return !results["no-mailman"].as<bool>();
   }

   bool IsGenericMissing() {
      return results["miss"].as<bool>();
   }

   bool IsGenericVarNorm() {
      return results["var-norm"].as<bool>();
   }

   bool IsGenericDebug() {
      return results["debug"].as<bool>();
   }

   int GetGenericEigenvecNumber() {
      return results["evec"].as<int>();
   }

   int GetGenericMailmanBlockSize() {
      int k = (int) ceil(GetGenericEigenvecNumber()/10.0) * 10;

      if (CheckRandHEMasterOption()) {
         k = GetGenericIteration();
      } else if (CheckEncMasterOption()) {
         k = GetEncK();
      }
      return k;
   }

   int GetGenericThreads() {
      return results["threads"].as<int>();
   }

   int GetGenericSeed() {
      return results["seed"].as<int>();
   }

   int GetGenericIteration() {
      return results["iter"].as<int>();
   }

//propc
   bool CheckPropcMasterOption() {
      return results["propc"].as<bool>();
   }

   bool IsPropcAccuracy() {
      return results["accu"].as<bool>();
   }

   int GetPropcMaxIteration() {
      return results["max-it"].as<int>();
   }

   int GetPropcAcceleratedEM() {
      return results["accel-em"].as<int>();
   }

   double GetPropcConvergenceLimit() {
      return results["conv-limit"].as<double>();
   }

//eigengwas
   bool CheckEigenGWASMasterOption() {
      return results["eigengwas"].as<bool>();
   }

   int GetEigenGWASAdjust() {
      return results["eg-adjust"].as<int>();
   }

//rand-he
   bool CheckRandHEMasterOption() {
      return results["rand-he"].as<bool>();
   }

//enc
   int GetEncK() {
      return results["enc-k"].as<int>();
   }

   bool CheckEncMasterOption() {
      return results["enc"].as<bool>();
   }

//cld
   bool CheckCLDMasterOption() {
      return results["cld"].as<bool>();
   }

private:

   void NotDefined() {
      if (IsGenericMissing() && !IsGenericFastMode()) {
//   	if (missing && !fast_mode) {
		   cout << "Missing version works only with mailman i.e. fast mode\n EXITING..." << endl;
		   exit(-1);
	   }
      if (IsGenericFastMode() && IsGenericMemoryEfficient()) {
//	   if (fast_mode && memory_efficient) {
		   cout << "Memory effecient version for mailman EM not yet implemented" << endl;
		   cout << "Ignoring Memory effecient Flag" << endl;
	   }
      if (IsGenericMissing() && IsGenericVarNorm()) {
//	   if (missing && var_normalize) {
		   cout << "Missing version works only without variance normalization\n EXITING..." << endl;
		   exit(-1);
	   }
   }

   void SetOptions() {
      SetGenericOptions();
      SetPropcOptions();
      SetEigenGWASOptions();
      SetEncOptions();
      SetRandHEOptions();
      SetCLDOptions();
      SetCommonOptions();
      SetEnvMapping();
   }



   void SetGenericOptions() {
      genericOpts.add_options()
         ("help", "produce help message.")
         ("version,v", "print version string.")

         ("debug", po::bool_switch()->default_value(false), "debug mode.")
         ("txt", po::bool_switch()->default_value(false), "text pedigree files (default false).")
         ("mem", po::bool_switch()->default_value(false), "the flag states whether to use a memory effecient version for the EM algorithm or not. The memory efficient version is a little slow than the not efficient version (default: false)")
         ("var-norm", po::bool_switch()->default_value(true), "normalization for mailman.")
         ("no-mailman", po::bool_switch()->default_value(false), "no mailman (default, false).")
         ("miss", po::bool_switch()->default_value(false), "no missing (default, false, when true the missing genotypes will be imputed.")
 //        ("substract-mean", po::bool_switch()->default_value(true), "substract-mean for mailman (default, true).")

         ("bfile", po::value<string>(&generic_bGeno_file), "root of plink binary pedigree files.")
         ("out", po::value<string>()->default_value("out"), "root for output files.")
         ("pheno", po::value<string>(&generic_pheno_file), "pheno file.")
         ("pheno-num", po::value<vector<int> >(&generic_pheno_index)->multitoken(), "pheno index")
         ("threads", po::value<int>(&generic_thread)->default_value(1), "thread number.")
         ("seed", po::value<int>(&generic_seed)->default_value(2021), "seed for generating random numbers.")
         ("evec", po::value<int>()->default_value(2), "eigenvector to estimate.")
         ("iter", po::value<int>()->default_value(5), "iteration for randomization.")

         ("config,c", po::value<string>(&config_file), "config files to parse (always parses default.cfg).")
         ;
   }

   void SetPropcOptions() {
      propcOpts.add_options()
         ("propc", po::bool_switch(&propc_switch)->default_value(false), "master option for propc.")
         ("accu", po::bool_switch()->default_value(false), "output the likelihood computation as a function of iterations.")
         ("max-it", po::value<int>(&propc_max_it)->default_value(2), "maximun iteration for propc.")
         ("conv-limit", po::value<double>()->default_value(0.05), "The value of the threshold telling the algorithm that it has converged (the value of -1, meaning no auto termination condition.")
         ("accel-em", po::value<int>()->default_value(0), "The flag stating whether to use accelerated EM or not (default: 0, and can be 1 or 2).")
         ;
   }

   void SetEigenGWASOptions() {
      eigengwasOpts.add_options()
         ("eigengwas", po::bool_switch(&eigengwas_switch)->default_value(false), "master option for eigengwas.")
         ("eg-adjust", po::value<int>(&eigengwas_adjust)->default_value(0), "adjustment.")
         ;
   }

   void SetEncOptions() {
      encOpts.add_options()
         ("enc", po::bool_switch(&enc_switch)->default_value(false), "master option for enc.")
         ("enc-k", po::value<int>(&enc_k)->default_value(10), "iteration for generating tags.")
         ;
   }

   void SetRandHEOptions() {
      randheOpts.add_options()
         ("rand-he", po::bool_switch(&randhe_switch)->default_value(false), "master option for randomized Haseman-Elston regression.")
         ;
   }

   void SetCLDOptions() {
      cldOpts.add_options()
         ("cld", po::bool_switch()->default_value(false), "master option for chr-ld.")
      ;
   }

   void SetCommonOptions() {
      common_options.add_options()
         ("path", po::value<std::string>()->default_value(""),
            "the execution path to use (imports from environment if not specified)")
         ("verbosity", po::value<std::string>()->default_value("INFO"),
            "set verbosity: DEBUG, INFO, WARN, ERROR, FATAL")
         ("include-path,I", po::value<std::vector<std::string> >()->composing(),
            "paths to search for include files")
         ("run-gui", po::bool_switch(), "start the GUI")
         ;
   }

   void SetEnvMapping() {
      env_to_option["PATH"] = "path";
      env_to_option["EXAMPLE_VERBOSE"] = "verbosity";
   }

   void ParseCommandLine(int argc, char const *argv[]) {
      po::options_description cmd_opts;
      cmd_opts.add(genericOpts)
      .add(propcOpts)
      .add(eigengwasOpts)
      .add(encOpts)
      .add(randheOpts)
      .add(cldOpts)
      .add(common_options);

      store(po::command_line_parser(argc, argv).
         options(cmd_opts).run(), results);
      notify(results);
   }

   void CheckForHelp() {
      if (results.count("help")) {
         PrintHelp();
      }
   }

   void PrintHelp() {
      std::cout << "Program Options Example" << std::endl;
      std::cout << "Usage: example [OPTION]... MASTER-FILE [FILE]...\n";
      std::cout << "  or   example [OPTION] --run-gui\n";
      po::options_description help_opts;
      help_opts
      .add(genericOpts)
      .add(propcOpts)
      .add(eigengwasOpts)
      .add(encOpts)
      .add(randheOpts)
      .add(cldOpts)
      .add(common_options);
      std::cout << help_opts << std::endl;
      throw OptionsExitsProgram();
   }

   void CheckForVersion() {
      if (results.count("version")) {
         PrintVersion();
      }
   }

   void PrintVersion() {
      std::cout << "Program Options Example " << version << std::endl;
      throw OptionsExitsProgram();
   }

   std::string EnvironmentMapper(std::string env_var) {
      // ensure the env_var is all caps
      std::transform(env_var.begin(), env_var.end(), env_var.begin(), ::toupper);

      auto entry = env_to_option.find(env_var);
      if (entry != env_to_option.end()) {
         return entry->second;
      }
      return "";
   }

   po::options_description genericOpts;
   po::options_description propcOpts;
   po::options_description eigengwasOpts;
   po::options_description encOpts;
   po::options_description randheOpts;
   po::options_description cldOpts;
 
   std::map<std::string, std::string> env_to_option;
   po::options_description common_options;
   
   po::variables_map results;

   int opt;
   string config_file;
   string generic_bGeno_file;
   string generic_pheno_file;
   vector<int> generic_pheno_index;
   int generic_thread;
   int generic_seed;

   bool propc_switch;
   int propc_vec_num;
   int propc_max_it;

   bool eigengwas_switch;
   int eigengwas_vec;
   int eigengwas_adjust;

   bool enc_switch;
   int enc_k;

   bool randhe_switch;
   bool cld_swith;
   int cld_k;
};

#endif
