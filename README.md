# proEigenGWAS for biobank-scale genetic data

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Citation

Please cite our manuscript if you use our software.

```
Chen G-B, Lee, SH, Zhu, Z-X, Benyamin, B, Robinson, MR (2016) EigenGWAS: finding loci under selection through genome-wide association studies of eigenvectors in structured populations. Heredity 117:51-61
Agrawal A, Chiu AM, Le M, Halperin E, Sankararaman S (2020) Scalable probabilistic PCA for large-scale genetic variation data. PLOS Genetics 16(5): e1008773. https://doi.org/10.1371/journal.pgen.1008773
```

### Prerequisites

The following packages are required on a linux machine to compile and use the software package.

```
g++
cmake
make
```

### Installing

Installing proEG is fairly simple. Just issue the following commands on a linux machine

```
git clone https://github.com/gc5k/proEigenGWAS.git
cd proEigenGWAS
mkdir build
cd build
```
By default, the release version is built, if you wish to build the debug version, build with cmake flag `-D DEBUG=1` as shown below.

proEigenGWAS supports, [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) instructions.

If your architecure is Intel and supports SSE instructions, build with the cmake flag `-D SSE_SUPPORT=1` for an faster improved version as shown below.


```
cmake -D SSE_SUPPORT=1 -D DEBUG=1 ..
make
```

Else just issue these commands below:

```
cmake ..
make
```


## Documentation for proEigenGWAS

After compiling the executable propca is present in the build directory.
Running the propca is fairly simple and can be done in two different ways

* ``./proEG -p <parameter_file>``
* ``./proEG <various_command_line arguments>``

### Parameters

The values in the brackets are the command line flags for running the code without the parameter file.

```
* genotype (-g) : The path of the genotype file or plink bed file prefix
* l (-l) : The extra calculation to be performed so that k_effective  = k + l (default: num_evec)
* max_iterations (-m) : The maximum number of iterations to run the EM for (default: num_evec + 2)
* debug (-v) : Enabling debug mode to output various debug informations (default: false). Need to build with DEBUG=1 as described above for this flag to work.
* accuracy (-a) : Output the likelihood computation as a function of iterations (default: false)
* convergence_limit (-cl) : The value of the threshold telling the algorithm that it has converged (default: -1, meaning no auto termination condition )
* output_path (-o) : The output prefix along with the path where the results will be stored
* accelerated_em (-aem) : The flag stating whether to use accelerated EM or not (default: 0).
* var_normalize (-vn) : The flag stating whether to perform varinance normalization or not (default: false).
* fast_mode (-nfm) : The flag whether to use a fast mode for the EM algorithm(default: true). Note: Setting the -nfm (NOT fast_mode) at command line will use a slow version of EM.
* missing (-miss) : This flag states whether there is any missing data present in the genotype matrix or not. 
* text_version (-txt) : This flag makes the input genotype file to be in the text format as described below. If not used, plink format will be used. (default: false)
* memory_efficient (-mem) : The flag states whether to use a memory effecient version for the EM algorithm or not. The memory efficient version is a little slow than the not efficient version (default: false)
* nthreads (-nt): Number of threads to use (default: 1)
* seed (-seed): Seed to use (default: system time)
* scan (-scan): Scan EigenGWAS
* inbred (-inbred): inbred
```

An example parameter file is provided in the examples directory.

You can run the code using the command:

```
../build/proEG -p par.txt
``` 

The equivalent command to issue for running the same code from the examples directory is:

```
../build/proEG -g example -l 2 -m 20 -a -cl 0.001 -o example_ -aem 1 -vn -scan
../build/proEG -g Arab295Line -l 2 -m 5 -a -cl 0.001 -o Arab295Line -aem 1 -vn -scan
../build/proEG -g Arab295Line -l 2 -m 5 -a -cl 0.001 -o Arab295Line -aem 1 -vn -enc -enc-k 100 -seed 200
```

proEG wil generate three files containing the eigenvectors/principal components, projections, and eigenvalues.

### Genotype File

The input can be in the plink binary format, as descibed at [Plink BED](https://www.cog-genomics.org/plink/1.9/input#bed)

Make sure to set the text_version to false in the parameter file, or don't use the -txt command line flag, when running. 

## TODO List:

1. Add Missing version which runs on Naive EM.
2. Memory Effecient version of Mailman EM
3. Add Variance normalized version for Missing EM.
4. Improvize the memory requirements for naive EM and not memory effecient code.
5. Initialize C with gaussian distribution.
6. Memory Effecient SSE Multiply

## Built With

* [Eigen](http://eigen.tuxfamily.org/) - The Linear algebra library for C++
* [Boost](http://boost.org) - Boost library for C++.
In addition, both eigen and boost should be installed first, and the cmake will automatically find the both libraries. Of note, some of Boost's library should be built first.

## Authors

* **Aman Agrawal** - [http://www.cse.iitd.ernet.in/~cs1150210/](http://www.cse.iitd.ernet.in/~cs1150210/)
* **Alec Chiu** - [alecmchiu.github.io](alecmchiu.github.io)

See also the list of [contributors](https://github.com/aman71197/ProPCA/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
