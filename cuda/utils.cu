//#include<algorithm>
//#include<cstdlib>
//#include<ctime>
#ifndef UTILS_H_
#define UTILS_H_

#include<armadillo>
#include<vector>
#include<string>
#include<sstream>
#include<iomanip>
#include<cublas_v2.h>
//#include<cuda.h>
#include<cuda_runtime.h>
using namespace std;
using namespace arma;

typedef double elem_type;

typedef struct{
    int width;
    int height;
    elem_type *elements;
}matrix;

template <typename T>
void split(string s,vector<T> &vec)
{ // splits the given string at whitespaces and stores them in the vector provided

	stringstream ss(s);
	T val;
	while(ss>>val)
	{
		vec.push_back(val);
	}
}

int no_of_lines(const char *fname)
{ // Return the number of lines in the file given
	ifstream fh(fname);
	string line;
	int nLines;
	nLines=0;
	if(fh.is_open())
		while (getline(fh, line))
			++nLines;
	fh.close();
	return nLines;
}

template <typename T>
void print_vec(vector<T> &vec)
{
	for(int i=0;i<vec.size();i++)
	{
		cout<<vec[i]<<" ";
	}
	cout<<endl;
}

template<typename T>
string list_tostring(T listValues,int nValues)
{ //convert a array of values into a string
	stringstream ss;
	for(int i=0;i<nValues;i++)
	{
		ss<<listValues[i]<<" ";
	}
	ss<<endl;
	return ss.str();
}

template<typename T>
void str_to_vec(string s,vector<T> &vec)
{ //convert a array of values into a string
	stringstream ss(s);
	T val;
	while(ss>>val)
		vec.push_back(val);
}

template<typename T>
void str_to_type(string str,T &val)
{
	stringstream ss(str);
	ss>>val;
}


//void read_params(const char *paramsFname)
//{
//	ifstream fh(paramsFname);
//	string line;
//	vector<int> temp;
//	for(int lineNo=1;getline(fh,line);lineNo++)
//	{
//		switch(lineNo)
//		{
//		case 1:
//			str_to_vec(line,unitsInLayer);
//			break;
//		case 2:
//			str_to_vec(line,outFnType);
//			break;
//		case 3:
//			str_to_vec(line,temp);
//			batchSize = temp[0];
//			if(temp.size()==2)
//				batchesPerEpoch = temp[1];
//			else
//				batchesPerEpoch = 0;
//			break;
//		case 4:
//			str_to_type(line,eta);
//			break;
//		}
//	}
//	nLayers = outFnType.size();
////	cout<<"no.of layers: "<<nLayers<<endl;
//}

int ReadData(const char *fname,mat &Data)
{	// Reads the data from the file given as input into a matrix and returns the matrix

	ifstream fh(fname);
	int nPatterns,nFeatures;
	string line;
	vector<double> pattern;
//	cout<<"Reading data from file: "<<fname<<endl;
	if(fh.is_open())
	{
		nPatterns = no_of_lines(fname);
		getline(fh,line);
		str_to_vec(line,pattern);
		nFeatures = pattern.size();
//		cout<<"no of patterns: "<<nPatterns<<endl;
//		cout<<"no of features per pattern: "<<nFeatures<<endl;
//		Data.set_size(nPatterns,nFeatures);
		Data.zeros(nFeatures,nPatterns);
		vector<double> linef; // a vector of floats representing a row of input patterns
		str_to_vec(line,linef);
		for(int j=0;j<nFeatures;j++)
			Data(j,0) = linef[j];
		for(int i=1;getline(fh,line);i++)
		{
			linef.clear();
//			cout<<line<<endl;
			split(line,linef);
			for(int j=0;j<nFeatures;j++)
				Data(j,i) = linef[j];
		}
		fh.close();
	}
	return nPatterns;
}


int myReadData(const char *fname,matrix &Data)
{	// Reads the data from the file given as input into a matrix and returns the matrix

	ifstream fh(fname);
	int nPatterns,nFeatures,colIdx,rowIdx;
	string line;
	vector<elem_type> pattern;
	//	cout<<"Reading data from file: "<<fname<<endl;
	if(fh.is_open())
	{
		nPatterns = no_of_lines(fname);
		getline(fh,line);
		str_to_vec(line,pattern);
		nFeatures = pattern.size();
		Data.width = nPatterns;
		Data.height = nFeatures;
		//		cout<<"no of patterns: "<<nPatterns<<endl;
		//		cout<<"no of features per pattern: "<<nFeatures<<endl;
		//		Data.set_size(nPatterns,nFeatures);
		Data.elements = new elem_type[nFeatures*nPatterns];
		vector<elem_type> linef; // a vector of floats representing a row of input patterns
		str_to_vec(line,linef);
		rowIdx = 0;
		colIdx = 0;
		for(int j=0;j<nFeatures;j++)
		{
             		Data.elements[rowIdx*nPatterns + colIdx] = linef[j];
			rowIdx++;
		}
		colIdx++;
		for(int i=1;getline(fh,line);i++)
		{
			rowIdx = 0;
			linef.clear();
			//			cout<<line<<endl;
			split(line,linef);
			for(int j=0;j<nFeatures;j++){
				Data.elements[rowIdx*nPatterns + colIdx] = linef[j];
				rowIdx++;
			}
			colIdx++;
		}
		fh.close();
	}
	return nPatterns;
}

void print_matrix(matrix &m)
{
    for(int i = 0;i<m.height;i++)
    {
        for(int j=0;j<m.width;j++)
            cout<<fixed<<showpoint<<setprecision(4)<<m.elements[i*m.width + j]<<" ";
        cout<<endl;
    }
}

void insert_onesrow(Mat<elem_type> &A)
{
    Row<elem_type> onesRow;
    onesRow.ones(A.n_cols);
    A.insert_rows(A.n_rows,onesRow);
}

void preprocess_data(Mat<elem_type> &A)
{	//mean subtraction and variance normalization of data
	Col<elem_type> meanA,varA;
//	A.print("A:");
	meanA = mean(A,1);
//	meanA.print("meanA:");
	varA = var(A,0,1);
//	varA.print("varA:");
	A.each_col() -= meanA;
//	A.print("meanSubA:");
	A.each_col() /= varA;
//	A.print("varNormA:");
}

template<typename T>
int matmul(const T *A,cublasOperation_t transA,const T *B,cublasOperation_t transB,
           int HA,int WA,int WB,double alpha,double beta,T *hC)
{ //performs matrix-matrix mutliplication using cublas gemm() function.
  //dim(A): HA x WA , dim(B): WA x WB
    
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    T *dA,*dB,*dC;
//    double *alpha,*beta;
//    alpha = (double*)malloc(sizeof(double));
//    beta = (double*)malloc(sizeof(double));
//    *alpha = 1;
//    *beta = 0;
//    hC = (double *)malloc(HA*WB*sizeof(double));
//    Mat<T> *C = new Mat<T>(HA,WB,fill::zeros);
//    hC = C->memptr();
    
    //Allocate memory on device(GPU) for A,B,C
    cudaStat = cudaMalloc((void**)&dA,HA*WA*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc((void**)&dB,WA*WB*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc((void**)&dC,HA*WB*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
//  create a cuda handle
    stat = cublasCreate(&handle); 
    if (stat != CUBLAS_STATUS_SUCCESS) {
     printf ("CUBLAS initialization failed\n"); 
     return EXIT_FAILURE;
    }
    
    stat = cublasSetMatrix(HA,WA,sizeof(T),A,HA,dA,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("set matrix failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }
    
    stat = cublasSetMatrix(WA,WB,sizeof(T),B,WA,dB,WA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("set matrix failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }

//  call the cublas matrix-matrix mulitplication kernel    
    stat = cublasDgemm(handle,transA,transB,HA,WB,WA,&alpha,dA,HA,dB,WA,&beta,dC,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("matrix-matrix(cublasDgemm) mulitiplication failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }

// copy the prodcut matrix back to host memory
    stat = cublasGetMatrix(HA,WB,sizeof(T),dC,HA,hC,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("fetching data from device memory failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);   
    return EXIT_SUCCESS;
}

template<typename T>
int matadd(const T *A,const T *B,int HA,int WA,double alpha,double beta,T *C)
{
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    T *dA,*dB,*dC;
//    double *alpha,*beta;
//    alpha = (double*)malloc(sizeof(double));
//    beta = (double*)malloc(sizeof(double));
//    *alpha = 1;
//    *beta = 1;
    //allocate memory for the matrices A,B,C on device
    cudaStat = cudaMalloc((void**)&dA,HA*WA*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc((void**)&dB,HA*WA*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc((void**)&dC,HA*WA*sizeof(T));
    if (cudaStat != cudaSuccess) { 
    printf ("device memory allocation failed");
    return EXIT_FAILURE;
    }
    
    //create a cuda handle
    stat = cublasCreate(&handle); 
    if (stat != CUBLAS_STATUS_SUCCESS) {
     printf ("CUBLAS initialization failed\n"); 
     return EXIT_FAILURE;
    }
    
    //copy the matrices A and B from host memory to device memory    
    stat = cublasSetMatrix(HA,WA,sizeof(T),A,HA,dA,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("set matrix failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }
    
    stat = cublasSetMatrix(HA,WA,sizeof(T),B,HA,dB,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("set matrix failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }
    
    //call the cublas matrix-matrix addition function
    stat = cublasDgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,HA,WA,&alpha,dA,HA,&beta,dB,HA,dC,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("matrix-matrix(cublasDgeam) addition failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }

// copy the prodcut matrix back to host memory
    stat = cublasGetMatrix(HA,WA,sizeof(T),dC,HA,C,HA);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
    printf ("fetching data from device memory failed");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);   
    return EXIT_SUCCESS;

}
#endif


