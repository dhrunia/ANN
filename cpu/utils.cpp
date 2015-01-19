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

using namespace std;
using namespace arma;

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

#endif




