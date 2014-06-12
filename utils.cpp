//#include<algorithm>
//#include<cstdlib>
//#include<ctime>

#include<armadillo>
#include<vector>
#include<string>
#include<sstream>

using namespace std;
using namespace arma;

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

//void gen_batchdata(mat &input,mat &output,mat &batchInput,mat &batchOutput,int batchSize)
//{
//	//picks random frames from inputdata and their corresponding output
//	//and generates a matrices for batch input and output
//
//	int frameNo;
//	int nFrames = input.n_cols;
////	srand(time(NULL));
//	for(int i=0;i<batchSize;i++)
//	{
//		frameNo = rand()%nFrames;
////		cout<<frameNo<<endl;
//		batchInput.col(i) = input.col(frameNo);
//		batchOutput.col(i) = output.col(frameNo);
//	}
////	cout<<endl;
//}

//template<typename T>
//void get_shuffled_framenums(T &frameNos,int batchSize,int nPatterns)
//{
//	for(int i=0;i<batchSize;i++)
//	{
//		frameNos[i] = rand()%nPatterns;
//	}
//}




