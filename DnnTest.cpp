#include<iostream>
#include"Dnn_v1.cpp"
#include<armadillo>

using namespace std;
using namespace arma;

int main(int argc,char **argv)
{
	Mat<elem_type> input,output;
	const char *inpFname,*outFname,*wtsFname,*biasFname,*paramsFname,*taskType;
	if(argc<7)
	{
		cerr<<"5 arguments expected only "<<argc<<" given"<<endl;
		cerr<<"Run the program as follows:"<<endl;
		cerr<<"./DnnTest <NN params file> <input file> <output file> <weights file> <bias file> <taskType[regression/classification]>"<<endl;
		exit(0);
	}
	else
	{
		paramsFname = argv[1];
		inpFname = argv[2];
		outFname = argv[3];
		wtsFname = argv[4];
		biasFname = argv[5];
		taskType = argv[6];
	}
	ReadData(inpFname,input);
//	cout<<"Input read"<<endl;
	DNN *nn = new DNN(paramsFname,wtsFname,biasFname,"arma_ascii");
	cout<<"Predicting output..."<<endl;
	if(!strcmp(taskType,"classification"))
		nn->gen_output(input,outFname,true);
	else
		nn->gen_output(input,outFname);
	return 0;
}
