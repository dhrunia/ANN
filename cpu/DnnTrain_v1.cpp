
#include<iostream>
#include<armadillo>
#include "dnn.h"
#include "utils.cpp"
#include "Dnn_v1.cpp"
#include "TrainingAlgorithm.cpp"
#include "params.cpp"
#include<sstream>
#include<fstream>
#include<vector>
#include<ctime>
#include<iomanip>


using namespace arma;
using namespace std;

int main(int argc,char** argv)
{
	const char *inpFname,*outFname,*paramsFname,*weightsFname;
	int nFrames,temp,nEpochs;
	int validCount = 0; // keeps a count of number of times validation error increased from previous validation error contiguously
	float momentum;
	double timeElapsed;
	clock_t startTime,endTime;
	uvec frameNos;
    Mat<elem_type> *inputData = new Mat<elem_type>();
	Mat<elem_type> *outputData = new Mat<elem_type>();
	Mat<elem_type> *temp_mat;
	startTime = clock();
//	cout<<"number of arguments passed: "<<argc<<endl;
	if(!(argc<7))
	{
		paramsFname = argv[1];
		inpFname = argv[2];
		outFname = argv[3];
		weightsFname = argv[4];
//		nEpochs=lexical_cast<int>(argv[5]);
//		momentum = lexical_cast<float>(argv[6]);
		str_to_type(argv[5],nEpochs);
		str_to_type(argv[6],momentum);
		cout<<"inp_fname: "<<inpFname<<endl;
		cout<<"out_fname: "<<outFname<<endl;
		cout<<"no.of epochs:"<<nEpochs<<endl;
	}
	else{
		cerr<<"6 arguments are expected as input to the program, only "<<argc<<" are given"<<endl;
		cerr<<"Run it as follows:"<<endl;
		cerr<<"./DnnTrain_v1 <parameters file> <input file> <output file> <weights file> <#Epochs> <momentum>"<<endl<<endl;
		cerr<<"argument 1: Neural network parameters file name in the following format:"<<endl;
		cerr<<"-------------------------"<<endl;
		cerr<<" <units in input layer> <units in first layer> <units in second hidden layer> ... <units in output layer>"<<endl;
		cerr<<" <output function type of first hidden layer> ... <output function type of output layer>"<<endl;
		cerr<<" <batchsize> [<batches per epoch> optional]"<<endl;
		cerr<<" <learning rate>"<<endl;
		cerr<<"-------------------------"<<endl<<endl;
		cerr<<"argument 2: Input data file name"<<endl;
		cerr<<"argument 3: Output data file name"<<endl;
		cerr<<"argument 4: Weights file name"<<endl;
		cerr<<"argument 5: Number of Epochs"<<endl;
		cerr<<"argument 6: Momentum"<<endl;
		exit(0);
	}
	Params nnParams(paramsFname);
	nFrames =  ReadData(inpFname,(*inputData));
//	cout<<"Total Frames in input:"<<nFrames<<endl;
	cout<<"Dim. of input data: "<<inputData->n_rows<<"x"<<inputData->n_cols<<endl;
	temp = ReadData(outFname,(*outputData));
	cout<<"Dim. of output data: "<<outputData->n_rows<<"x"<<outputData->n_cols<<endl;
	if(temp != nFrames)
	{
		cout<<"No.of frames in input and output does not match"<<endl;
		exit(0);
	}
	if(nnParams.preProcessData)
	{
		preprocess_data(*inputData);
//		preprocess_data(*outputData);
	}
	//shuffle the data
//	frameNos = randi<uvec>(nFrames,distr_param(0,nFrames-1));
    frameNos = linspace<uvec>(0,nFrames-1,nFrames);
    frameNos = shuffle(frameNos);
	temp_mat = new Mat<elem_type>();
	*temp_mat = inputData->cols(frameNos);
	inputData->reset();
	delete inputData;
    inputData = temp_mat;
    temp_mat = new Mat<elem_type>();
    *temp_mat = outputData->cols(frameNos);
    outputData->reset();
    delete outputData;
    outputData = temp_mat;
//	cout<<InputData.submat(0,0,2,14);
//	cout<<OutputData;
//	cout<<"Dim. of output data :"<<outputData->n_rows<<"x"<<outputData->n_cols<<endl;
//	cout<<outputData.submat(0,0,4,2);
    DNN *nn;
    if(nnParams.loadWeights == "random")
    	nn = new DNN(nnParams);
    else
    	nn = new DNN(nnParams,nnParams.loadWeights.c_str(),"arma_ascii");
//	cout<<"DNN object created"<<endl;

	if(nnParams.trainAlgoType == "BGD" || nnParams.trainAlgoType == "bgd")
	{
		BGD *trainAlgo = new BGD();
		trainAlgo->train(*nn,*inputData,*outputData,0.3,nEpochs,weightsFname);
	}
	else if(nnParams.trainAlgoType == "CGD" || nnParams.trainAlgoType == "cgd")
	{
		CGD *trainAlgo = new CGD();
//		cout<<"CGD object created"<<endl;
		trainAlgo->initialise(inputData,outputData,nn);
//		cout<<"CGD initialisation completed"<<endl;
		trainAlgo->train(nEpochs,weightsFname);
	}
    endTime = clock();
	timeElapsed = (endTime-startTime)/((double)CLOCKS_PER_SEC*60);
	cout << fixed << showpoint << setprecision(2);
	cout<<"Time elapsed: "<< timeElapsed <<" minutes" <<endl;
}


