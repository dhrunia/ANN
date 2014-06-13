#include<iostream>
#include<armadillo>
#include"Dnn_v2.cpp"
#include<sstream>
#include<fstream>
#include<vector>
#include<ctime>
#include<iomanip>

using namespace arma;
using namespace std;

int main(int argc,char** argv)
{
	const char *inpFname,*outFname,*paramsFname,*errorFname,*weightsFname;
	int nEpochs;
	float momentum;
	double timeElapsed;
	clock_t startTime,endTime;
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
//		cout<<"inp_fname: "<<inpFname<<endl;
//		cout<<"out_fname: "<<outFname<<endl;
//		cout<<"no.of epochs:"<<nEpochs<<endl;
	}
	else{
		cout<<"6 arguments are expected as input to the program, only "<<argc<<" are given"<<endl;
		cout<<"Run it as follows:"<<endl;
		cout<<"./DnnTrain_v1 <parameters file> <input file> <output file> <weights file> <#Epochs> <momentum>"<<endl;
		cout<<"argument 1: Neural network parameters file name in the following format:"<<endl;
		cout<<" <units in input layer> <units in first layer> <units in second hidden layer> ... <units in output layer>"<<endl;
		cout<<" <output function type of first hidden layer> ... <output function type of output layer>"<<endl;
		cout<<" <batchsize> [<batches per epoch> optional]"<<endl;
		cout<<" <learning rate>"<<endl;
		cout<<"argument 2: Input data file name"<<endl;
		cout<<"argument 3: Output data file name"<<endl;
		cout<<"argument 4: Weights file name(weights are saved to this file)"<<endl;
		cout<<"argument 5: Number of Epochs"<<endl;
		cout<<"argument 6: Momentum"<<endl;
		exit(0);
	}
	errorFname = "errors.txt";
	mat *inputData = new mat();
	mat *outputData = new mat();
	mat *batchInput;
	mat *batchOutput;
	int nFrames,temp,batchSize,batchesPerEpoch;
	uvec frameNos,batchFrameNos;
	nFrames =  ReadData(inpFname,(*inputData));
//	cout<<"Total Frames in input:"<<nFrames<<endl;
//	cout<<"Dim. of input data: "<<inputData->n_rows<<"x"<<inputData->n_cols<<endl;
	temp = ReadData(outFname,(*outputData));
	if(temp != nFrames)
	{
		cout<<"No.of frames in input and output does not match"<<endl;
		exit(0);
	}
//	cout<<InputData.submat(0,0,2,14);
//	cout<<OutputData;
//	cout<<"Dim. of output data :"<<outputData->n_rows<<"x"<<outputData->n_cols<<endl;
//	cout<<outputData.submat(0,0,4,2);
	DNN *nn = new DNN(paramsFname);
	batchSize = nn->batchSize;
	if(nn->batchesPerEpoch)
		batchesPerEpoch = nn->batchesPerEpoch;
	else
		batchesPerEpoch = nFrames/batchSize;
//	cout<<"batchsize and bathcesperpoch: "<<batchSize<<" and "<<batchesPerEpoch<<endl;
	batchInput = new mat(nn->inputDimension,batchSize,fill::zeros);
	batchOutput = new mat(nn->outputDimension,batchSize,fill::zeros);
	double batchError;
	int batchSP;
	ofstream errfh(errorFname);
	if(!errfh.is_open())
		cout<<"unable to open file "<<errorFname<<endl;
	arma_rng::set_seed_random();
	cout<<"Training started";
	for(int epochNo =0;epochNo<nEpochs;epochNo++)
	{
		frameNos = randi<uvec>(nFrames-(nFrames%batchSize),distr_param(0,nFrames-1));
//		cout<<"Epoch: "<<epochNo<<""<<endl;
		for(int batchNo=0;batchNo<batchesPerEpoch;batchNo++)
		{
//			inds = randi<uvec>(nn->batchSize,distr_param(0,nFrames-1));
			batchSP = batchNo*batchSize;
			batchFrameNos = frameNos.subvec(batchSP,batchSP+batchSize-1);
			*batchInput = inputData->cols(batchFrameNos);
			*batchOutput = outputData->cols(batchFrameNos);
//			gen_batchdata((*inputData),(*outputData),(*batchInput),(*batchOutput),nn->batchSize);
			nn->compute_output(*batchInput);
//			cout<<"outputs computed"<<endl;
			batchError += nn->compute_outputerror(*batchOutput) / batchesPerEpoch;
//			cout<<"error computed"<<endl;
			nn->compute_localgradients();
//			cout<<"local gradients computed"<<endl;
//			nn->compute_deltas(*batchInput);
			nn->compute_deltaswithmomandlayerlr(*batchInput,momentum);
//			cout<<"deltas computed"<<endl;
			nn->increment_weights();
//			cout<<"weights incremented"<<endl;
		}
//		cout<<"Error: "<<batchError<<endl;
		errfh<<batchError<<endl;
		batchError=0;
		cout<<".";
		cout.flush();
	}
	cout<<endl;
	cout<<"Training completed"<<endl;
	nn->save_weights(weightsFname);
	endTime = clock();
	timeElapsed = (endTime-startTime)/((double)CLOCKS_PER_SEC*60);
	cout << fixed << showpoint << setprecision(2);
	cout<<"Time elapsed: "<< timeElapsed <<" minutes" <<endl;
	errfh.close();
}




