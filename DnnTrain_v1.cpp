#include<iostream>
#include<armadillo>
#include"Dnn_v1.cpp"
#include<sstream>
#include<fstream>
#include<vector>
#include<ctime>
#include<iomanip>
#include<limits>

#define VALIDATION_TOLERANCE 3 //validation error tolerance count. If validation error contiguously increases
                               //for more than defined here training is stopped.

using namespace arma;
using namespace std;

int main(int argc,char** argv)
{
	const char *inpFname,*outFname,*paramsFname,*errorFname,*weightsFname,*biasFname,*valErrFname;
	int nFrames,temp,batchSize,batchesPerEpoch,nFramesTrain,nFramesValid,batchSP,nEpochs;
	int validCount = 0; // keeps a count of number of times validation error increased from previous validation error contiguously
	float momentum;
	double timeElapsed,batchError,prevValidError,validError;
	clock_t startTime,endTime;
	uvec frameNos,batchFrameNos;
    Mat<elem_type> *inputData = new Mat<elem_type>();
	Mat<elem_type> *outputData = new Mat<elem_type>();
	Mat<elem_type> *batchInput;
	Mat<elem_type> *batchOutput;
	Mat<elem_type> *validationSetInput,*validationSetOutput;
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
//		cout<<"inp_fname: "<<inpFname<<endl;
//		cout<<"out_fname: "<<outFname<<endl;
//		cout<<"no.of epochs:"<<nEpochs<<endl;
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
	errorFname = "errors.txt";
	valErrFname = "valid_error.txt";
	nFrames =  ReadData(inpFname,(*inputData));
//	cout<<"Total Frames in input:"<<nFrames<<endl;
//	cout<<"Dim. of input data: "<<inputData->n_rows<<"x"<<inputData->n_cols<<endl;
	temp = ReadData(outFname,(*outputData));
	if(temp != nFrames)
	{
		cout<<"No.of frames in input and output does not match"<<endl;
		exit(0);
	}
	//shuffle the data
	frameNos = randi<uvec>(nFrames,distr_param(0,nFrames-1));
	temp_mat = new Mat<elem_type>();
	*temp_mat = inputData->cols(frameNos);
	delete inputData;
    inputData = temp_mat;
    temp_mat = new Mat<elem_type>();
    *temp_mat = outputData->cols(frameNos);
    delete outputData;
    outputData = temp_mat;
//	cout<<InputData.submat(0,0,2,14);
//	cout<<OutputData;
//	cout<<"Dim. of output data :"<<outputData->n_rows<<"x"<<outputData->n_cols<<endl;
//	cout<<outputData.submat(0,0,4,2);
	DNN *nn = new DNN(paramsFname);
//	DNN *nn = new DNN(paramsFname,weightsFname,biasFname);
	batchSize = nn->batchSize;
//	cout<<"batchsize and bathcesperpoch: "<<batchSize<<" and "<<batchesPerEpoch<<endl;
	batchInput = new Mat<elem_type>(nn->inputDimension,batchSize,fill::zeros);
	batchOutput = new Mat<elem_type>(nn->outputDimension,batchSize,fill::zeros);
	ofstream errfh(errorFname);
	ofstream valErrfh(valErrFname);
	if(!errfh.is_open())
		cout<<"unable to open file "<<errorFname<<endl;
    nFramesTrain = 0.8*nFrames;
    nFramesValid = nFrames - nFramesTrain;
//    cout<<"nFramesTrain: "<<nFramesTrain<<endl;
//    cout<<"nFramesValid: "<<nFramesValid<<endl;
    if(nn->batchesPerEpoch)
		batchesPerEpoch = nn->batchesPerEpoch;
	else
		batchesPerEpoch = nFramesTrain/batchSize;
    validationSetInput = new Mat<elem_type>();
    validationSetOutput = new Mat<elem_type>();
    *validationSetInput = inputData->submat(0,nFramesTrain-1,inputData->n_rows - 1,nFrames-1);
    *validationSetOutput = outputData->submat(0,nFramesTrain-1,outputData->n_rows - 1,nFrames-1);
//    cout<<"validation data copied"<<endl;
	arma_rng::set_seed_random();
	validError = numeric_limits<double>::max();
	cout<<"Training started";
	for(int epochNo =0;epochNo<nEpochs;epochNo++)
	{
        batchError=0;
		frameNos = randi<uvec>(nFramesTrain-(nFramesTrain%batchSize),distr_param(0,nFramesTrain-1));
//		frameNos << 1 << 4 << 0;
//		cout<<"Epoch: "<<epochNo<<""<<endl;
		for(int batchNo=0;batchNo<batchesPerEpoch;batchNo++)
		{
//            cout<<"Batch Number: "<<batchNo<<endl;
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

//CROSS VALIDATION
//		calculate validation error once every 5 Epochs and terminate training if validation error increases
		if((epochNo+1)%5 == 0)
		{
            nn->compute_output(*validationSetInput);
            prevValidError = validError;
            validError = nn->compute_outputerror(*validationSetOutput);
            valErrfh<<validError<<endl;
//            cout<<"previous validation error:"<<prevValidError<<endl;
//            cout<<"validation Error:"<<validError<<endl;
            if(validError > prevValidError)
            {
                validCount++;
                if(validCount >= VALIDATION_TOLERANCE)
                {
                    cout<<"Stopped Training after "<<epochNo<<" epochs as validation error increased"<<endl;
                    break;
                }
            }
            else
                validCount = 0;
		}

		cout<<".";
		cout.flush();
	}
	cout<<endl;
	cout<<"Training completed"<<endl;
//	nn->print_weights();
	nn->save_weights(weightsFname);
	endTime = clock();
	timeElapsed = (endTime-startTime)/((double)CLOCKS_PER_SEC*60);
	cout << fixed << showpoint << setprecision(2);
	cout<<"Time elapsed: "<< timeElapsed <<" minutes" <<endl;
	errfh.close();
}




