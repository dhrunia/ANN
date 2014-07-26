#include "dlib/optimization.h"
#include "dnn.h"


CGD::CGD()
{
//	cout<<"In CGD constructor"<<endl;
//	input = &inputData;
//	output = &outputData;
//	cout<<"nLayers:"<<nn.nLayers<<endl;
//	for(int layerNo = 1;layerNo <= nn->nLayers;layerNo++)
//	{
//		conjGrad.push_back(new Mat<elem_type>(nn->unitsInLayer[layerNo],
//						   nn->unitsInLayer[layerNo-1],fill::zeros));
////		cout<<"Dimension of conjGrad["<<layerNo<<"]: "<<nn.unitsInLayer[layerNo]<<"x"<<nn.unitsInLayer[layerNo-1]<<endl;
//		conjBiasGrad.push_back(new Col<elem_type>(nn->unitsInLayer[layerNo],fill::zeros));
////		cout<<"Dimension of conjGrad["<<layerNo<<"]: "<<nn.unitsInLayer[layerNo]<<endl;
//	}
//	cout<<"conjugate weights and bias gradients created"<<endl;
	sigma = 0.1;
	rho = sigma/2;
//	cout<<"exiting CGD constructor"<<endl;
}

void CGD::initialise(Mat<elem_type> *inputData, Mat<elem_type> *outputData, DNN *nn_init)
{
//	cout<<"inside initialise"<<endl;
	CGD::input = inputData;
	CGD::output = outputData;
	CGD::nn = nn_init;
	cout<<"nLayers: "<<nn->nLayers<<endl;
	for(int layerNo = 1;layerNo <= nn->nLayers;layerNo++)
	{
		CGD::conjGrad.push_back(new Mat<elem_type>(nn->unitsInLayer[layerNo],
											  nn->unitsInLayer[layerNo-1],fill::zeros));
//		cout<<"Dimension of conjGrad["<<layerNo<<"]: "<<nn->unitsInLayer[layerNo]<<"x"<<nn->unitsInLayer[layerNo-1]<<endl;
		CGD::conjBiasGrad.push_back(new Col<elem_type>(nn->unitsInLayer[layerNo],fill::zeros));
//		cout<<"Dimension of conjGrad["<<layerNo<<"]: "<<nn->unitsInLayer[layerNo]<<endl;
	}
	cout<<"conjugate weights and bias gradients created"<<endl;
}

double CGD::cgd(int epochNo)
{
	double error;
	double f0,d0; // average error(cost function) and its derivative at alpha = 0, i.e f(w) and f'(w)

	nn->compute_output(*input);
//	cout<<"outputs computed"<<endl;
	error = nn->compute_outputerror(*output);
//	cout<<"error computed"<<endl;
	nn->compute_localgradients();
//	cout<<"local gradients computed"<<endl;
	nn->compute_gradients(*input);
//	cout<<"gradients computed"<<endl;
	if(epochNo == 0)
	{
		for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
		{
			*(conjGrad[layerNo]) = *(nn->curGradient[layerNo]);
			*(conjBiasGrad[layerNo]) = *(nn->curBiasGradient[layerNo]);
		}
//		cout<<"conjugate gradients computed"<<endl;
	}
	else
	{
		compute_beta();
		cout<<"Beta: "<<beta<<endl;
		for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
		{
			*(conjGrad[layerNo]) = *(nn->curGradient[layerNo]) + beta * (*(conjGrad[layerNo]));
			*(conjBiasGrad[layerNo]) = *(nn->curBiasGradient[layerNo]) + beta * (*(conjBiasGrad[layerNo]));
		}
//		cout<<"conjugate gradients computed"<<endl;
	}
	f0 = avgerror(0.0);
//	cout<<"f0: "<<f0<<endl;
	d0 = avgerror_der(0.0);
//	cout<<"d0: "<<d0<<endl;
	double fval = 0;
//	eta = 0;
//	fval = dlib::find_min_single_variable(CGD::avgerror,eta,0,1,1e-3,1e5);
	eta = dlib::line_search(CGD::avgerror,f0,CGD::avgerror_der,d0,rho,sigma,0,1000000);
//	eta = dlib::backtracking_line_search(CGD::avgerror,f0,d0,1,rho,1000);
//	eta = 0.1;
//	cout<<"fval: "<<fval<<endl;
	cout<<"Eta: "<<eta<<endl;
	adjust_weights();
//	cout<<"weights adjusted"<<endl;
	return error;
}

double CGD::avgerror(double alpha)
{ //compute the average error for the given alpha i.e f(start + alpha*direction) where start is the
  //current weights and direction is the conjugate gradient direction.

	std::vector< Mat<elem_type> > weights;
	std::vector< Col<elem_type> > bias;
	vector< Mat<elem_type> > tempOutput;
	Mat<elem_type> activation;
	Mat<elem_type> outputError;
	double avgError;
	for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
	{
		weights.push_back(nn->weights[layerNo] + alpha * (*(conjGrad[layerNo])) );
		bias.push_back(*(nn->bias[layerNo]) + alpha * (*(conjBiasGrad[layerNo])) );
		tempOutput.push_back(Mat<elem_type>());
	}

//	for(int layerNo=0; layerNo<nn->nLayers; layerNo++)
//	{
//		if(layerNo == 0)
//			activation=weights[layerNo]* (*input); //compute the activation value of each unit in layer zero
//		else
//			activation=weights[layerNo] * tempOutput; //compute the activation value of each unit in layer h > 0
//		activation.each_col() += (bias[layerNo]);
////		cout<<"activation values of layer "<<layerNo<<" computed"<<endl;
//		nn->output_function(activation,layerNo,tempOutput); //apply the output function to the activation value of each unit
////		cout<<"outputs of layer "<<layerNo<<" computed"<<endl;
//	}

	nn->compute_output(*input,weights,bias,tempOutput);
	avgError = nn->compute_outputerror(*output,tempOutput[nn->nLayers-1],outputError);
	return avgError;
}

double CGD::avgerror_der(double alpha)
{ //compute the derivate average error for the given alpha i.e f'(start + alpha*direction) where start is the
  //current weights and direction is the conjugate gradient direction.
	Col<elem_type> r; // flattened negative gradient of cost function(avgerror)
	Col<elem_type> s; // flattened conjugate gradient direction
	std::vector< Mat<elem_type> > weights;
	std::vector< Col<elem_type> > bias;
	std::vector< Mat<elem_type> > tempOutput; //store the outputs computed using current weights(=w(n)+alpha*s(n))
	Mat<elem_type> outputError;
	std::vector< Mat<elem_type> > localGrad;
	std::vector< Mat<elem_type> > wtsGrad;
	std::vector< Col<elem_type> > biasGrad;
	Mat<elem_type> temp_output;
	Mat<elem_type> activation;
	double avgError;
	double avgerror_der = 0;
	for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
	{
		weights.push_back(nn->weights[layerNo] + alpha * (*(conjGrad[layerNo])) );
		bias.push_back(*(nn->bias[layerNo]) + alpha * (*(conjBiasGrad[layerNo])) );
		tempOutput.push_back(Mat<elem_type>());
		localGrad.push_back(Mat<elem_type>());
		wtsGrad.push_back(Mat<elem_type>());
		biasGrad.push_back(Col<elem_type>());
	}
	nn->compute_output(*input,weights,bias,tempOutput);
	avgError = nn->compute_outputerror(*output,tempOutput[nn->nLayers-1],outputError);
	nn->compute_localgradients(weights,tempOutput,outputError,localGrad);
	nn->compute_gradients(*input,tempOutput,localGrad,wtsGrad,biasGrad);
	for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
	{
		r = join_cols(vectorise(wtsGrad[layerNo]),biasGrad[layerNo]);
		s = join_cols(vectorise(*(conjGrad[layerNo])),*(conjBiasGrad[layerNo]));
		avgerror_der += dot(r,s) ;
	}
	return -(avgerror_der);

}

void CGD::adjust_weights()
{
	for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
	{
		nn->weights[layerNo] = nn->weights[layerNo] + eta * (*(conjGrad[layerNo]));
		*(nn->bias[layerNo]) = *(nn->bias[layerNo]) + eta * (*(conjBiasGrad[layerNo]));

	}
}

void CGD::compute_beta()
{
	Col<elem_type> r; // temporary variable to store flattened current gradient
	Col<elem_type> old_r; // temporary variable to store flattened previous gradient
	double beta_temp;
	double beta_num = 0; //numerator in the Polak Ribere formula from "Neural networks and machine learning by Haykin pg:191"
	double beta_den = 0; //denominator in the Polak Ribere formula "Neural networks and machine learning by Haykin pg:191"
	for(int layerNo = 0;layerNo < nn->nLayers;layerNo++)
	{
		r = join_cols(vectorise(*(nn->curGradient[layerNo])),*(nn->curBiasGradient[layerNo]));
		old_r = join_cols(vectorise(*(nn->prevGradient[layerNo])),*(nn->prevBiasGradient[layerNo]));
//		r.print("r:");
//		old_r.print("old_r:");
		beta_num += dot(r,(r - old_r));
//		beta_num += dot(r,r);
		beta_den += dot(old_r,old_r);
//		cout<<"beta_num:"<<beta_num<<endl;
//		cout<<"beta_den:"<<beta_den<<endl;
		r.reset();
		old_r.reset();
	}
	beta_temp = beta_num / beta_den;
//	cout<<"beta_temp: "<<beta_temp<<endl;
	if(beta_temp > 0)
		beta = beta_temp;
	else
		beta = 0;
}

void CGD::train(int nEpochs,const char *weightsFname)
{
	ofstream err("errors.txt");

	if(!err.is_open())
		cerr<<"unable to open file errors.txt"<<endl;
	double error;

	for(int epochNo=0;epochNo<nEpochs;epochNo++)
	{
		cout<<endl;
		cout<<"Epoch No: "<<epochNo<<endl;
		error = cgd(epochNo);
		err<<error<<endl;
		cout<<"Error: "<<error<<endl;
		nn->save_weights(weightsFname);
	}
//	cout<<"weights saved to file "<<weightsFname<<endl;
}

double BGD::bgd(DNN &nn,Mat<elem_type> &batchInput, Mat<elem_type> &batchOutput,
							  float momentum,double prevError1,double prevError2,int epochNo)
{ // mini-batch Gradient descent algorithm
    double batchError;
    nn.compute_output(batchInput);
//			cout<<"outputs computed"<<endl;
    batchError = nn.compute_outputerror(batchOutput);
//			cout<<"error computed"<<endl;
    nn.compute_localgradients();
//			cout<<"local gradients computed"<<endl;
    nn.compute_gradients(batchInput);
//			nn->compute_deltas(*batchInput);
//			nn->compute_deltaswithmomandlayerlr(*batchInput,momentum);

    if(abs(prevError1-prevError2) < 1e-5 && epochNo >= 2)
        nn.compute_deltaswithdlrandmom(0.99);
//                nn->compute_deltaswithmomandlayerlr(*batchInput,0.9);
    else if(abs(prevError1-prevError2) < 1e-4 && epochNo >= 2)
        nn.compute_deltaswithdlrandmom(0.9);
//                nn->compute_deltaswithmomandlayerlr(*batchInput,0.7);
    else if(abs(prevError1-prevError2) < 1e-3 && epochNo >= 2)
        nn.compute_deltaswithdlrandmom(0.8);
//                nn->compute_deltaswithmomandlayerlr(*batchInput,0.5);
    else
//        nn.compute_deltaswithdlrandmom(momentum);
    	nn.compute_deltas(batchInput,momentum);
//                nn->compute_deltaswithmomandlayerlr(*batchInput,momentum);

//			cout<<"deltas computed"<<endl;
    nn.increment_weights();
//			cout<<"weights incremented"<<endl;
    return batchError;
}


void BGD::train(DNN &nn,Mat<elem_type> &input, Mat<elem_type> &output,
				float momentum, int nEpochs, const char *weightsFname)
{
	const char *errorFname,*valErrFname;
	int nFramesTrain,nFramesValid,nFrames,batchesPerEpoch,batchSize,batchSP,validCount;
	double batchError,prevError1,prevError2,validError,prevValidError;
	uvec frameNos,batchFrameNos;
	Mat<elem_type> *batchInput;
	Mat<elem_type> *batchOutput;
	Mat<elem_type> *validationSetInput,*validationSetOutput;
	errorFname = "errors.txt";
	valErrFname = "valid_error.txt";
	batchSize = nn.batchSize;
	validCount = 0;
	batchInput = new Mat<elem_type>(nn.inputDimension,batchSize,fill::zeros);
	batchOutput = new Mat<elem_type>(nn.outputDimension,batchSize,fill::zeros);
	ofstream errfh(errorFname);
	ofstream valErrfh(valErrFname);
	if(!errfh.is_open())
		cout<<"unable to open file "<<errorFname<<endl;
	if(!valErrfh.is_open())
		cout<<"unable to open file "<<valErrFname<<endl;
	nFrames = input.n_cols;
	nFramesTrain = 0.75*nFrames;
	nFramesValid = nFrames - nFramesTrain;
    cout<<"nFramesTrain: "<<nFramesTrain<<endl;
//    cout<<"nFramesValid: "<<nFramesValid<<endl;
	if(nn.batchesPerEpoch)
		batchesPerEpoch = nn.batchesPerEpoch;
	else
		batchesPerEpoch = nFramesTrain/batchSize;
	cout<<"batchesPerEpoch:"<<batchesPerEpoch<<endl;
	validationSetInput = new Mat<elem_type>();
	validationSetOutput = new Mat<elem_type>();
	*validationSetInput = input.submat(0,nFramesTrain,input.n_rows - 1,nFrames-1);
	*validationSetOutput = output.submat(0,nFramesTrain,output.n_rows - 1,nFrames-1);
//	arma_rng::set_seed_random();
	validError = numeric_limits<double>::max();
//	cout<<"Training started";
	for(int epochNo =0;epochNo<nEpochs;epochNo++)
	{
		batchError=0;
//		frameNos = randi<uvec>(nFramesTrain-(nFramesTrain%batchSize),distr_param(0,nFramesTrain-1));
		frameNos = linspace<uvec>(0,nFramesTrain-1,nFramesTrain);
		frameNos = shuffle(frameNos);
//		frameNos << 1 << 4 << 0;
//		cout<<"Epoch: "<<epochNo<<""<<endl;
		for(int batchNo=0;batchNo<batchesPerEpoch;batchNo++)
		{
//            cout<<"Batch Number: "<<batchNo<<endl;
			batchSP = batchNo*batchSize;
			batchFrameNos = frameNos.subvec(batchSP,batchSP+batchSize-1);
			*batchInput = input.cols(batchFrameNos);
			*batchOutput = output.cols(batchFrameNos);
			batchError += bgd(nn,*batchInput,*batchOutput,momentum,prevError1,prevError2,epochNo) / batchesPerEpoch;
		}
		prevError2 = prevError1;
		prevError1 = batchError;
		cout<<"Error: "<<batchError<<endl;
		errfh<<batchError<<endl;

//CROSS VALIDATION
//		calculate validation error once every 5 Epochs and terminate training if validation error increases
		if((epochNo+1)%5 == 0)
		{
			nn.compute_output(*validationSetInput);
			prevValidError = validError;
			validError = nn.compute_outputerror(*validationSetOutput);
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
		nn.save_weights(weightsFname);
//		cout<<".";
//		cout.flush();
	}
	cout<<endl;
	cout<<"Training completed"<<endl;
//	nn->print_weights();

	errfh.close();
}
