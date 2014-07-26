#include<iostream>
#include<armadillo>
#include<sstream>
#include<fstream>
#include<vector>
#include<ctime>
#include<cmath>
//#include "utils.cpp"
#include "dnn.h"

using namespace std;
using namespace arma;


DNN::DNN(Params &nnParams)
{
	read_nnparams(nnParams);
//	cout<<"units in each layer: ";
//	print_vec(unitsInLayer);
//	cout<<"output function of units in each layer: ";
//	print_vec(outFnType);
//	cout<<"batchsize/bathcesperpoch: "<<batchSize<<"/"<<batchesPerEpoch<<endl;
//	cout<<"learning rate: "<<eta<<endl;
	totalUnits=0;
	for(int i=1;i<unitsInLayer.size();i++)
		totalUnits +=unitsInLayer[i];
	inputDimension = unitsInLayer[0];
	outputDimension = unitsInLayer[nLayers];
//	cout<<"Total Units: "<<totalUnits<<endl;
	configure_network();
//	cout<<"configuring the network completed"<<endl;
	initialize_weights();
//	cout<<"weight  initialize completed"<<endl;
	_A=1.716;
	_B=2.0/3.0;
	_Bby2A=_B/(2*_A); //0.1943 Tanh Parameters
}

DNN::DNN(Params &nnParams,const char *wtsFname,string fileType)
{ //fileType represents how the data in weights and bias files are to be identified

	read_nnparams(nnParams);
//	cout<<"units in each layer: ";
//	print_vec(unitsInLayer);
//	cout<<"output function of units in each layer: ";
//	print_vec(outFnType);
//	cout<<"batchsize/bathcesperpoch: "<<batchSize<<"/"<<batchesPerEpoch<<endl;
//	cout<<"learning rate: "<<eta<<endl;
	totalUnits=0;
	for(int i=1;i<unitsInLayer.size();i++)
		totalUnits +=unitsInLayer[i];
	inputDimension = unitsInLayer[0];
	outputDimension = unitsInLayer[nLayers];
//	cout<<"Total Units: "<<totalUnits<<endl;
	configure_network();
//	cout<<"configuring the network completed"<<endl;
	if(fileType == "raw_ascii")
		read_weights(wtsFname,"raw_ascii");
	else if(fileType == "arma_ascii")
		read_weights(wtsFname,"arma_ascii");
	else
		cerr<<"Invalid file type"<<endl;
//	cout<<"weights and bias initialized from files"<<wtsFname<<" and "<<biasFname<<endl;
	_A=1.716;
	_B=2.0/3.0;
	_Bby2A=_B/(2*_A); //0.1943 Tanh Parameters

}

void DNN::configure_network()
{
	for(int layerNo=1;layerNo<=nLayers;layerNo++)
	{
		// create the outputs matrix of each layer say "h" [dim(h) x batchsize]
		output.push_back(new Mat<elem_type>(unitsInLayer[layerNo],batchSize,fill::zeros));
		//create the weights matrix of each layer say "h" [dim(h) x dim(h-1)]
		weights.push_back(randu< Mat<elem_type> >(unitsInLayer[layerNo],unitsInLayer[layerNo-1]));
		//create the bias' vector of units in each layer say "h" [dim(h)]
		bias.push_back(new Col<elem_type>());
		bias[layerNo-1]->randu(unitsInLayer[layerNo]);
		prevBiasGradient.push_back(new Col<elem_type>(unitsInLayer[layerNo],fill::zeros));
		curBiasGradient.push_back(new Col<elem_type>(unitsInLayer[layerNo],fill::zeros));
		//create the matrix for first derviative of each layer say "h" [dim(h) x batchsize]
		firstDerivative.push_back(new Mat<elem_type>(unitsInLayer[layerNo],batchSize,fill::zeros));
		//create the local gradients matrix of each layer say "h" [dim(h) x batchsize]
		localGradient.push_back(new Mat<elem_type>(unitsInLayer[layerNo],batchSize,fill::zeros));
		//create the  gradients matrix of each layer say "h" [dim(h) x dim(h-1)]
		prevGradient.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
		//create the gradients matrix of each layer say "h" [dim(h) x (h-1)]
		curGradient.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
		//create the delta weights matrix of each layer say "h" [dim(h) x dim(h-1)]
		deltaWeights.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
		//create the delta bias matrix of units in each layer say "h"[dim(h)]
		deltaBias.push_back(new Col<elem_type>(unitsInLayer[layerNo],fill::zeros));
		//intialize layer wise learning rate
		if(layerNo == nLayers)
		{
			//learning rate for all connections going into units in output layer
			layerLr.push_back(1.0/(25*totalUnits)); // OutLayer Low Learning Rate

			//learning rate for each connection going into units in output layer
			lrPerCon.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
			*(lrPerCon[layerNo-1]) = *lrPerCon[layerNo-1] + layerLr[layerNo-1];
		}
		else
		{
			//learning rate for each connection going into units in a hidden layer
			layerLr.push_back(1.0/(25*unitsInLayer[layerNo-1]));

			//learning rate for each connection going into units in a hidden layer
			lrPerCon.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
			*(lrPerCon[layerNo-1]) = *(lrPerCon[layerNo-1]) + layerLr[layerNo-1];

		}
//		cout<<"lr of layer "<<layerNo-1<<": "<<layerLr[layerNo-1]<<endl;
        lrgf.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::ones));
        biasgf.push_back(new Col<elem_type>(unitsInLayer[layerNo],fill::ones));
	}

	//create the output error matrix for output layer say "o" [dim(o) x batchsize]
	outputError = new Mat<elem_type>(unitsInLayer[nLayers],batchSize,fill::zeros);
	firstEpoch = true;
}

void DNN::read_nnparams(Params &nnParams)
{
	unitsInLayer = nnParams.unitsInLayer;
	outFnType = nnParams.outFnType;
	batchSize = nnParams.batchSize;
	batchesPerEpoch = nnParams.batchesPerEpoch;
	eta = nnParams.eta;
	nLayers = outFnType.size();
//	cout<<"no.of layers: "<<nLayers<<endl;
}

void DNN::initialize_weights()
{
	float maxweight;
//	arma_rng::set_seed(time(NULL));

//	for(int layerNo = 0; layerNo<nLayers; layerNo++)
//	{
//		weights.push_back(randu< Mat<elem_type> >(unitsInLayer[layerNo+1],unitsInLayer[layerNo]));
//		*(bias[layerNo]) = randu< Mat<elem_type> >(unitsInLayer[layerNo+1]);
//	}
	for(int layerNo=0; layerNo<nLayers; layerNo++)
	{
		maxweight=(float)3/sqrt((double)unitsInLayer[layerNo]);
		weights[layerNo] = 2*maxweight*weights[layerNo]-maxweight;
		*(bias[layerNo]) = 2*maxweight*(*bias[layerNo])-maxweight;
	}
//	for(int layerNo=0; layerNo<nLayers; layerNo++)
//	{
//		weights[layerNo].print("layer weights:");
//	}
}

void DNN::read_weights(const char* wtsFname,string fileType)
{
	ifstream wtsfh(wtsFname);
	rowvec tempRowVec = rowvec();
	char temp[100];
	if(fileType == "raw_ascii")
		for(int layerNo=0;layerNo<nLayers;layerNo++)
		{
			weights.push_back(Mat<elem_type>());
			weights[layerNo].load(wtsfh,raw_ascii);
			wtsfh.getline(temp,100);
			wtsfh.getline(temp,100);
			bias.push_back( new colvec());
			bias[layerNo]->load(wtsfh,raw_ascii);
			wtsfh.getline(temp,100);
			wtsfh.getline(temp,100);
		}
	else if(fileType == "arma_ascii")
		for(int layerNo=0;layerNo<nLayers;layerNo++)
		{
			weights.push_back(Mat<elem_type>());
			weights[layerNo].load(wtsfh,arma_ascii);
			bias.push_back( new colvec());
			bias[layerNo]->load(wtsfh,arma_ascii);
		}
    wtsfh.close();
}

void DNN::compute_output(Mat<elem_type> &input)
{
	Mat<elem_type> activation;
	activation=weights[0]*input;
	activation.each_col() += (*bias[0]);
//	cout<<"activation values of layer 0 computed"<<endl;
	output_function(activation,0);
//	cout<<"outputs of layer 0 computed"<<endl;
	for(int layerNo=1; layerNo<nLayers; layerNo++)
	{
		activation=weights[layerNo]*(*(output[layerNo - 1])); //compute the activation value of each unit
		activation.each_col() += (*bias[layerNo]);
//		cout<<"activation values of layer "<<layerNo<<" computed"<<endl;
		output_function(activation,layerNo); //apply the output function to the activation value of each unit
//		cout<<"outputs of layer "<<layerNo<<" computed"<<endl;
	}
//	output[nLayers-1]->print("output:");
}

void DNN::compute_output(Mat<elem_type> &input,vector< Mat<elem_type> > &weights,
						 vector< Col<elem_type> > &bias,vector< Mat<elem_type> > &output)
{
	Mat<elem_type> activation;
	activation=weights[0]*input;
	activation.each_col() += bias[0];
//	cout<<"activation values of layer 0 computed"<<endl;
	output_function(activation,0,output[0]);
//	cout<<"outputs of layer 0 computed"<<endl;
	for(int layerNo=1; layerNo<nLayers; layerNo++)
	{
		activation=weights[layerNo] * output[layerNo - 1]; //compute the activation value of each unit
		activation.each_col() += bias[layerNo];
//		cout<<"activation values of layer "<<layerNo<<" computed"<<endl;
		output_function(activation,layerNo,output[layerNo]); //apply the output function to the activation value of each unit
//		cout<<"outputs of layer "<<layerNo<<" computed"<<endl;
	}
//	output[nLayers-1]->print("output:");
}

void DNN::output_function(Mat<elem_type> &act,int layerNo)
{ // calculates the output given the activation values and the type(outFnType[layerNo]) of output function
	if("L"== outFnType[layerNo] || "l"== outFnType[layerNo])
		 (*(output[layerNo]))= act;
	else if("N"== outFnType[layerNo] || "n"== outFnType[layerNo])
		(*(output[layerNo]))= _A * tanh(_B*act);
	else if("S"== outFnType[layerNo] || "s"== outFnType[layerNo])
		(*(output[layerNo])) = 1/(1+_A*exp((-1*_B)*act));
	else if("SM" == outFnType[layerNo] || "sm" == outFnType[layerNo])
	{
		rowvec sum_rvec; // sum of all the outputs of the given layer for all frames in a batch
		(*(output[layerNo])) = exp(act);
		sum_rvec = sum(*(output[layerNo]));
		output[layerNo]->each_row() /= sum_rvec;
	}
	else
	{
		cout<<" Such a output function is not implemented"<<endl;
		exit(1);
	}
}

void DNN::output_function(Mat<elem_type> &act,int layerNo,Mat<elem_type> &temp_output)
{ // calculates the output given the activation values and the type(outFnType[layerNo]) of output function
	if("L"== outFnType[layerNo] || "l"== outFnType[layerNo])
		 temp_output = act;
	else if("N"== outFnType[layerNo] || "n"== outFnType[layerNo])
		temp_output = _A * tanh(_B*act);
	else if("S"== outFnType[layerNo] || "s"== outFnType[layerNo])
		temp_output = 1/(1+_A*exp((-1*_B)*act));
	else if("SM" == outFnType[layerNo] || "sm" == outFnType[layerNo])
	{
		rowvec sum_rvec; // sum of all the outputs of the given layer for all frames in a batch
		temp_output = exp(act);
		sum_rvec = sum(temp_output);
		temp_output.each_row() /= sum_rvec;
	}
	else
	{
		cout<<" Such an output function is not implemented"<<endl;
		exit(1);
	}
}

void DNN::gen_output(Mat<elem_type> &input,const char *outFileName,bool one_hot=false)
{ // generates the output of the NN for the given input data and saves in the given file
	rowvec row_temp;
	uword maxIdx;
	ofstream wtsfh; // used only for classification tasks
	compute_output(input);
	*(output[nLayers-1]) = output[nLayers-1]->t();
	if(!one_hot)
		output[nLayers-1]->save(outFileName,raw_ascii);
	else
	{
		wtsfh.open(outFileName,ios::out);
		for(int i=0;i<output[nLayers-1]->n_rows;i++)
		{
			row_temp = output[nLayers-1]->row(i);
			row_temp.max(maxIdx);
			maxIdx += 1;
			for(int j=1;j<=unitsInLayer[nLayers];j++)
			{
                if(j != maxIdx)
                    wtsfh<<0;
                else
                    wtsfh<<1;
                if(j == unitsInLayer[nLayers] )
                    wtsfh<<endl;
                else
                    wtsfh<<" ";
			}
		}
		wtsfh.close();
	}
//	output[nLayers-1]->save(outFileName,raw_ascii);
//	cout<<"Dimension of outputs is:"<<output[nLayers-1]->n_rows<<"x"<<output[nLayers-1]->n_cols<<endl;
	cout<<"outputs saved to file "<<outFileName<<endl;
}

double DNN::compute_outputerror(Mat<elem_type> &T)
{ // T is the desired output
	double mse;
	int nFrames;
	rowvec temp;
	nFrames = T.n_cols;
	(*outputError) = T - (*(output[nLayers-1]));
//	mse = accu(square(*(outputError))) / batchSize;
	temp = sum(square(*(outputError))) / sum(square(T));
//	cout<<"temp: "<<temp<<endl;
	mse = sum(temp) / nFrames;
	return mse;
}

double DNN::compute_outputerror(Mat<elem_type> &T,Mat<elem_type> &Y,Mat<elem_type> &outputError)
{ // T is the desired output and Y is the actual output
	double mse;
	int nFrames;
//	Mat<elem_type> *error = new Mat<elem_type>();
	nFrames = T.n_cols;
	outputError = T - Y;
	mse = sum(sum(square(outputError)));
	mse = mse / nFrames;
	return mse;
}

void DNN::compute_firstderivative(int layerNo)
{
	if("L"== outFnType[layerNo] || "l"== outFnType[layerNo])
		 firstDerivative[layerNo]->fill(1.0);
	else if("N"== outFnType[layerNo] || "n"== outFnType[layerNo])
		(*(firstDerivative[layerNo]))= _Bby2A*(_A - (*output[layerNo])) % (_A + (*output[layerNo]));
	else if("S"== outFnType[layerNo] || "s"== outFnType[layerNo])
		(*(firstDerivative[layerNo])) = _B * (*output[layerNo]) % (1 - (*output[layerNo]));
	else if("SM"== outFnType[layerNo] || "sm"== outFnType[layerNo])
			(*(firstDerivative[layerNo])) = (*output[layerNo]) % (1 - (*output[layerNo]));
	else
	{
		cout<<"First derivative of such a output function is not implemented"<<endl;
		exit(1);
	}
}

void DNN::compute_firstderivative(int layerNo,Mat<elem_type> &firstDerivative,vector< Mat<elem_type> > &output)
{
	if("L"== outFnType[layerNo] || "l"== outFnType[layerNo])
		 firstDerivative.ones(output[layerNo].n_rows,output[layerNo].n_cols);
	else if("N"== outFnType[layerNo] || "n"== outFnType[layerNo])
		firstDerivative = _Bby2A* (_A - output[layerNo]) % (_A + output[layerNo]);
	else if("S"== outFnType[layerNo] || "s"== outFnType[layerNo])
		firstDerivative = _B * output[layerNo] % (1 - output[layerNo]);
	else if("SM"== outFnType[layerNo] || "sm"== outFnType[layerNo])
			firstDerivative = output[layerNo] % (1 - output[layerNo]);
	else
	{
		cout<<"First derivative of such a output function is not implemented"<<endl;
		exit(1);
	}
}

void DNN::compute_localgradients()
{
	for(int layerNo=nLayers-1;layerNo>=0;layerNo--)
	{
		compute_firstderivative(layerNo);
		if(layerNo == nLayers-1)
			*(localGradient[layerNo])=(*outputError)%(*(firstDerivative[layerNo]));
		else
			*(localGradient[layerNo])=weights[layerNo+1].t() * (*localGradient[layerNo+1]) % (*(firstDerivative[layerNo]));
	}

//	for(int layerNo = 0;layerNo<nLayers;layerNo++)
//		localGradient[layerNo]->print("Local Gradients:");
}


void DNN::compute_localgradients(vector< Mat<elem_type> > &weights, vector< Mat<elem_type> > &output,
								 Mat<elem_type> &outputError, vector< Mat<elem_type> > &localGradient)
{
	Mat<elem_type> firstDerivative;
	for(int layerNo=nLayers-1;layerNo>=0;layerNo--)
	{
		compute_firstderivative(layerNo,firstDerivative,output);
		if(layerNo == nLayers-1)
			localGradient[layerNo] = outputError % firstDerivative;
		else
			localGradient[layerNo] = (weights[layerNo+1].t() * localGradient[layerNo+1]) % firstDerivative;
	}

//	for(int layerNo = 0;layerNo<nLayers;layerNo++)
//		localGradient[layerNo]->print("Local Gradients:");
}

void DNN::compute_gradients(Mat<elem_type> &input)
{
	int nFrames;
	nFrames = input.n_cols;
    for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
        *(prevGradient[layerNo]) = *(curGradient[layerNo]);
        *(prevBiasGradient[layerNo]) = *(curBiasGradient[layerNo]);

		//gradient of error w.r.t weights
		if(layerNo==0)
			*(curGradient[layerNo]) = (1.0/nFrames) * (*(localGradient[layerNo])) * input.t();
		else
			*(curGradient[layerNo]) = (1.0/nFrames) * (*(localGradient[layerNo])) * (output[layerNo-1])->t();

        //gradient of error w.r.t bias of each unit
        *(curBiasGradient[layerNo]) = (1.0/nFrames) * sum(*(localGradient[layerNo]),1);

//        curGradient[layerNo]->print("curGradient:");
//        curBiasGradient[layerNo]->print("curBiasGradient");
	}
}

void DNN::compute_gradients(Mat<elem_type> &input,vector< Mat<elem_type> > &output,
							vector< Mat<elem_type> > &localGradient,vector< Mat<elem_type> > &weightsGrad,
							vector< Col<elem_type> > &biasGrad)
{
	int nFrames;
	nFrames = input.n_cols;
    for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
//        *(prevGradient[layerNo]) = *(curGradient[layerNo]);
//        *(prevBiasGradient[layerNo]) = *(curBiasGradient[layerNo]);

		//gradient of error w.r.t weights
		if(layerNo==0)
			weightsGrad[layerNo] = (1.0/nFrames) * localGradient[layerNo] * input.t();
		else
			weightsGrad[layerNo] = (1.0/nFrames) * localGradient[layerNo] * output[layerNo-1].t();

        //gradient of error w.r.t bias of each unit
        biasGrad[layerNo] = (1.0/nFrames) * sum(localGradient[layerNo],1);

//        weightsGradient[layerNo]->print("weightsGradient:");
//        biasGradient[layerNo]->print("biasGradient");
	}
}

void DNN::adapt_lrgf()
{
    Mat<elem_type> tempWts;
    Col<elem_type> tempBias;
    if(!firstEpoch)
    for(int layerNo=0;layerNo<nLayers;layerNo++)
    {
        tempWts = (*(prevGradient[layerNo])) % (*(curGradient[layerNo]));
        tempBias = (*(prevBiasGradient[layerNo])) % (*(curBiasGradient[layerNo]));

        for(int i=0;i<tempWts.n_rows;i++)
            for(int j=0;j<tempWts.n_cols;j++)
                if(tempWts(i,j)>0)
                {
                    (*lrgf[layerNo])(i,j) += 0.05;
//                    cout<<"increased"<<endl;
                }
                else
                {
                    (*lrgf[layerNo])(i,j) *= 0.95;
//                    cout<<"decreased"<<endl;
                }
        for(int i=0;i<tempBias.n_rows;i++)
            if(tempBias(i)>0)
                (*(biasgf[layerNo]))(i) += 0.05;
            else
                (*(biasgf[layerNo]))(i) *= 0.95;
    }
    else
        firstEpoch = false;

}

void DNN::compute_deltas(Mat<elem_type> &input,float momentum)
{
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		if(layerNo==0)
			*(deltaWeights[layerNo]) = (eta/batchSize) * (*(localGradient[layerNo])) * input.t();
		else
			*(deltaWeights[layerNo]) = (eta/batchSize) * (*(localGradient[layerNo])) * (output[layerNo-1])->t();

		*(deltaBias[layerNo]) = (eta/batchSize) * sum(*(localGradient[layerNo]),1);
	}
}

void DNN::compute_deltaswithmomentum(Mat<elem_type> &input,float momentum)
{
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		if(layerNo==0)
			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (eta/batchSize) * (*(localGradient[layerNo])) * input.t();
		else
			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (eta/batchSize) * (*(localGradient[layerNo])) * (output[layerNo-1])->t();

		*(deltaBias[layerNo]) = momentum * (*(deltaBias[layerNo])) + (eta/batchSize) * sum(*(localGradient[layerNo]),1);
	}
}

void DNN::compute_deltaswithmomandlayerlr(Mat<elem_type> &input,float momentum)
{
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		if(layerNo==0)
			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (layerLr[layerNo]/batchSize) * (*(localGradient[layerNo]) * input.t());
		else
			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (layerLr[layerNo]/batchSize) * (*(localGradient[layerNo]) * (output[layerNo-1])->t());

		*(deltaBias[layerNo]) = momentum * (*(deltaBias[layerNo])) + (layerLr[layerNo]/batchSize) * sum(*(localGradient[layerNo]),1);
	}
}

void DNN::compute_deltaswithdlrandmom(float momentum)
{
    adapt_lrgf();
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
//		if(layerNo==0)
//			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (layerLr[layerNo]/batchSize) * (*(lrgf[layerNo]) % (*(curGradient[layerNo])));
//		else
//			*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + (layerLr[layerNo]/batchSize) * (*(lrgf[layerNo]) % (*(curGradient[layerNo])));

		*(deltaWeights[layerNo]) = momentum * (*(deltaWeights[layerNo])) + layerLr[layerNo] * (*(lrgf[layerNo]) % (*(curGradient[layerNo])));
		*(deltaBias[layerNo]) = momentum * (*(deltaBias[layerNo])) + layerLr[layerNo] * (*(biasgf[layerNo]) % (*(curBiasGradient[layerNo])));
	}
}


void DNN::increment_weights()
{
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		weights[layerNo] = weights[layerNo] + (*(deltaWeights[layerNo]));
		*(bias[layerNo]) = *(bias[layerNo]) + (*(deltaBias[layerNo]));
	}
}

void DNN::save_weights(const char *wtsFname)
{
	ofstream wtsfh(wtsFname);
//	ofstream biasfh(biasFname);
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		if(!(weights[layerNo]).save(wtsfh,arma_ascii))
			cerr<<"Error in saving weights of layer "<<layerNo<<endl;
		wtsfh<<endl;
		if(!bias[layerNo]->save(wtsfh,arma_ascii))
			cerr<<"Error in saving biases of layer "<<layerNo<<endl;
		wtsfh<<endl;
	}
	wtsfh.close();
//	biasfh.close();
}

void DNN::print_weights()
{
	cout<<"WEIGHTS:";
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		cout<<endl;
		cout<<weights[layerNo];
	}

	cout<<"BIAS:";
	for(int layerNo=0;layerNo<nLayers;layerNo++)
	{
		cout<<endl;
		bias[layerNo]->print();
	}
}



