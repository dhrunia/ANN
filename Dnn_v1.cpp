#include<iostream>
#include<armadillo>
#include<sstream>
#include<fstream>
#include<vector>
#include<ctime>
#include<cmath>
#include "utils.cpp"
#include "dnn.h"

using namespace std;
using namespace arma;


DNN::DNN(const char *dnnParamsFname)
{
	paramsFileName = dnnParamsFname;
//	cout<<"parameters file name: "<<paramsFileName<<endl;
	read_nnparams();
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

DNN::DNN(const char *dnnParamsFname,const char *wtsFname,string fileType)
{ //fileType represents how the data in weights and bias files are to be identified

	paramsFileName = dnnParamsFname;
//	cout<<"parameters file name: "<<paramsFileName<<endl;
	read_nnparams();
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
		//create the bias' vector of units in each layer say "h" [dim(h)]
		bias.push_back(new colvec(unitsInLayer[layerNo],fill::zeros));
		//create the matrix for first derviative of each layer say "h" [dim(h) x batchsize]
		firstDerivative.push_back(new Mat<elem_type>(unitsInLayer[layerNo],batchSize,fill::zeros));
		//create the local gradients matrix of each layer say "h" [dim(h) x batchsize]
		localGradient.push_back(new Mat<elem_type>(unitsInLayer[layerNo],batchSize,fill::zeros));
		//create the delta weights matrix of each layer say "h" [dim(h) x dim(h-1)]
		deltaWeights.push_back(new Mat<elem_type>(unitsInLayer[layerNo],unitsInLayer[layerNo-1],fill::zeros));
		//create the delta bias matrix of units in each layer say "h"[dim(h)]
		deltaBias.push_back(new colvec(unitsInLayer[layerNo],fill::zeros));
		//intialize layer wise learning rate
		if(layerNo == nLayers)
		{
			layerLr.push_back(1.0/(totalUnits)); // OutLayer Low Learning Rate
		}
		else
		{
			layerLr.push_back(1.0/(unitsInLayer[layerNo-1]));
		}
//		cout<<"lr of layer "<<layerNo-1<<": "<<layerLr[layerNo-1]<<endl;
	}
	//create the output error matrix for output layer say "o" [dim(o) x batchsize]
	outputError = new Mat<elem_type>(unitsInLayer[nLayers],batchSize,fill::zeros);
}

void DNN::read_nnparams()
{
	ifstream fh(paramsFileName);
	string line;
	vector<int> temp;
	for(int lineNo=1;getline(fh,line);lineNo++)
	{
		switch(lineNo)
		{
		case 1:
			str_to_vec(line,unitsInLayer);
			break;
		case 2:
			str_to_vec(line,outFnType);
			break;
		case 3:
			str_to_vec(line,temp);
			batchSize = temp[0];
			if(temp.size()==2)
				batchesPerEpoch = temp[1];
			else
				batchesPerEpoch = 0;
			break;
		case 4:
			str_to_type(line,eta);
			break;
		}
	}
	nLayers = outFnType.size();
//	cout<<"no.of layers: "<<nLayers<<endl;
}

void DNN::initialize_weights()
{
	float maxweight;
	arma_rng::set_seed(time(NULL));

	for(int layerNo = 0; layerNo<nLayers; layerNo++)
	{
		weights.push_back(randu< Mat<elem_type> >(unitsInLayer[layerNo+1],unitsInLayer[layerNo]));
		*(bias[layerNo]) = randu< Mat<elem_type> >(unitsInLayer[layerNo+1]);
	}
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
			wtsfh<<maxIdx<<endl;
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
	temp = sum(sum(square(*(outputError))) / sum(square(T)));
	mse = sum(temp) / nFrames;
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

