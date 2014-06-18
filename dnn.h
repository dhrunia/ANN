/*
 * dnn_utils.h
 *
 *  Created on: Jun 9, 2014
 *      Author: anirudh
 */

#ifndef DNN_H_
#define DNN_H_

#include<string>
#include<armadillo>
#include<vector>

using namespace arma;
using namespace std;

class DNN
{
	vector<mat> weights;
	int nLayers;
	vector<int> unitsInLayer; //number of units in each layer
	int totalUnits;
	float eta;	//learning rate
	vector<float> layerLr; //layer wise learning rate
	vector<float> localGradients;
	const char *paramsFileName; //file containing parameters for the neural network
	vector<string> outFnType; //type of output function for each layer
	vector<mat*> output; //outputs of each layer for a given input batch
	mat* outputError;
	vector<colvec*> bias;
	vector<mat*> localGradient; // Note: local gradients of all connections into an unit are same
								// for a given input pattern
	vector<mat*> firstDerivative;
	vector<mat*> deltaWeights;
	vector<colvec*> deltaBias;
	float _A;		// Tanh Parameter
	float _B;		// Tanh Parameter
	float _Bby2A;

public:
	int batchSize;
	int batchesPerEpoch;
	int inputDimension;
	int outputDimension;
	DNN(const char*);
	DNN(const char*,const char*,const char*,string);
	void initialize_weights();
	void configure_network();
	void read_nnparams();
	void compute_output(mat &input);	/* Pass the Input Pattern*/
	void compute_deltas();
	void compute_deltaswithmomentum(mat&,float);
	void compute_deltaswithmomandlayerlr(mat&,float);
	void output_function(mat&,int);
	double compute_outputerror(mat&); // computes the difference in original and desired outputs
								   //and MSE(mean squared error)
	void compute_firstderivative(int);
	void compute_localgradients();
	void gen_output(mat&,const char*,bool);
	void compute_deltas(mat&,float);
	void increment_weights();
	void save_weights(const char*,const char*);
	void read_weights(const char*,const char*,string);
	void print_weights();
};

#endif /* DNN_UTILS_H_ */
