
#ifndef DNN_H_
#define DNN_H_

#include<string>
#include<armadillo>
#include<vector>

using namespace arma;
using namespace std;

typedef double elem_type;

class DNN
{
	vector< Mat<elem_type> > weights;
	int nLayers;
	vector<int> unitsInLayer; //number of units in each layer
	int totalUnits;
	float eta;	//learning rate
	vector<float> layerLr; //layer wise learning rate
	vector<float> localGradients;
	const char *paramsFileName; //file containing parameters for the neural network
	vector<string> outFnType; //type of output function for each layer
	vector< Mat<elem_type>* > output; //outputs of each layer for a given input batch
	Mat<elem_type>* outputError;
	vector<colvec*> bias;
	vector< Mat<elem_type>* > localGradient; // Note: local gradients of all connections into an unit are same
								// for a given input pattern
	vector< Mat<elem_type>* > firstDerivative;
	vector< Mat<elem_type>* > deltaWeights;
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
	void compute_output(Mat<elem_type> &input);	/* Pass the Input Pattern*/
	void compute_deltas();
	void compute_deltaswithmomentum(Mat<elem_type>&,float);
	void compute_deltaswithmomandlayerlr(Mat<elem_type>&,float);
	void output_function(Mat<elem_type>&,int);
	double compute_outputerror(Mat<elem_type>&); // computes the difference in original and desired outputs
								   //and MSE(mean squared error)
	void compute_firstderivative(int);
	void compute_localgradients();
	void gen_output(Mat<elem_type>&,const char*,bool);
	void compute_deltas(Mat<elem_type>&,float);
	void increment_weights();
	void save_weights(const char*,const char*);
	void read_weights(const char*,const char*,string);
	void print_weights();
};

#endif /* DNN_UTILS_H_ */
