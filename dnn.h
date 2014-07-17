
#ifndef DNN_H_
#define DNN_H_

#include<string>
#include<armadillo>
#include<vector>

using namespace arma;
using namespace std;

class DNN
{
	vector< Mat<elem_type> > weights;
	int nLayers;
	vector<int> unitsInLayer; //number of units in each layer
	int totalUnits;
	float eta;	//learning rate
	vector<float> layerLr; //layer wise learning rate
	vector< Mat<elem_type>* > lrPerCon; //learning rate per each connection
	vector< Mat<elem_type>* > lrgf; //learning rate gain factor. Used in adapative learning rate
    vector< Col<elem_type>* >biasgf; //bias gain factor
	bool firstEpoch;
	vector<float> localGradients;
	const char *paramsFileName; //file containing parameters for the neural network
	vector<string> outFnType; //type of output function for each layer
	vector< Mat<elem_type>* > output; //outputs of each layer for a given input batch
	Mat<elem_type>* outputError;
	vector< Col<elem_type>* > bias;
	vector< Mat<elem_type>* > localGradient; // Note: local gradients of all connections into an unit are same
								// for a given input pattern
	vector< Mat<elem_type>* > prevGradient;
	vector< Mat<elem_type>* > curGradient;
	vector< Mat<elem_type>* > firstDerivative;
	vector< Mat<elem_type>* > deltaWeights;
	vector< Col<elem_type>* > deltaBias;
	vector< Col<elem_type>* > prevBiasGradient;
	vector< Col<elem_type>* > curBiasGradient;

	float _A;		// Tanh Parameter
	float _B;		// Tanh Parameter
	float _Bby2A;

public:
	int batchSize;
	int batchesPerEpoch;
	int inputDimension;
	int outputDimension;
	DNN(const char*);
	DNN(const char*,const char*,string);
	void initialize_weights();
	void configure_network();
	void read_nnparams();
	void compute_output(Mat<elem_type> &input);	/* Pass the Input Pattern*/
//	void compute_deltas();
	void output_function(Mat<elem_type>&,int);
	double compute_outputerror(Mat<elem_type>&); // computes the difference in original and desired outputs
								   //and MSE(mean squared error)
	void compute_firstderivative(int);
	void compute_localgradients();
	void compute_gradients(Mat<elem_type>&);
	void adapt_lrgf();
	void gen_output(Mat<elem_type>&,const char*,bool);
	void compute_deltas(Mat<elem_type>&,float);
    void compute_deltaswithmomentum(Mat<elem_type>&,float);
	void compute_deltaswithmomandlayerlr(Mat<elem_type>&,float);
	void compute_deltaswithdlrandmom(float);
	void increment_weights();
	void save_weights(const char*);
	void read_weights(const char*,string);
	void print_weights();
};

#endif /* DNN_UTILS_H_ */
