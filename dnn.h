
#ifndef DNN_H_
#define DNN_H_

typedef double elem_type;

#include<string>
#include<armadillo>
#include<vector>

#define VALIDATION_TOLERANCE 3 //validation error tolerance count. If validation error contiguously increases
                               //for more than defined here training is stopped.

using namespace arma;
using namespace std;

class Params
{
	public:
		vector<int> unitsInLayer;
//		int nLayers;
		vector<string> outFnType;
		int batchSize;
		int batchesPerEpoch;
		float eta;
		string trainAlgoType;
		Params(const char*);
};

class DNN
{
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
	vector< Mat<elem_type>* > localGradient; // Note: local gradients of all connections into an unit are same
								// for a given input pattern
//	vector< Mat<elem_type>* > prevGradient;
//	vector< Mat<elem_type>* > curGradient;
	vector< Mat<elem_type>* > firstDerivative;
	vector< Mat<elem_type>* > deltaWeights;
	vector< Col<elem_type>* > deltaBias;

	float _A;		// Tanh Parameter
	float _B;		// Tanh Parameter
	float _Bby2A;

public:
	vector< Mat<elem_type> > weights;
	vector< Col<elem_type>* > bias;
	vector<int> unitsInLayer; //number of units in each layer
	int nLayers;
	int batchSize;
	int batchesPerEpoch;
	int inputDimension;
	int outputDimension;
	vector< Mat<elem_type>* > curGradient;
	vector< Mat<elem_type>* > prevGradient;
	vector< Col<elem_type>* > prevBiasGradient;
	vector< Col<elem_type>* > curBiasGradient;
	DNN(Params &);
	DNN(Params&,const char*,string);
	void initialize_weights();
	void configure_network();
	void read_nnparams(Params &nnParams);
	void compute_output(Mat<elem_type> &input);	/* Pass the Input Pattern*/
	void compute_output(Mat<elem_type>&, vector< Mat<elem_type> >&,
						vector< Col<elem_type> >&,vector< Mat<elem_type> >&);
//	void compute_deltas();
	void output_function(Mat<elem_type>&,int);
	void output_function(Mat<elem_type>&, int, Mat<elem_type>&);
	double compute_outputerror(Mat<elem_type>&); // computes the difference in original and desired outputs
								   //and MSE(mean squared error)
	double compute_outputerror(Mat<elem_type>&, Mat<elem_type>&,Mat<elem_type>&);
	void compute_firstderivative(int);
	void compute_firstderivative(int, Mat<elem_type>&, vector< Mat<elem_type> >&);
	void compute_localgradients();
	void compute_localgradients(vector< Mat<elem_type> >&, vector< Mat<elem_type> >&,
								Mat<elem_type>&, vector< Mat<elem_type> >&);
	void compute_gradients(Mat<elem_type>&);
	void compute_gradients(Mat<elem_type>&, vector< Mat<elem_type> >&, vector< Mat<elem_type> >&,
								vector< Mat<elem_type> >&, vector< Col<elem_type> >&);

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

class TrainingAlgorithm
{

    public:
    	void train();

};

class BGD: public TrainingAlgorithm
{
	public:
		double bgd(DNN&,Mat<elem_type>&, Mat<elem_type>&,float,double,double,int);
		void train(DNN&, Mat<elem_type>&, Mat<elem_type>&, float, int, const char*);
};

class CGD: public TrainingAlgorithm
{
	private:
		static vector< Mat<elem_type>* > conjGrad; //conjugate gradient for weights
		static vector< Col<elem_type>* > conjBiasGrad; //conjugate gradient for bias
		double beta;
		double rho;
		double sigma;
	public:
		static double eta;
		static Mat<elem_type> *input;
		static Mat<elem_type> *output;
		static DNN *nn;
//		CGD(Mat<elem_type>&, Mat<elem_type>&, DNN&);
		CGD();
		void initialise(Mat<elem_type>*, Mat<elem_type>*, DNN*);
		double cgd(int);
		void compute_beta();
		static double avgerror(double);
		static double avgerror_der(double);
		void adjust_weights();
		void train(int, const char*);

};

Mat<elem_type>* CGD::input;
Mat<elem_type>* CGD::output;
vector< Mat<elem_type>* > CGD::conjGrad;
vector< Col<elem_type>* > CGD::conjBiasGrad;
DNN* CGD::nn;
double CGD::eta;

//class Train
//{ //This class is a wrapper for training the neural network model
//  //on given data using a given trainining algorithm
//
//    public:
//	void train_nn(Mat<elem_type> &input,Mat<elem_type> &output,
//            DNN &nn,TrainingAlgorithm &trainAlgo,float momentum,int nEpochs,
//            const char *weightsFname);
//};

#endif /* DNN_H_ */
