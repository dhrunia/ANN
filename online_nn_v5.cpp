/*************************************************************************
This file is implementaion of neural network class which contains the 
necessary modules to train a neural network model.

Author: S.P. Kishore
Affiliation: LTRC, IIIT Hyd.
Date of Last Modification: 16 May 2001

**************************************************************************/


#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<math.h>
#include<time.h>

using namespace std;


class ANN
{

	char *ParametersFileName;

	int TotalLayers;

	int *UnitsInLayer;

	char *TypeOfLayer;

	int *StartingPosition;

	int OutputLayer;
	

	int InputDimension;

	int kiran;

	int TotalUnits;

	float **Weights;

	float **DeltaWeights;

	float **DynamicLr;	// To put A Learning Rate for each weight 

	float *Bias;

	float *DeltaBias;

	float *Output;

	float *ErrorAtOutputLayer;

	float *LocalGradient;

	float _A;		// Tanh Parameter

	float _B;		// Tanh Parameter

	float _Bby2A;

	float Neta;
	
	float *LayerLr;		// To Put A seperate Learning Rate for each Layer.

	float NetaDecFactor;

	float NetaIncFactor;

	float MeanSquareError;

	float PrevMSE;

	long RandSeed;


	public:
	
	int InputLayer;

	void Read_NNParameters();

	void ConfigureNetwork();

	void Intialise_Weights();

	void Compute_Output(float*);	/* Pass the Input Pattern*/

	void Compute_Error(float*,float*);	/* Pass the Desired Pattern */

	void Compute_LocalGradients();

	void Compute_Deltas(float*); 		/* Compute Delta weights,bias */
						/* Pass the InputPattern*/

	float firstderivative(int,float);

	void Compute_DeltasWithMoments(float*);
	
	void Compute_DeltasWithDLR(float*);
	
	void Compute_DeltasWithLayerLr(float*,float =0.0);

	void Set_DLR();

	void IncrementWeights();
	
	void Undo_WeightChanges();

	void Change_Neta(float);
	
	void Change_Neta(float,int);

	float Get_Neta();

	float Get_Neta(int);

	int Get_TotalLayers();

	void StoreTheMSE();

	float Activation(float,int);

	int Get_InputDimension();

	int Get_UnitsOfOutputLayer();

	void Set_DynamicNeta(float*,float*,float*);

	void Read_Weights(char*);

	void Write_Weights(char*);

	void FreeTheSpace();


	ANN(char[]="nn_parameters");

	void Compute_ErrorOnSelectiveDim(float*,float*,int);

	void GetErrorForEachDimension(float*,float*);

	void Get_Output(float*);

	void writevalues_to_files(float*,float*);

};

	int ANN::Get_InputDimension()
	{
		return InputDimension;
	}

	int ANN::Get_UnitsOfOutputLayer()
	{
		return UnitsInLayer[OutputLayer];

	}

	int ANN::Get_TotalLayers()
	{
		return TotalLayers;
	}

	float ANN::Get_Neta(int LayerNo)
	{

		return LayerLr[LayerNo];
	}

	float ANN::Get_Neta()
	{
		return Neta;
	}

	void ANN::Change_Neta(float x)
	{
		Neta =x ;
	}

	void ANN::Change_Neta(float x,int LayerNo)
	{
		LayerLr[LayerNo]=x;
	}

	void ANN::StoreTheMSE()
	{
		PrevMSE = MeanSquareError;
	}

	void ANN::Intialise_Weights()
	{

		int UnitNo;

		int SP_thislayer;
		
		int SP_prevlayer;

		int LayerNo;

		int UnitsInPrevLayer;

		float x;

		float maxweight;

		LayerNo=0; 
		
		SP_thislayer = StartingPosition[LayerNo];

		if(time(NULL)==RandSeed)
		{
			system("sleep 1");
		}

		RandSeed=time(NULL);

		srand(RandSeed);
		
		maxweight=(float)3/sqrt((double)InputDimension);
		//REF : Hassoun pg : 211

		/* The reason for commenting srand()in the for loop is 
			that the "time" function returns the same old value
			to the srand() untill you restart the execution of 
			your program. This is because your recalling time 
			of TIME(NULL) will be called within a second, and
			this function returns the time in seconds.

			When I checked it, it behaved
			liked that. For any miscellaneous behaviour the
			author is not responsible. Kindly check it by yourself
		*/
		
		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			//srand(time(NULL));
			//cout<<rand()<<endl;
			for(int k=0;k<InputDimension;k++)
			{
			  x=(2*maxweight*(double)rand()/RAND_MAX)- maxweight;
				
				Weights[UnitNo + SP_thislayer][k]=x;
				
				//DynamicLr[UnitNo + SP_thislayer][k]=0.01*(double)rand()/RAND_MAX;
			
				//cout<<"Input N = "<<DynamicLr[UnitNo + SP_thislayer][k]<<endl;	
				//cout<<x<<endl;
			}
 
                       x=(2*maxweight*(double)rand()/RAND_MAX)- maxweight;
				//cout<<x<<endl;

                       Bias[UnitNo + SP_thislayer]=x;

		}

		for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		{
			SP_thislayer = StartingPosition[LayerNo];	
			
			UnitsInPrevLayer=UnitsInLayer[LayerNo-1];
                       	
			maxweight=(float)3/sqrt((double)UnitsInPrevLayer);
			
			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                	{
                        	//srand(time(NULL));
				//cout<<rand()<<endl;

                        	for(int k=0;k<UnitsInPrevLayer;k++)
                        	{
                          		x=(2*maxweight*(double)rand()/RAND_MAX)- maxweight;
                                	
					Weights[UnitNo + SP_thislayer][k]=x;

					/*if(LayerNo==TotalLayers-1)
					{
						//DynamicLr[UnitNo + SP_thislayer][k]=0.0001*(double)rand()/RAND_MAX;
					//cout<<"Out N = "<<DynamicLr[UnitNo + SP_thislayer][k]<<endl;
					}
					else
					{
						//DynamicLr[UnitNo + SP_thislayer][k]=0.01*(double)rand()/RAND_MAX;
						cout<<"N = "<<DynamicLr[UnitNo + SP_thislayer][k]<<endl;
					}*/
					
					//cout<<x<<endl;
                        	}
				
				x=(2*maxweight*(double)rand()/RAND_MAX)- maxweight;
				
				//cout<<x<<endl;

				Bias[UnitNo + SP_thislayer]=x;
                	}
		}

	}	
	
			

	float ANN::Activation(float Sum,int LayerNo)
	{

		if('L'==TypeOfLayer[LayerNo] || 'l'==TypeOfLayer[LayerNo])
		{
			return Sum;
		}
		else
		{
			if('N'==TypeOfLayer[LayerNo] || 'n'==TypeOfLayer[LayerNo])
			{
				return _A * tanh((double)(_B*Sum));

				//return _A * _B * 1.0/(cosh((double)(_B*Sum))* cosh((double)(_B*Sum)));
			}
			else
			{
				if('S'==TypeOfLayer[LayerNo] || 's'==TypeOfLayer[LayerNo])
				{
				
					return 1/(1+_A*exp(-(double)(_B*Sum)));

				}

				else
				{

					cout<<" Such a Activation Value is not implemented"<<endl;
					exit(1);
				}
			}
		}
	}	
			


	ANN::ANN(char *FileName)
	{

		ParametersFileName=FileName;
	
		// 21/12/2007: for default Input settings
		InputLayer=-1;
		
		OutputLayer=0;

         	InputDimension=0;

         	kiran=100;

         	TotalUnits=0;

         	_A=1.716;

		_B=2.0/3.0;

		_Bby2A=_B/(2*_A); //0.1943 Tanh Parameters

         	NetaDecFactor=0.5;

         	NetaIncFactor=1.1;

         	MeanSquareError=0.0;

         	PrevMSE=0.0;

		RandSeed=0;

	}

	void ANN::ConfigureNetwork()
	{

		/* Dynamic allocation is done for Weights, Deltaweights,etc.
	       	   For Local Gradient, Output, ErrorAtOutputLayer,
			Bias memory allocation is done here.
		   It Establishes the necessary Connections among neurons.
		*/
		Output = new float[TotalUnits];

		Weights = new float*[TotalUnits];

		DeltaWeights = new float*[TotalUnits];

		//DynamicLr = new float*[TotalUnits];

		LocalGradient = new float[TotalUnits];

		if(0==Weights || 0==Output || 0==DeltaWeights || 0==LocalGradient)
		{
			cerr<<"Not enough Memory "<<endl;
			exit(1);
		}

		for(int UnitNo=0;UnitNo<UnitsInLayer[0];UnitNo++)
		{
			Weights[UnitNo] = new float[InputDimension];

			DeltaWeights[UnitNo] = new float[InputDimension];
			
			//DynamicLr[UnitNo] = new float[InputDimension];
			
			for(int k=0;k<InputDimension;k++)
			{
				DeltaWeights[UnitNo][k]=0.0;
			}
		}
		

		for(int LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		{
			int SP=StartingPosition[LayerNo];
		
			for(int UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
			{	
				Weights[SP+UnitNo]=new float[UnitsInLayer[LayerNo-1]];
		
				DeltaWeights[SP+UnitNo]=new float[UnitsInLayer[LayerNo-1]];
				
				//DynamicLr[SP+UnitNo]=new float[UnitsInLayer[LayerNo-1]];
			
				for(int k=0; k < UnitsInLayer[LayerNo-1]; k++)
                        	{
                                	DeltaWeights[SP+UnitNo][k]=0.0;
                        	}

			}
		}

		ErrorAtOutputLayer = new float[UnitsInLayer[OutputLayer]];

		Bias = new float[TotalUnits];

		DeltaBias = new float[TotalUnits];

		for(int UnitNo=0;UnitNo<TotalUnits;UnitNo++)
		{
			Bias[UnitNo]=0.0;

			DeltaBias[UnitNo]=0.0;
		}
}

	void ANN::Read_NNParameters()
	{

		/* Read the Parameters from the file.
		  As and when required dynamic allocation is done to read 
		  parameters further
		*/

		ifstream fp;

		fp.open(ParametersFileName,ios::in);

		if(!fp)
		{
			cerr<<ParametersFileName<<" Couldnot Found"<<endl;
			exit(1);
		}

		fp>>TotalLayers;
		
		//fp>>InputLayer;

		fp>>OutputLayer;

		fp>>InputDimension;

		UnitsInLayer = new int[TotalLayers];

		TypeOfLayer = new char[TotalLayers];

		StartingPosition = new int[TotalLayers];

		LayerLr = new float[TotalLayers];

		TotalUnits=0;

		for(int LayerNo=0;LayerNo<TotalLayers;LayerNo++)
		{
			fp>>UnitsInLayer[LayerNo];

			fp>>TypeOfLayer[LayerNo];

			TotalUnits = TotalUnits + UnitsInLayer[LayerNo];

			if(0==LayerNo)
			{
				StartingPosition[LayerNo]=0;
			}
			else
			{
				StartingPosition[LayerNo]=StartingPosition[LayerNo-1]
							  + UnitsInLayer[LayerNo-1];
			}

			if(LayerNo==TotalLayers-1)
			{
				LayerLr[LayerNo]=1.0/(25*TotalUnits); // OutLayer Low Learning Rate

				//cout<<"Layer "<<LayerNo<<"LR is "<<LayerLr[LayerNo]<<endl;
			}
			else
			{
				if(0==LayerNo)
				{
					LayerLr[LayerNo]=1.0/(25* InputDimension);
				//cout<<"Layer "<<LayerNo<<"LR is "<<LayerLr[LayerNo]<<endl;
				}
				else
				{
					LayerLr[LayerNo]=1.0/(25* UnitsInLayer[LayerNo-1]);
				//cout<<"Layer "<<LayerNo<<"LR is "<<LayerLr[LayerNo]<<endl;
				}
			}
			// There is No sanctity about this , but Some Intialisation.......
		}

		fp>>Neta;

		//cout<<"Total Layers are "<<TotalLayers<<endl;

		//cout<<"Output Layer is "<<OutputLayer<<endl;

		//cout<<"Input Dimension is "<<InputDimension<<endl;

		//cout<<" Total Units are "<<TotalUnits<<endl;

		for(int i=0;i<TotalLayers;i++)
		{
			//cout<<"Units in "<<i<<" Layer are "<<UnitsInLayer[i];
			//cout<<" Activation Type is "<<TypeOfLayer[i]<<endl;
			//cout<<"Starting Position is "<<StartingPosition[i]<<endl;
		}
			
		//cout<<"NETA value is "<<Neta<<endl;

		fp.close();
	}


	void ANN::Compute_Output(float *InputPattern)
	{

		/* The Dimension of the InputPattern should equal to InputDimesnion

			specified in "nn_parameters" file/user given file.

			A check is not put here, because in the application this module 
			
			has to be computed may be thousands of time may be in lakhs.

			If this condition is not taken care unpredictable behaviour or 
	
			dumping core is possible. So Please Take care of this condition.
		*/

		float Sum=0.0;
		//int InputLayer = 1;

		int SP_thislayer,SP_prevlayer,LayerNo,UnitNo=0;

		// 21/12/07: changed this function so that the input can be given to the input or any hidden layer
		//OLD: LayerNo=0;  
		LayerNo=InputLayer+1;  /* Intialising to the hidden Layer Number */
	
		SP_thislayer=StartingPosition[LayerNo]; /* This will be 0 */

		int IDim = -1;

		for(UnitNo=0; UnitNo<UnitsInLayer[LayerNo]; UnitNo++)
		{
			Sum = Bias[UnitNo + SP_thislayer];

			// 21/12/07: added the 4 lines below to consider the input layer.
			if(InputLayer == -1)
				IDim = InputDimension;
			else 
				IDim = UnitsInLayer[InputLayer];

			for(int k=0;k<IDim;k++)
			{
				Sum = Sum+InputPattern[k] * Weights[UnitNo+SP_thislayer][k];
				//cout<<"INput is "<<InputPattern[k]<<endl;
			}
		
			Output[UnitNo+SP_thislayer]=Activation(Sum,LayerNo);

	
			//cout<<"OUTPUT= "<<Output[UnitNo+SP_thislayer]<<"SUM = "<<Sum<<endl;
		}

		//OLD: for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		for(LayerNo=InputLayer+2; LayerNo<TotalLayers; LayerNo++)
		{
			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
			{

				SP_thislayer=StartingPosition[LayerNo];

				SP_prevlayer=StartingPosition[LayerNo-1];

				Sum = Bias[UnitNo + SP_thislayer];

				for(int k=0;k<UnitsInLayer[LayerNo-1];k++)
				{
					Sum=Sum + Output[k + SP_prevlayer] *  Weights[UnitNo + SP_thislayer][k];
				}

				Output[UnitNo+SP_thislayer]=Activation(Sum,LayerNo);
				//cout<<"OUTPUT= "<<Output[UnitNo+SP_thislayer]<<"SUM = "<<Sum<<endl;
			}
		}

	}
					
			
	void ANN::Compute_Error(float *DesiredPattern,float *error)
	{
		/* The error at the Output Layer is calculated */
	
		*error=0;

		float NormDen=0; /* Normalizing Denominator */

		int SP=StartingPosition[OutputLayer];

		for(int UnitNo=0;UnitNo<UnitsInLayer[OutputLayer];UnitNo++)
		{
			ErrorAtOutputLayer[UnitNo] =	DesiredPattern[UnitNo]
						- Output[UnitNo+SP];
			//cout<<"Error At "<<UnitNo<<"is "<<ErrorAtOutputLayer[UnitNo]<<endl;
			
			*error = *error + (ErrorAtOutputLayer[UnitNo])*(ErrorAtOutputLayer[UnitNo]);

			NormDen = NormDen + (DesiredPattern[UnitNo])*(DesiredPattern[UnitNo]);

		}

		*error = *error/ NormDen;

		MeanSquareError = *error;

		//cout<<"MeanSquareError = "<<*error<<endl;
	}

	void ANN::Compute_LocalGradients()
	{

		int SP_thislayer;
		
		int SP_nextlayer;

		int UnitNo,LayerNo;

		int k;

		float LocalOutput;

		float Sum=0.0;

		SP_thislayer = StartingPosition[TotalLayers - 1];
			
					/* Last Layer Starting Position */

		for(UnitNo=0;UnitNo < UnitsInLayer[TotalLayers -1 ];UnitNo++)
		{
			LocalOutput = Output[SP_thislayer + UnitNo];
			
			//cout<<"Local Output of "<<UnitNo<<" is "<<LocalOutput<<endl;

			LocalGradient[SP_thislayer + UnitNo]
			
				= ErrorAtOutputLayer[UnitNo]
				
				  * firstderivative(TotalLayers-1,LocalOutput);
		}

			//cout<<"LOCAL Gradients = "<<LocalGradient[SP_thislayer + UnitNo]<<endl;

		for(LayerNo=TotalLayers - 2; LayerNo>=0; LayerNo--)
		{

			SP_thislayer = StartingPosition[LayerNo];

			SP_nextlayer = StartingPosition[LayerNo+1];

			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
			{
				Sum=0.0;

				for(k=0;k<UnitsInLayer[LayerNo+1];k++)
				{
					Sum=Sum+LocalGradient[SP_nextlayer + k]
						
					    *Weights[SP_nextlayer + k][UnitNo];
				}

				LocalOutput = Output[SP_thislayer + UnitNo];

				LocalGradient[SP_thislayer + UnitNo]
				
					=Sum * firstderivative(LayerNo,LocalOutput);

				//cout<<"LOCAL Gradients = "<<LocalGradient[SP_thislayer + UnitNo]<<endl;
			}
		}

		/* If j th neuron is in the output layer 

			grad(j) = e(j) * o'(j) 

		  else

			grad(j) = sum[ grad(k) * W(k,j) ] * o'(j)

		   k varies from 1 to max units in Upper layer.

			o'(j) is the first derivative of the output function.

			Its value become one if the neuron is linear.
		*/		
	
	}

	float ANN::firstderivative(int LayerNo,float LocalOutput)
	{

		float derivative =1;

		if('L'==TypeOfLayer[LayerNo] || 'l'==TypeOfLayer[LayerNo])
		{

			derivative = 1.0; 
		}
		else
		{

			if('N'==TypeOfLayer[LayerNo] || 'n'==TypeOfLayer[LayerNo])
                	{

                        	derivative = _Bby2A*(_A - LocalOutput) * (_A + LocalOutput);
                	}
			else
			{
				if('S'==TypeOfLayer[LayerNo] || 's'==TypeOfLayer[LayerNo])
                		{
					derivative = _B * LocalOutput * (1 - LocalOutput);
				}
			}
			
		}

		return derivative;
	}

	void ANN::Compute_Deltas(float *InputPattern)
	{
		int SP_thislayer;

		int SP_prevlayer;

		int UnitNo,LayerNo,k;

		LayerNo=0;

		SP_thislayer = StartingPosition[LayerNo];

		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(k=0;k<InputDimension;k++)
			{
				DeltaWeights[SP_thislayer + UnitNo][k]

				= Neta * InputPattern[k] * LocalGradient[SP_thislayer + UnitNo];
			}

			DeltaBias[SP_thislayer + UnitNo] = Neta * LocalGradient[SP_thislayer + UnitNo];

		}

		for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		{
			SP_thislayer = StartingPosition[LayerNo];

			SP_prevlayer = StartingPosition[LayerNo-1];
			
			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                	{
				for(k=0;k<UnitsInLayer[LayerNo-1];k++)
				{
					DeltaWeights[SP_thislayer + UnitNo][k]
					
					= Neta * Output[SP_prevlayer + k]

						* LocalGradient[SP_thislayer + UnitNo];
				}

			DeltaBias[SP_thislayer + UnitNo] = Neta * LocalGradient[SP_thislayer + UnitNo];

			}
		}

		
	}

	void ANN::IncrementWeights()
	{
		int LayerNo,UnitNo,k;

		int SP;

		LayerNo=0;

		SP = StartingPosition[LayerNo];

		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(k=0;k<InputDimension;k++)
			{
				Weights[SP+UnitNo][k] = Weights[SP+UnitNo][k] + DeltaWeights[SP+UnitNo][k];
				//cout<<"W = "<<Weights[SP+UnitNo][k]<<endl;
			
			}
			
			Bias[SP+UnitNo] = Bias[SP+UnitNo] +  DeltaBias[SP+UnitNo];
		}

		for(LayerNo=1;LayerNo < TotalLayers; LayerNo++)
		{
			SP = StartingPosition[LayerNo];

			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
			{
				for(k=0;k<UnitsInLayer[LayerNo-1];k++)
				{
                                	Weights[SP+UnitNo][k] = Weights[SP+UnitNo][k] + DeltaWeights[SP+UnitNo][k];
					//cout<<"W = "<<Weights[SP+UnitNo][k]<<endl;
				}
				
				Bias[SP+UnitNo] = Bias[SP+UnitNo] +  DeltaBias[SP+UnitNo];
			}
			
		}

	}
			

	void ANN::Undo_WeightChanges()
	{
		/* This module removes the delta weights from the weights. 
		   This may be useful in implementing Dynamic neta */
	
		int LayerNo,UnitNo,k;

		int SP;

		LayerNo=0;

		SP = StartingPosition[LayerNo];

		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(k=0;k<InputDimension;k++)
			{
				Weights[SP+UnitNo][k] = Weights[SP+UnitNo][k] - DeltaWeights[SP+UnitNo][k];
			
			}
			
			Bias[SP+UnitNo] = Bias[SP+UnitNo] -  DeltaBias[SP+UnitNo];
		}

		for(LayerNo=1;LayerNo < TotalLayers; LayerNo++)
		{
			SP = StartingPosition[LayerNo];

			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
			{
				for(k=0;k<UnitsInLayer[LayerNo-1];k++)
				{
                                	Weights[SP+UnitNo][k] = Weights[SP+UnitNo][k] - DeltaWeights[SP+UnitNo][k];
				}
			
				Bias[SP+UnitNo] = Bias[SP+UnitNo] -  DeltaBias[SP+UnitNo];
			}
			
		}

	}

	void ANN::Set_DynamicNeta(float *InputPattern, float *OutputPattern,float *error)
	{
		float MaxFactor=1.1;
		
		StoreTheMSE();
		
		Compute_Output(InputPattern);

		Compute_Error(OutputPattern,error);

		if(PrevMSE * MaxFactor > MeanSquareError)
		{
			Neta = Neta * NetaIncFactor;
		}
		
		while(MeanSquareError > PrevMSE * MaxFactor)
		{
			Undo_WeightChanges();
	
			Neta = NetaDecFactor * Neta;

			//cout<<"Neta = "<<Neta<<endl;

	                Compute_Output(InputPattern);

        	        Compute_Error(OutputPattern,error);

			Compute_LocalGradients();

			Compute_Deltas(InputPattern);

			IncrementWeights();

		        Compute_Output(InputPattern);

                        Compute_Error(OutputPattern,error);
		}

		//Neta = Neta * NetaIncFactor;
		//cout<<"Neta = "<<Neta<<endl;
	}

	void ANN::Write_Weights(char *FileName)
	{

		int UnitNo;

		int SP_thislayer;
		
		int SP_prevlayer;

		int LayerNo;

		int UnitsInPrevLayer;

		float x;

		ofstream fp;

		fp.open(FileName,ios::out);

		if(!fp)
		{
			cerr<<"Cannot Write into "<<FileName<<endl;
			exit(1);
		}
			
		LayerNo=0; 
		
		SP_thislayer = StartingPosition[LayerNo];

		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(int k=0;k<InputDimension;k++)
			{
				
				fp<<Weights[UnitNo + SP_thislayer][k]<<endl;
			}
 

                       fp<<Bias[UnitNo + SP_thislayer]<<endl;

		}

		for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		{
			SP_thislayer = StartingPosition[LayerNo];	
			
			UnitsInPrevLayer=UnitsInLayer[LayerNo-1];
                       	
			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                	{

                        	for(int k=0;k<UnitsInPrevLayer;k++)
                        	{
                                	fp<<Weights[UnitNo + SP_thislayer][k]<<endl;
                        	}
				

				fp<<Bias[UnitNo + SP_thislayer]<<endl;
                	}
		}

		fp.close();

	}	
	
	void ANN::Read_Weights(char *FileName)
	{

		int UnitNo;

		int SP_thislayer;
		
		int SP_prevlayer;

		int LayerNo;

		int UnitsInPrevLayer;

		float x;

		ifstream fp;

		fp.open(FileName,ios::in);

		if(!fp)
		{
			cerr<<"Cannot Read From "<<FileName<<endl;
			exit(1);
		}
			
		LayerNo=0; 
		
		SP_thislayer = StartingPosition[LayerNo];

		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(int k=0;k<InputDimension;k++)
			{
				
				fp>>Weights[UnitNo + SP_thislayer][k];
			}
 

                       fp>>Bias[UnitNo + SP_thislayer];

		}

		for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
		{
			SP_thislayer = StartingPosition[LayerNo];	
			
			UnitsInPrevLayer=UnitsInLayer[LayerNo-1];
                       	
			for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                	{

                        	for(int k=0;k<UnitsInPrevLayer;k++)
                        	{
                                	fp>>Weights[UnitNo + SP_thislayer][k];
                        	}
				

				fp>>Bias[UnitNo + SP_thislayer];
                	}
		}

		fp.close();

	}

	void ANN::FreeTheSpace()
	{
		delete [] Output;
		
		//delete []* Weights;
		
		//delete []* DeltaWeights;

		for(int i=0;i<TotalUnits;i++)
		{
			delete [] Weights[i];

			delete [] DeltaWeights[i];
		}

		delete [] Weights;

		delete [] DeltaWeights;
		
		delete [] LocalGradient;
		
		delete [] ErrorAtOutputLayer;
		
		delete [] Bias;
		
		delete [] DeltaBias;
		
		delete [] UnitsInLayer;
		
		delete [] TypeOfLayer;
		
		delete [] StartingPosition;
	}


      	void ANN::Compute_DeltasWithMoments(float *InputPattern)
        {
                int SP_thislayer;

                int SP_prevlayer;

                int UnitNo,LayerNo,k;

		float Alpha=0.5;

                LayerNo=0;

                SP_thislayer = StartingPosition[LayerNo];

                for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                {
                        for(k=0;k<InputDimension;k++)
                        {
                                DeltaWeights[SP_thislayer + UnitNo][k]

                                = Neta * InputPattern[k] * LocalGradient[SP_thislayer + UnitNo]+ Alpha * DeltaWeights[SP_thislayer + UnitNo][k];
                        }

                        DeltaBias[SP_thislayer + UnitNo] = Neta * LocalGradient[SP_thislayer + UnitNo]+ Alpha * DeltaBias[SP_thislayer + UnitNo];

                }

                for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
                {
                        SP_thislayer = StartingPosition[LayerNo];

                        SP_prevlayer = StartingPosition[LayerNo-1];

                        for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                        {
                                for(k=0;k<UnitsInLayer[LayerNo-1];k++)
                                {
                                        DeltaWeights[SP_thislayer + UnitNo][k]

                                        = Neta * Output[SP_prevlayer + k]

                                                * LocalGradient[SP_thislayer + UnitNo] + Alpha * DeltaWeights[SP_thislayer + UnitNo][k];
                                }

                        DeltaBias[SP_thislayer + UnitNo] = Neta * LocalGradient[SP_thislayer + UnitNo] + Alpha * DeltaBias[SP_thislayer + UnitNo];

                        }
                }


        }


	void ANN::Compute_DeltasWithLayerLr(float *InputPattern,float AlphaPassed)
        {
                int SP_thislayer;

                int SP_prevlayer;

                int UnitNo,LayerNo,k;

                LayerNo=0;

                SP_thislayer = StartingPosition[LayerNo];

                for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                {

                        for(k=0;k<InputDimension;k++)
                        {
                                DeltaWeights[SP_thislayer + UnitNo][k]

                                = AlphaPassed * DeltaWeights[SP_thislayer + UnitNo][k] + LayerLr[LayerNo] * InputPattern[k] * LocalGradient[SP_thislayer + UnitNo];
                        }

                        DeltaBias[SP_thislayer + UnitNo] = AlphaPassed * DeltaBias[SP_thislayer + UnitNo] + LayerLr[LayerNo]  * LocalGradient[SP_thislayer + UnitNo];

                }


                for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
                {
                        SP_thislayer = StartingPosition[LayerNo];

                        SP_prevlayer = StartingPosition[LayerNo-1];

                        for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
                        {
                                for(k=0;k<UnitsInLayer[LayerNo-1];k++)
                                {
                                        DeltaWeights[SP_thislayer + UnitNo][k]

                                        = AlphaPassed * DeltaWeights[SP_thislayer + UnitNo][k] + LayerLr[LayerNo] * Output[SP_prevlayer + k] * LocalGradient[SP_thislayer + UnitNo];
                                }

                        DeltaBias[SP_thislayer + UnitNo] = AlphaPassed * DeltaBias[SP_thislayer + UnitNo] + LayerLr[LayerNo] * LocalGradient[SP_thislayer + UnitNo];

                        }
                }

        }

        void ANN::Compute_ErrorOnSelectiveDim(float *DesiredPattern,float *error,int PlacesUpto)
        {
                /* The error at the Output Layer is calculated */

                *error=0;

                float NormDen=0; /* Normalizing Denominator */

		float *ErrorValues,TV;

                int SP=StartingPosition[OutputLayer];

		ErrorValues = new float[UnitsInLayer[OutputLayer]];

                for(int UnitNo=0;UnitNo<UnitsInLayer[OutputLayer];UnitNo++)
                {
                        ErrorAtOutputLayer[UnitNo] =    DesiredPattern[UnitNo]
                                                - Output[UnitNo+SP];

                        //*error = *error + (ErrorAtOutputLayer[UnitNo])*(ErrorAtOutputLayer[UnitNo]);

			ErrorValues[UnitNo] = (ErrorAtOutputLayer[UnitNo])*(ErrorAtOutputLayer[UnitNo]);	
                        NormDen = NormDen + (DesiredPattern[UnitNo])*(DesiredPattern[UnitNo]);

                }

		/* SORT THE ERRORS IN DECREASING ORDER*/

		for(int i=0;i<UnitsInLayer[OutputLayer];i++)
                {
			for(int j=UnitsInLayer[OutputLayer]-1;j>i;j--)
			{
				if(ErrorValues[j]>ErrorValues[j-1])
				{
					TV=ErrorValues[j];

					ErrorValues[j]=ErrorValues[j-1];
	
					ErrorValues[j-1]=TV;
				}
			}
		}

		*error=0;

		/* SUM THE ERRORS LEAVING THE SPECIFIED NUMBER OF
		   DIMENSIONS*/
                
		for(int i=PlacesUpto;i<UnitsInLayer[OutputLayer];i++)
                {
                        *error = *error + ErrorValues[i];
                }

                *error = *error/NormDen;
		
		/*for(int i=0;i<UnitsInLayer[OutputLayer];i++)
                {
			cout<<ErrorValues[i]<<" ";
		}
		cout<<endl;
		char ch;
		cin>>ch;*/

                MeanSquareError = *error;

		delete [] ErrorValues;

                //cout<<"MeanSquareError = "<<*error<<endl;
        }

        void ANN::GetErrorForEachDimension(float *DesiredPattern,float *Error)
        {
                /* The error at the Output Layer is calculated */
		 /*1: The arugument "Error" passed here is an array
			where the error for each dimension
			is stored and returned
		   2: This module should be called after 
			Compute_Output(..);
		   3: The consequences of violating the two suggestions
			are unexplained (possibly core dump for not intailization or for accessing the nonallocated position of the array)
		*/

                int SP=StartingPosition[OutputLayer];

                for(int UnitNo=0;UnitNo<UnitsInLayer[OutputLayer];UnitNo++)
                {
                        Error[UnitNo] =    DesiredPattern[UnitNo]
                                                - Output[UnitNo+SP];
                }

        }

	void ANN::Get_Output(float *ActualOutputPattern)
        {
                /* This module is intended to return the 
		   output obtained at the output layer.
		   The output layer is as specfied in nn_parameters file.
		   If you call Compute_Output() and then
		   this routine you will the corresponding 
		   output for that input. If that is not called 
		   this will give whatever value is stored in the array
		   and sometimes junk values are also possible */

                int SP=StartingPosition[OutputLayer];

                for(int UnitNo=0;UnitNo<UnitsInLayer[OutputLayer];UnitNo++)
                {
                        ActualOutputPattern[UnitNo] =  Output[UnitNo+SP];

		}
	} 

void ANN::writevalues_to_files(float *input,float *output)
{
	int UnitNo;

	int SP_thislayer;

	int SP_prevlayer;

	int LayerNo;

	int UnitsInPrevLayer;

	float x;
	const char *inpdatafname = "input_frames";
	const char *outdatafname = "output_frames";
	const char *outfname="outputs";
	const char *errorfname="errors";
	const char *lgfname="local_gradients";
	const char *wtsfname="weights.debug";
	const char *deltawtsfname="delta_weights";
	ofstream outfh;
	ofstream errorfh;
	ofstream lgfh;
	ofstream wtsfh;
	ofstream deltawtsfh;
	ofstream outdatafh;
	ofstream inpdatafh;
	outfh.open(outfname,ios::app);
	errorfh.open(errorfname,ios::app);
	lgfh.open(lgfname,ios::app);
	wtsfh.open(wtsfname,ios::app);
	deltawtsfh.open(deltawtsfname,ios::app);
	outdatafh.open(outdatafname,ios::app);
	inpdatafh.open(inpdatafname,ios::app);
	ostringstream inpbuff(ostringstream::ate);
	ostringstream outbuff(ostringstream::ate);
	for(int i=0;i<InputDimension;i++)
	{
		inpbuff<<input[i]<<" ";
	}
	for(int i=0;i<UnitsInLayer[TotalLayers-1];i++)
	{
		outbuff<<output[i]<<" ";
	}
	inpdatafh<<inpbuff.str()<<endl;
	outdatafh<<outbuff.str()<<endl;
	if(!outfh.is_open() || !errorfh.is_open()|| !lgfh.is_open()|| !wtsfh.is_open()|| !deltawtsfh.is_open()
			|| !outdatafh.is_open()|| !inpdatafh.is_open())
	{
		cerr<<"unable to open file in writevalues_to_files() "<<endl;
		exit(1);
	}
	LayerNo=0;
	SP_thislayer = StartingPosition[LayerNo];
	for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
	{
		for(int k=0;k<InputDimension;k++)
		{
			wtsfh<<Weights[UnitNo + SP_thislayer][k]<<" ";
			deltawtsfh<<Weights[UnitNo + SP_thislayer][k]<<" ";
		}
		outfh<<Output[SP_thislayer+UnitNo]<<" ";
		lgfh<<LocalGradient[SP_thislayer+UnitNo]<<" ";
		wtsfh<<Bias[UnitNo + SP_thislayer]<<" ";
		deltawtsfh<<DeltaBias[UnitNo + SP_thislayer]<<" ";
	}
	for(LayerNo=1;LayerNo<TotalLayers;LayerNo++)
	{
		SP_thislayer = StartingPosition[LayerNo];
		for(UnitNo=0;UnitNo<UnitsInLayer[LayerNo];UnitNo++)
		{
			for(int k=0;k<InputDimension;k++)
			{
				wtsfh<<Weights[UnitNo + SP_thislayer][k]<<" ";
				deltawtsfh<<Weights[UnitNo + SP_thislayer][k]<<" ";
			}
			outfh<<Output[SP_thislayer+UnitNo]<<" ";
			lgfh<<LocalGradient[SP_thislayer+UnitNo]<<" ";
			wtsfh<<Bias[UnitNo + SP_thislayer]<<" ";
			deltawtsfh<<DeltaBias[UnitNo + SP_thislayer]<<" ";
			if(LayerNo == TotalLayers-1)
				errorfh<<ErrorAtOutputLayer[SP_thislayer+UnitNo]<<" ";
		}
	}
	wtsfh<<endl;
	outfh<<endl;
	errorfh<<endl;
	lgfh<<endl;
	deltawtsfh<<endl;

	wtsfh.close();
	outfh.close();
	errorfh.close();
	lgfh.close();
	deltawtsfh.close();
	inpdatafh.close();
	outdatafh.close();
}

 
