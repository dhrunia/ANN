/*************************************************************************
This Program is useful to test the neural network model trained with 
the program "JustTrain.cpp".

Author: S.P. Kishore
Affiliation: LTRC, IIIT Hyd.
Date of Last Modification: 16 May 2001

**************************************************************************/


#include<iostream>
#include<fstream>
#include<sstream>
#include<stdlib.h>
#include<string>
#include<algorithm>

using namespace std;

#include"nn_v5.cpp"

void TestANN_WithGivenFrameNumbers(ANN *,float**,float**,int*,int);

/* Passing the arguments */
/*

        1: Network Configuration file for Format see the end.

        2: Input Pattern File name

        3: Output Pattern File name

        4: Weight File name

*/
	
 	char *ConfigFileName;

        char *InputFileName;

        char *OutputFileName;

        char *WeightFileName;

        int NoOfPatterns;

        int InputDimension;

        int OutputDimension;



int main(int argc,char *argv[])
{


	if(argc<5)
        {

                cout<<endl;

                cout<<"Few  Parameters Passed  ... Aborting main program "<<endl<<endl;

                cout<<"Invoke online help........"<<endl<<endl;

                cout<<"To run the program, pass the Parameters in the Displayed order"<<endl<<endl;

                cout<<"1: Network Configuration filename whose contents are ordered  in the format :"<<endl;

                cout<<" TotalLayers Excluding InputLayer ( Layers Numbering Start From 0 )"<<endl;

                cout<<" OutputLayer"<<endl;

                cout<<" Input Dimension"<<endl;

                cout<<" structure of the NN ex: 6 N 22 N 19 L .... "<<endl;

                cout<<"         Learning Rate(default Learning rate if any adv. neta adaptaion is not used"<<endl;


                cout<<"2: Input Pattern File name"<<endl;

                cout<<"3: Output File name to store the output of the network"<<endl;

                cout<<"4: Weight File name"<<endl;
                
		cout<<"5: OPTIONAL - Input layer. If not specified uses default input layer."<<endl;

                cout<<"NOTE: The first line of input pattern file should indicate the number of rows and columns present in the file. "<<endl;


                exit(1);
        }


	
	ConfigFileName 	= argv[1];

	//cout<<"Config file name is "<<ConfigFileName<<endl;

	InputFileName 	= argv[2];

	//cout<<"input file name "<<InputFileName<<endl;

	OutputFileName 	= argv[3];

	//cout<<"Output file name is "<<OutputFileName<<endl;
	
	WeightFileName	= argv[4];

	//cout<<"Weight File name is "<<WeightFileName<<endl;


        ifstream fp1;

	ofstream fp2;

        fp1.open(InputFileName,ios::in);

        if(!fp1)
        {
                cout<<"Error Opening bglpc.6"<<endl;

                exit(1);
        }

        fp2.open(OutputFileName,ios::out);

        if(!fp2)
        {
                cout<<"Error Opening bglpc.14"<<endl;

                exit(1);
        }


        // Reading the no of patterns from the input header.

        fp1>>NoOfPatterns;

        //cout<<"No of Patterns are "<<NoOfPatterns<<endl;

	// Reading the Input Dimension

        fp1>>InputDimension;

        //cout<<"InputDimension is "<<InputDimension<<endl;


	// Intialize the network class.

	ANN bgann(ConfigFileName);

        bgann.Read_NNParameters();

        bgann.ConfigureNetwork();

        bgann.Intialise_Weights();


	OutputDimension = bgann.Get_UnitsOfOutputLayer();


	float **BGInputLPCC, **BGOutputLPCC;

	int *FrameNumbers;

	BGInputLPCC = new float*[NoOfPatterns];

	FrameNumbers = new int[NoOfPatterns];

	// 21/12/2007: This is an optional argument to provide the input at the hidden layers, else default Input is used.
	if(argc == 6)
	{
		bgann.InputLayer = atoi(argv[5]);
	}

	if(0==BGInputLPCC)
	{
		cout<<"Memory Allocation Problem BGIn....."<<endl;
		
		exit(1);
	}
		
	BGOutputLPCC = new float*[NoOfPatterns];
	
	if(0==BGOutputLPCC)
	{
		cout<<"Memory Allocation Problem BGOu...."<<endl;
		
		exit(1);
	}

	for(int FrameNo=0;FrameNo<NoOfPatterns;FrameNo++)
	{
		BGInputLPCC[FrameNo] = new float[InputDimension];
	
	        if(0==BGInputLPCC[FrameNo])
        	{
                	cout<<"Memory Allocation Problem BGIn.....FrameNo"<<endl;

                	exit(1);
        	}

		BGOutputLPCC[FrameNo] = new float[OutputDimension];

		if(0==BGOutputLPCC[FrameNo])
        	{
                	cout<<"Memory Allocation Problem BGOu....FrameNo"<<endl;

                	exit(1);
        	}
		
		FrameNumbers[FrameNo]=FrameNo;

	}


	// Read the input patterns.

	
        for(int FrameNo=0;FrameNo<NoOfPatterns;FrameNo++)
	{
		for(int i=0;i<InputDimension;i++)
		{
			fp1>>BGInputLPCC[FrameNo][i];
		}

	}


	TestANN_WithGivenFrameNumbers(&bgann,BGInputLPCC,BGOutputLPCC,FrameNumbers,NoOfPatterns);


	// Writing the output of the network.

	for(int FrameNo=0;FrameNo<NoOfPatterns;FrameNo++)
        {

                for(int i=0;i<OutputDimension;i++)
                {
                      fp2<<BGOutputLPCC[FrameNo][i]<<" ";
                }

		fp2<<endl;
        }



	fp1.close();

	fp2.close();


	delete [] FrameNumbers;

	for(int i=0;i<NoOfPatterns;i++)
	{
		delete [] BGInputLPCC[i];

		delete [] BGOutputLPCC[i];

	}

	delete [] BGInputLPCC;

	delete [] BGOutputLPCC;

	// Added on Sep 11 2001 

	bgann.FreeTheSpace();

	
}
	
void TestANN_WithGivenFrameNumbers(ANN *ann,float **BGInputLPCC,float **BGOutputLPCC,int *FrameNumbers,int FrameCount)
{
        int FrameNo=0;

	float *InputPattern;

	float *OutputPattern;

		ann->Read_Weights(WeightFileName);

                for(FrameNo=0;FrameNo<FrameCount;FrameNo++)
                {
		
			InputPattern = BGInputLPCC[FrameNumbers[FrameNo]];

			OutputPattern = BGOutputLPCC[FrameNumbers[FrameNo]];

                        //speech->GetInput_OutputLPCC(FrameNumbers[FrameNo],InputPattern,OutputPattern)

                        ann->Compute_Output(InputPattern);

			ann->Get_Output(OutputPattern);

                        //ann->Compute_Error(OutputPattern,&error);
		                

                }

}
