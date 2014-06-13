/*************************************************************************
This Program is the implementation of Backpropogation learning algorithm
to train a neural network model.

Author: S.P. Kishore
Affiliation: LTRC, IIIT Hyd. 
Date of Last Modification: 16 May 2001

**************************************************************************/

#include<iostream>
#include<fstream>
#include<sstream>
#include<stdlib.h>
#include<string.h>
#include<algorithm>
#include<ctime>

using namespace std;
#include"nn_v5.cpp"

#define _MaxChangeinError 0.0005  /* .05 % percentage */

#define _BatchSize 100

#define _BatchesPerIter 100

float _MomentumFactor = 0.3;

#define _validEpochs 20

int _Validation = 0;

void TrainANN_WithGivenFrameNumbers(ANN *,float**,float**,int*,int);

void TrainANN_Batch(ANN *, float**, float**,int*,int,int,int);


/* Passing the arguments */
/* 

	1: Network Configuration file for Format see the end.

	2: Input Pattern File name

	3: Output Pattern File name

	4: Weight File name 

	5: No Of Epochs

*/
	
 	char *ConfigFileName;

        char *InputFileName;

        char *OutputFileName;

        char *WeightFileName;

        int NoOfPatterns;

        int InputDimension;

        int OutputDimension;

        int Epochs;



int main(int argc,char *argv[])
{
	if(argc<8)
	{

		cout<<endl;
		
		cout<<"Few  Parameters Passed  ... Aborting main program "<<endl<<endl;

		cout<<"Invoke online help........"<<endl<<endl;

		cout<<"To run the program, pass the Parameters in the Displayed order"<<endl<<endl;

		cout<<"1: Network Configuration filename whose contents are ordered  in the format :"<<endl;

		cout<<"	TotalLayers Excluding InputLayer ( Layers Numbering Start From 0 )"<<endl;

		cout<<"	OutputLayer"<<endl;

		cout<<"	Input Dimension"<<endl;

		cout<<"	structure of the NN ex: 6 N 22 N 19 L .... "<<endl;

		cout<<" 	Learning Rate(default Learning rate if any adv. neta adaptaion is not used"<<endl;
	
		

        	//cout<<"2: No Of patterns"<<endl;

        	cout<<"2: Input Pattern File name"<<endl;

        	//cout<<"4: Dimension Of Input Pattern"<<endl;

        	cout<<"3: Output Pattern File name"<<endl;

        	//cout<<"6: Dimension Of Output Pattern"<<endl;

        	cout<<"4: Weight File name"<<endl;

        	cout<<"5: No Of Epochs"<<endl;

                cout<<"6: Validation Flag [1/0] "<<endl; 

                cout<<"7: Momentum Factor "<<endl; 


		cout<<"NOTE: The first line of input pattern file should indicate the number of rows and columns present in the file. Similarly the same applies to the output pattern file"<<endl;


		exit(1);
	}

	clock_t starttime = clock();

	ConfigFileName 	= argv[1];

	//cout<<"Config file name is "<<ConfigFileName<<endl;

	//NoOfPatterns 	= atoi(argv[2]);

	//cout<<"No of Patterns are "<<NoOfPatterns<<endl;

	InputFileName 	= argv[2];

	//cout<<"input file name "<<InputFileName<<endl;

	//InputDimension 	= atoi(argv[4]);

	//cout<<"InputDimension is "<<InputDimension<<endl;
	
	OutputFileName 	= argv[3];

	//cout<<"Output file name is "<<OutputFileName<<endl;
	
	//OutputDimension	= atoi(argv[6]);

	//cout<<"Output Dimension is "<<OutputDimension<<endl;	

	WeightFileName	= argv[4];

	//cout<<"Weight File name is "<<WeightFileName<<endl;

	Epochs		= atoi(argv[5]);

	//cout<<"No Of Epochs are "<<Epochs<<endl;
  
        _Validation = atoi(argv[6]);

        _MomentumFactor = atof(argv[7]);

	

	ifstream fp1,fp2;

        fp1.open(InputFileName,ios::in);

        if(!fp1)
        {
                cout<<"Error Opening bglpc.6"<<endl;

                exit(1);
        }

        fp2.open(OutputFileName,ios::in);

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

        // Cross checking the No of Patterns from the output file

        float TempNumber = 0;

        fp2>>TempNumber;

        if(TempNumber != NoOfPatterns)
        {

                cout<<"Mismatch in the number of patterns specified in the input and output file"<<endl;

                cout<<TempNumber<<" is specified in output file, while "<<NoOfPatterns<<" is specified in input file"<<endl;

                cout<<"Aborting............"<<endl;

                exit(1);
        }

	// Reading the dimension of the output pattern

        fp2>>OutputDimension;

	//cout<<"Output Dimension is "<<OutputDimension<<endl;



	float **BGInputLPCC, **BGOutputLPCC;

	int *FrameNumbers;

	BGInputLPCC = new float*[NoOfPatterns];

	FrameNumbers = new int[NoOfPatterns];
	
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


	// Store the patterns in the corresponding arrays

        for(int FrameNo=0;FrameNo<NoOfPatterns;FrameNo++)
	{
		for(int i=0;i<InputDimension;i++)
		{
			fp1>>BGInputLPCC[FrameNo][i];
		}

		for(int i=0;i<OutputDimension;i++)
		{
			fp2>>BGOutputLPCC[FrameNo][i];
		}
	}

	fp1.close();

	fp2.close();

        ANN bgann(ConfigFileName);

        bgann.Read_NNParameters();

        bgann.ConfigureNetwork();

	//cout<<"Here"<<endl;
	if(argc == 9 && atoi(argv[8]) == 1) /* If the 9th argument is 1, n/w will initialize weights from weight file */
        {
		//cout<<"using weight"<<endl;
		bgann.Read_Weights(WeightFileName);
        }
	else
	{
		//cout<<"Random Initialise"<<endl;
		bgann.Intialise_Weights();
	}
	//cout<<"weight Initialize completed"<<endl;
	//	exit(0);
	//cout<<"started Training..."<<endl;
	//cout<<endl;
	//TrainANN_WithGivenFrameNumbers(&bgann,BGInputLPCC,BGOutputLPCC,FrameNumbers,NoOfPatterns);
	TrainANN_Batch(&bgann,BGInputLPCC,BGOutputLPCC,FrameNumbers,NoOfPatterns,_BatchSize,_BatchesPerIter);

	delete [] FrameNumbers;

	for(int i=0;i<NoOfPatterns;i++)
	{
		delete [] BGInputLPCC[i];

		delete [] BGOutputLPCC[i];

	}

	delete [] BGInputLPCC;

	delete [] BGOutputLPCC;

	//cout<<"Time took: "<<(clock()-starttime)/ ((double)CLOCKS_PER_SEC*60)<< " minutes." << endl;

	
}
	
void TrainANN_WithGivenFrameNumbers(ANN *ann,float **BGInputLPCC,float **BGOutputLPCC,int *FrameNumbers,int FrameCount)
{
        float AvgError=0.0; 

	float PrevAvgError=9999;

        float PrevValError = 99999999;
        float PrevValAvgError = 99999999;
        float valAvgError = 0;

        float error;
        float valError;

        int FrameNo=0;

	float *InputPattern;

	float *OutputPattern;

        ofstream errfp;
        ofstream valfp;

        errfp.open("Err",ios::out);

        valfp.open("ValErr",ios::out);

	random_shuffle(FrameNumbers,FrameNumbers + FrameCount);

        int valSP = (FrameCount * 3)/4;

        int remSamples = FrameCount - valSP;

        char sysCommand[500];
        char tempWtFile[500];
        strcpy(tempWtFile,"tempwt.txt");

        int It = 0;
        //for(int It=0;It<Epochs;It++)
        while(1) 
        {

                if(_Validation == 0 && It >= Epochs) {
                   break; 
                } // 23/12/07: if val flag=1, does not chk for no. of iterations in old code.
                else if(_Validation == 1 && It >= Epochs) {
                   break; 
                }

                cout<<"Epoch "<<It+1<<":"<<endl;
                AvgError=0.0; 

	        //random_shuffle(FrameNumbers,FrameNumbers + FrameCount);
	        random_shuffle(FrameNumbers,FrameNumbers + valSP);

                for(FrameNo=0;FrameNo<valSP;FrameNo++)
                {
		
			InputPattern = BGInputLPCC[FrameNumbers[FrameNo]];
			OutputPattern = BGOutputLPCC[FrameNumbers[FrameNo]];

                        //speech->GetInput_OutputLPCC(FrameNumbers[FrameNo],InputPattern,OutputPattern)

                        ann->Compute_Output(InputPattern);
                        ann->Compute_Error(OutputPattern,&error);
                        //cout<<"error:"<<error<<endl;
                        ann->Compute_LocalGradients();

                        //ann->Compute_Deltas(InputPattern);
                        //ann->Compute_DeltasWithLayerLr(InputPattern,0.3);

                        ann->Compute_DeltasWithLayerLr(InputPattern,_MomentumFactor);
			ann->IncrementWeights();

                        AvgError=AvgError+ error/valSP;

                }
        cout<<"average Error:"<<AvgError<<endl;
		errfp<<AvgError<<endl;

               
                // CROSS VALIDATION

                valError = 0;

                for(FrameNo = valSP; FrameNo<FrameCount; FrameNo++)
                {
                        InputPattern = BGInputLPCC[FrameNumbers[FrameNo]];
                        OutputPattern = BGOutputLPCC[FrameNumbers[FrameNo]];
                        ann->Compute_Output(InputPattern);
                        ann->Compute_Error(OutputPattern,&error);
                        valError = valError+ error/remSamples;
                }
 
                valfp<<valError<<endl;
                valAvgError = valError + valAvgError;
                cout<<"Validation Error: "<<valError<<endl;
		/* LOOK WHETHER % CHANGE IN TRAINERROR IS BELOW THRESHOLD*/

                if(It % _validEpochs == 0 && It != 0 && _Validation) {

                   //float ChangeinError = (PrevValError - valError);
                   float ChangeinError = (PrevValAvgError - valAvgError);

                   if(ChangeinError < 0)
                   {
                      cout<<"so far training is done for "<<(It+1)<<" epochs"<<endl;
                      sprintf(sysCommand,"cp %s %s",tempWtFile,WeightFileName);
                      system(sysCommand);
                      break;

                   } else { 
                      sprintf(sysCommand,"cp %s %s",WeightFileName, tempWtFile);
                      system(sysCommand);
                   }

                   PrevValError = valError;
                   PrevValAvgError = valAvgError;
                   valAvgError = 0;
                }

		PrevAvgError = AvgError;
 
               // Writing Weights on every epoch  
                ann->Write_Weights(WeightFileName);
 
               if(It == 0) {
                   PrevValError = valError;
                   sprintf(sysCommand,"cp %s %s",WeightFileName, tempWtFile);
                   system(sysCommand);
               }

          It++;
          //cout<<endl;
        }  // For all Iterations

        errfp.close();
        valfp.close();
}


void TrainANN_Batch(ANN *ann,float **BGInputLPCC,float **BGOutputLPCC,int *FrameNumbers,int FrameCount,
					int BatchSize,int BatchesPerIter)
{
        float AvgError=0.0;

	float PrevAvgError=9999;

        float PrevValError = 99999999;
        float PrevValAvgError = 99999999;
        float valAvgError = 0;

        float error;
        float valError;

        int FrameNo=0;

	float *InputPattern;

	float *OutputPattern;

        ofstream errfp;
        ofstream valfp;

        errfp.open("Err",ios::out);

        valfp.open("ValErr",ios::out);

	random_shuffle(FrameNumbers,FrameNumbers + FrameCount);

        int valSP = (FrameCount * 3)/4;

        int remSamples = FrameCount - valSP;

        char sysCommand[500];
        char tempWtFile[500];
        strcpy(tempWtFile,"tempwt.txt");

        int It = 0;
        //for(int It=0;It<Epochs;It++)
        while(1)
        {
                if(_Validation == 0 && It >= Epochs) {
                   break;
                } // 23/12/07: if val flag=1, does not chk for no. of iterations in old code.
                else if(_Validation == 1 && It >= Epochs) {
                   break;
                }
                //cout<<"Epoch "<<It+1<<":"<<endl;

                AvgError=0.0;

	        //random_shuffle(FrameNumbers,FrameNumbers + FrameCount);
	        random_shuffle(FrameNumbers,FrameNumbers + valSP);
	        //cout<<"shuflle completed"<<endl;
	        for(int BatchNo=0;BatchNo<BatchesPerIter;BatchNo++)
	        {
	        	random_shuffle(FrameNumbers,FrameNumbers + valSP);
	        	error = ann -> Compute_BatchDeltasAndError(BGInputLPCC,BGOutputLPCC,_MomentumFactor,
	        										  BatchSize,FrameNumbers);
	        	ann->BatchIncrementWeights();
	        	//cout<<"Batch Error: "<<error/BatchSize<<endl;
	        	AvgError=AvgError+ error/(BatchSize*BatchesPerIter);
	        }
	        //cout<<"Epoch "<<It+1<<" completed"<<endl;
	        //cout<<"average Error: "<<AvgError<<endl;
		errfp<<AvgError<<endl;


                // CROSS VALIDATION

                valError = 0;
                //cout<<"calculating validation error...";
                for(FrameNo = valSP; FrameNo<FrameCount; FrameNo++)
                {
                        InputPattern = BGInputLPCC[FrameNumbers[FrameNo]];
                        OutputPattern = BGOutputLPCC[FrameNumbers[FrameNo]];
                        ann->Compute_Output(InputPattern);
                        ann->Compute_Error(OutputPattern,&error);
                        valError = valError+ error/remSamples;
                }

                valfp<<valError<<endl;
                valAvgError = valError + valAvgError;
                //cout<<"completed"<<endl;
                //cout<<"Validation Error: "<<valError<<endl;
		/* LOOK WHETHER % CHANGE IN TRAINERROR IS BELOW THRESHOLD*/

                if(It % _validEpochs == 0 && It != 0 && _Validation) {

                   //float ChangeinError = (PrevValError - valError);
                   float ChangeinError = (PrevValAvgError - valAvgError);

                   /*if(ChangeinError < 0)
                   {
                      cout<<"so far training is done for "<<(It+1)<<" epochs"<<endl;
                      sprintf(sysCommand,"cp %s %s",tempWtFile,WeightFileName);
                      system(sysCommand);
                      break;

                   } else {
                      sprintf(sysCommand,"cp %s %s",WeightFileName, tempWtFile);
                      system(sysCommand);
                   }*/
                   sprintf(sysCommand,"cp %s %s",WeightFileName, tempWtFile);
                   system(sysCommand);
                   PrevValError = valError;
                   PrevValAvgError = valAvgError;
                   valAvgError = 0;
                }

		PrevAvgError = AvgError;
				//cout<<"saving weights to "<<WeightFileName;
               // Writing Weights on every epoch
                ann->Write_Weights(WeightFileName);
                //cout<<" completed"<<endl;
               if(It == 0) {
                   PrevValError = valError;
                   sprintf(sysCommand,"cp %s %s",WeightFileName, tempWtFile);
                   system(sysCommand);
               }

          It++;
          //cout<<endl;
        }  // For all Iterations

        errfp.close();
        valfp.close();
}

