#include<vector>
#include "dnn_cu.h"
#include "utils.cu"
using namespace std;

Params::Params(const char *paramsFname)
{
	ifstream fh(paramsFname);
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
		case 5:
			trainAlgoType = line;
			break;
		case 6:
			if(line == "true")
				preProcessData = true;
			else if(line == "false")
				preProcessData = false;
			else
				cout<<"invalid option for preProcessData"<<endl;
			break;
		case 7:
			loadWeights = line;
			cout<<"loadWeights:"<<loadWeights<<endl;
			break;
		}
	}
//	nLayers = outFnType.size();
//	cout<<"no.of layers: "<<nLayers<<endl;
}
