/*
 * TestModule.cpp
 *
 *  Created on: Jun 3, 2014
 *      Author: anirudh
 */
#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<ctime>
#include<armadillo>
//#include"Dnn_v1.cpp"
#include"utils.cpp"
#include<boost/lexical_cast.hpp>


using namespace std;
using namespace arma;
using boost::lexical_cast;

mat test_fn();
void test_fn1(mat&);
void test_fn2(mat);

//mat globalC(10,10,fill::zeros);
int main(int argc,char **argv)
{
//	string s,syscmd,fname;
//	vector<double> fvec;
//	s= "0.234 34.345 0.34534 23423.34534 564.211 45664 23425.4 56332 1231231";
//	const char *fname="train.lpcc";
//	Mat<float> m;
//	vector< Mat<float> > wts;
//	const char *cc;
//	cc="something";
//	cout<<cc<<endl;
//	string line = "this is a text string";
//	string line = "-0.37617 -0.21323 -0.18691 0.18276 0.081718 0.11981 0.034669 0.19177 0.25238 0.13051 0.13056 0.0093613 -0.025096 0.042469 0.081165";
//	syscmd = "wc -l "+fname;
//	cout<<syscmd<<endl;
//	system(syscmd);
//	stringstream ss(s);
//	split(s,fvec);
//	cout<<fvec[0]+fvec[1]<<endl;
//	cout<<fvec.size()<<endl;
//	mat matrix;
//	rand
//	matrix = randu(4,5);
//	cout<<matrix;
//	cout<<no_of_lines(fname)<<endl;
//	cout<<line<<endl;
//	split(line,fvec);
//	print_vec(fvec);
//	cout<<fvec[0]<<"\n"<<fvec[1]<<endl;
//	cout<<fvec[0]+fvec[1]<<endl;

//	DNN nn("params");
//	m.randu(5,10);
//	m.print("m:");
//	arma_rng::set_seed(time(NULL));
//	for(int i=0; i<3; i++)
//	{
//		wts.push_back(randu< Mat<float> >(5,10));
//	}
//	for(int i=0;i<3;i++)
//		wts[i].print("layer wts:");
//	fvec.push_back(2.3434);
//	fvec.push_back(45645.43542);
//	print_vec(fvec);
//	fvec[1]=1.001;
//	print_vec(fvec);
//	int batchSize=10000;
//	mat *A = new mat();
//	A->eye(5,4);
////	A->print("A:");
//	arma_rng::set_seed_random();
////	mat *B = new mat(A->n_rows,5);
//	mat *B = new mat();
//	uvec inds,inds2;
//	inds = randi<uvec>(5,distr_param(0,3));
//	inds.print("inds:");
//	inds2 = inds.subvec(0,3);
//	inds2.print("inds2:");
//	*B = A->cols(inds2);
//	B->print("B:");
//	B=square(2*(*A));
//	B.print("B:");
//	A->print("A:");
//	double s;
//	rowvec C;
//	rowvec sumA;
//	rowvec sumB;
//	sumA = sum(square(*A));
//	sumB = sum(square(B));
//	sumA.print("sumA:");
//	sumB.print("sumB:");
//	C = sum(square(*A)) / sum(square(B));
//	C.print("C:");
//	C = sum(square(*A)) / sum(square(B));
//	s=sum(C) / 10;
//////	C=sumA / sumB;
//	cout<<"Sum: "<<s<<endl;
//	C.print("C:");
//	A->print("A:");
//	mat *B = new mat(3,100000,fill::ones);
//	mat *sA = new mat(A->n_rows,batchSize);
//	mat *sB = new mat(B->n_rows,batchSize);
////	gen_batchdata(*A,*B,*sA,*sB,batchSize);
//	int sum=accu(*A);
//	cout<<"Sum:"<<sum<<endl;
//	rowvec rA(10,fill::ones);
//	rowvec rB(10,fill::ones);
//	rA=2*rA;
//	rB=rB / rA;
//	rB.print("rA:");
//	mat C;
//	C=2+(*A);
//	C.print("A:");
//	sA->print("A:");
//	sB->print("B:");
//	mat *M = new mat();
//	mat &C= (*M);
//	M->print("M before:");
//	C.print("C before:");
//	C.zeros(3,3);
//	C = (*A)+(*B);
//	M->print("M after:");
//	C.print("C after:");
//	mat D(3,3,fill::eye);
//	mat E(3,3,fill::eye);
//	mat F;
//	colvec R(3,fill::ones);
//	F.randn(3,4);
//	F.print("F:");
//	int t=10;
//	C=2*t*F-t;
//	C.print("C:");
//	E.print("E:");
//	E.each_col() += R;
//	E.print("E:");
//	int *j;
//	*j=20;
//	int &k=(*j);
//	cout<<"k="<<k<<endl;
//	cout<<"j="<<(*j)<<endl;
//
//	k=45;
//	cout<<"k="<<k<<endl;
//	cout<<"j="<<(*j)<<endl;

//	vector<mat*> matvec;
//	matvec.push_back(A);
//	matvec.push_back(B);
//	C = (*(matvec[0]))*(*(matvec[1]));
//	double &t=(*(matvec[0]))(1,2);
//	matvec[0]->print("A:");
////	t = (*A)(1,2);
//	cout<<t<<endl;
//	(*matvec[0])(1,2)=234.4;
//	matvec[0]->print("A:");
//	cout<<t<<endl;
//	C= D % E;
//	C.print("C:");
//	C.print("C:");
//	C=test_fn();
//	C.print("C:");
//	test_fn1(C);
//	C.print("C:");
//	C=(*A)*C;
//	C.print("C:");
//	test_fn2(C);
//	globalC.print("globalC:");

//	string s = "123.343";
//	float fs;
//	fs = lexical_cast<float>(s);
//	cout<<fs+10.0<<endl;
//	int batchSize = 1000;
//	int nPatterns = 102133;
//	int batchesPerEpoch = nPatterns/batchSize;
//	int batchesPerEpoch =10;
//	cout<<batchesPerEpoch<<endl;
//	int *frameNos = new int[batchSize];
//	vector<int> frameNos(batchSize);
//	srand(time(NULL));
//	for(int i=0;i<batchesPerEpoch;i++)
//	{
//		get_shuffled_framenums(frameNos,batchSize,nPatterns);
//		cout<<list_tostring(frameNos,batchSize);
//	}

//	string s = "123.343 453 3452 345.343 454";
//	vector<float> vec;
//	string_tolist(s,vec);
//	print_vec(vec);
	int choice=1;
	switch(choice)
	{
	case 1:
		cout<<"choice is"<<1<<endl;
		break;
	case 2:
		cout<<"choice is"<<2<<endl;
		break;
	default:
		cout<<"invalid choice"<<endl;

	}
}

mat test_fn()
{
	mat temp(5,5,fill::ones);
	return temp;
}

void test_fn1(mat& c)
{
	c(1,1) = 243.23234;
}

void test_fn2(mat c)
{
	mat d(c.n_rows,c.n_cols,fill::ones);
//	globalC=d;
}



