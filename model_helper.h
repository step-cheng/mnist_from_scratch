#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <array>
#include <eigen3/Eigen/Dense>
#include <eigen3/EigenRand-0.5.0/EigenRand/EigenRand>


using namespace std;
using namespace Eigen;

void initialize(vector<int> dims, map<string,MatrixXd> &params, map<string,MatrixXd> &vcs, map<string,MatrixXd> &grads) {
	cout << "initialize random number generator" << endl;
	Rand::P8_mt19937_64 urng{ 42 };

	for (int i=1; i<dims.size(); i++) {
		// constructs generator for normal distribution with He Initialization
		cout << dims[i-1] << endl;
		Rand::NormalGen<double> gen{0.0 ,sqrt(2.0/dims[i-1])};
		
		params["W"+to_string(i)] = gen.template generate<MatrixXd>(dims[i-1],dims[i],urng);
		params["b"+to_string(i)] = gen.template generate<MatrixXd>(1,dims[i],urng);
		vcs["dW"+to_string(i)] = MatrixXd::Zero(dims[i-1],dims[i]);
		vcs["db"+to_string(i)] = MatrixXd::Zero(1,dims[i]);
		grads["dW"+to_string(i)] = MatrixXd::Zero(dims[i-1],dims[i]);
		grads["db"+to_string(i)] = MatrixXd::Zero(1,dims[i]);
	}
}

MatrixXd relu(MatrixXd Z) {
	cout << "relu" << endl;
	return (Z.array() > 0).cast<double>();
}

MatrixXd softmax(MatrixXd Z) {
	cout << "softmax" << endl;
	MatrixXd Z_ = Z.colwise() - Z.rowwise().maxCoeff();
	assert(Z.rows() == Z_.rows());
	MatrixXd Z_exp = Z_.array().exp();
	assert(Z_exp.rows() == Z_.rows());
	MatrixXd A = Z_exp.array().colwise() / Z_exp.array().rowwise().sum();
	assert(Z_exp.rowwise().sum().cols() == 1);
	assert(A.rows() == Z_exp.rows());

	return A;
}

void forward_pass(map<string,MatrixXd> &forward, MatrixXd X, map<string,MatrixXd> &params) {
	int L = params.size() /2;
	forward["A0"] = X;
	
	for (int i=1; i<L; i++) {
		cout << "linear transform and relu" << endl;
		cout << forward.at("A"+to_string(i-1)).rows() << " x " << forward.at("A"+to_string(i-1)).cols() << endl;
		cout << params.at("W"+to_string(i)).rows() << " x " << params.at("W"+to_string(i)).cols() << endl;
		cout << params.at("b"+to_string(i)).rows() << " x " << params.at("b"+to_string(i)).cols() << endl;
		MatrixXd xW = forward.at("A"+to_string(i-1)) * params.at("W"+to_string(i));
		forward["Z"+to_string(i)] = xW.rowwise() + params.at("b"+to_string(i)).row(0);
		forward["A"+to_string(i)] = relu(forward["Z"+to_string(i)]);
	}
	cout << "linear transform and softmax" << endl;
	MatrixXd xW = forward.at("A"+to_string(L-1)) * params.at("W"+to_string(L));
	forward["Z"+to_string(L)] = xW.rowwise() + params.at("b"+to_string(L)).row(0);
	forward["A"+to_string(L)] = softmax(forward["Z"+to_string(L)]);
	cout << "finished forward pass" << endl;
}

double accuracy(MatrixXd A, MatrixXi Y) {
	assert (A.rows() == Y.rows() && A.cols() == Y.cols());
	MatrixXi pred = MatrixXi::Zero(A.rows(),A.cols());
	Index maxIndex;
	for (int i=0; i<A.rows(); i++) {
		// Check to see if this works
		int max = A.row(i).maxCoeff(&maxIndex);
		pred(i, maxIndex) = 1;
	}
	VectorXi results = (pred.rowwise().maxCoeff().array() == 2).cast<int>();
	// pred[]
	assert (results.size() == A.rows());
	return results.sum() / results.size();
}

MatrixXd relu_deriv(MatrixXd A) {
	return (A.array() > 0).cast<double>();
}

MatrixXd softmax_crossentropy_deriv(MatrixXd A, MatrixXi Y) {
	return A - Y.cast<double>();
	cout << "softmax successful" << endl;
}

void back_pass(map<string,MatrixXd> &forward, map<string,MatrixXd> &params, map<string,MatrixXd> &grads, MatrixXi Y) {
	int L = params.size() / 2;

	double N = Y.rows();
	cout << "dZ" << endl;
	MatrixXd dZ = 1* softmax_crossentropy_deriv(forward.at("A"+to_string(L)), Y);
	cout << "test1" << endl;
	cout << "forward A2 " << forward.at("A"+to_string(L-1)).transpose().rows() << " x " << forward.at("A"+to_string(L-1)).transpose().cols() << endl;
	cout << "dZ " << dZ.rows() << " x " << dZ.cols() << endl;
	grads["dW"+to_string(L)] = (1.0/N) * forward.at("A"+to_string(L-1)).transpose() * dZ;
	cout << "grads dW3 " << grads["dW"+to_string(L)].rows() << " x " << grads["dW"+to_string(L)].cols() << endl;
	cout << "test2" << endl;
	assert (grads.at("dW"+to_string(L)).rows() == params.at("W"+to_string(L)).rows());
	grads["db"+to_string(L)] = 1.0/N * dZ.colwise().sum();
	assert (grads.at("db"+to_string(L)).rows() == params.at("b"+to_string(L)).rows());

	MatrixXd dA;
	for (int i=L-1; i>0; i--) {
		cout << "thing a" << i << endl;
		cout << "dZ " << dZ.rows() << " x " << dZ.cols() << endl;
		cout << "W " << params.at("W"+to_string(i+1)).transpose().rows() << " x " << params.at("W"+to_string(i+1)).transpose().cols() << endl;
		dA = dZ * params.at("W"+to_string(i+1)).transpose();
		cout << "dA " << dA.rows() << " x " << dA.cols() << endl;
		cout << "relu forward Z" << i << " " << relu_deriv(forward["Z"+to_string(i)]).rows() << " x " << relu_deriv(forward["Z"+to_string(i)]).cols() << endl;
		dZ = dA.array() * relu_deriv(forward.at("Z"+to_string(i))).array();
		cout << "thing b" << i << endl;
		grads.at("dW"+to_string(i))= 1.0/N * forward.at("A"+to_string(i-1)).transpose()* dZ;
		assert (grads.at("dW"+to_string(i)).rows() == params.at("W"+to_string(i)).rows());
		cout << "thing c" << i << endl;
		grads.at("db"+to_string(i)) = 1.0/N * dZ.colwise().sum();
		cout << "db" << i << " " << grads.at("db"+to_string(i)).rows() << " x " << grads.at("db"+to_string(i)).cols() << endl;
		assert (grads.at("db"+to_string(i)).rows() == params.at("b"+to_string(i)).rows());
	}

}

// Cosine Annealing learning rate schedule
double get_lr(double rate, int i, int &iterations) {
	double lr = 0.01 + 1/2 * (rate-0.01)*(1+cos(double(i)/iterations*M_PI));
	return lr;
}

// Gradient Descent with Momentum
void learn(map<string,MatrixXd> &grads, map<string,MatrixXd> &params, map<string,MatrixXd>& vcs, double rate, double rho) {
	int L = params.size() / 2;
	for (int i=1; i<L+1; i++) {
		cout << "vcs dW " << i << vcs.at("dW"+to_string(i)).rows() << " x " << vcs.at("dW"+to_string(i)).cols() << endl;
		cout << "grads dW " <<  i << grads["dW"+to_string(i)].rows() << " x " << grads["dW"+to_string(i)].cols() << endl;
		vcs["dW"+to_string(i)] = rho*vcs["dW"+to_string(i)] + grads["dW"+to_string(i)];

		cout << "vcs db " << i << vcs.at("db"+to_string(i)).rows() << " x " << vcs.at("db"+to_string(i)).cols() << endl;
		cout << "grads db " <<  i << grads["db"+to_string(i)].rows() << " x " << grads["db"+to_string(i)].cols() << endl;
		vcs["db"+to_string(i)] = rho*vcs["db"+to_string(i)] + grads["db"+to_string(i)];

		cout << "params W " << i << params["W"+to_string(i)].rows() << " x " << params["W"+to_string(i)].cols() << endl;
		cout << "vcs dW " <<  i << vcs["dW"+to_string(i)].rows() << " x " << vcs["dW"+to_string(i)].cols() << endl;
		params["W"+to_string(i)] = params["W"+to_string(i)] - rate*vcs["dW"+to_string(i)];

		cout << "params b " << i << params["b"+to_string(i)].rows() << " x " << params["b"+to_string(i)].cols() << endl;
		cout << "vcs db " <<  i << vcs["db"+to_string(i)].rows() << " x " << vcs["db"+to_string(i)].cols() << endl;
		params["b"+to_string(i)] = params["b"+to_string(i)] - rate*vcs["db"+to_string(i)];
	}
}


map<string, MatrixXd> model_train(MatrixXd &img_data, MatrixXi &label_data, vector<int> dims, int iterations, int batches, double rate, double rho) {
	map<string,MatrixXd> params, vcs, forward, grads;
	cout << "Initializing" << endl;
	initialize(dims,params,vcs, grads);

	int batch_size = int(label_data.rows() / batches);
	double lr;

	int L = params.size()/2;
	vector<double> accuracies;
	for (int i=0; i<iterations; i++) {
		for (int j=0; j<batches; j++) {
			MatrixXd x = img_data;
			cout << "Doing a forward pass" << endl;
			forward_pass(forward, img_data(seqN(j*batch_size,batch_size), all), params);
			cout << "Calculating accuracy" << endl;
			double acc = accuracy(forward["A"+to_string(L)], label_data(seqN(j*batch_size,batch_size), all));
			accuracies.push_back(acc);
			cout << "Starting back pass" << endl;
			back_pass(forward, params, grads, label_data(seqN(j*batch_size,batch_size), all));
			cout << "get lr" << endl;
			lr = get_lr(rate,i,iterations);
			cout << "learn" << endl;
			learn(grads,params,vcs, lr,rho);
		}
		cout << "Accuracy at epoch " << i+1 << ": " << accuracies[i] << endl;
		
	}
	cout << "Final training accuracy: " << accuracies.back() << endl;

	return params;


}
