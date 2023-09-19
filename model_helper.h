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

void initialize(vector<int> dims, map<string,MatrixXf> &params, map<string,MatrixXf> &vcs, map<string,MatrixXf> &grads) {
	Rand::P8_mt19937_64 urng{ 42 };

	for (int i=1; i<dims.size(); i++) {
		// constructs generator for normal distribution with He Initialization
		Rand::NormalGen<float> gen{0 ,sqrt(float(2)/dims[i-1])};
		
		params["W"+to_string(i)] = gen.template generate<MatrixXf>(dims[i-1],dims[i],urng);
		params["b"+to_string(i)] = gen.template generate<MatrixXf>(1,dims[i],urng);
		vcs["dW"+to_string(i)] = MatrixXf::Zero(dims[i-1],dims[i]);
		vcs["db"+to_string(i)] = MatrixXf::Zero(1,dims[i]);
		grads["dW"+to_string(i)] = MatrixXf::Zero(dims[i-1],dims[i]);
		grads["db"+to_string(i)] = MatrixXf::Zero(1,dims[i]);
	}
}

MatrixXf relu(MatrixXf Z) {
	return (Z.array() > 0).cast<float>();
}

MatrixXf softmax(MatrixXf Z) {
	MatrixXf Z_ = Z.colwise() - Z.rowwise().maxCoeff();
	assert(Z.rows() == Z_.rows());
	MatrixXf Z_exp = Z_.array().exp();
	assert(Z_exp.rows() == Z_.rows());
	MatrixXf A = Z_exp.array().colwise() / Z_exp.array().rowwise().sum();
	assert(Z_exp.rowwise().sum().cols() == 1);
	assert(A.rows() == Z_exp.rows());

	return A;
}

void forward_pass(map<string,MatrixXf> &forward, MatrixXf X, map<string,MatrixXf> &params) {
	int L = params.size() /2;
	forward["A0"] = X;
	
	for (int i=1; i<L; i++) {
		MatrixXf xW = forward.at("A"+to_string(i-1)) * params.at("W"+to_string(i));
		forward["Z"+to_string(i)] = xW.rowwise() + params.at("b"+to_string(i)).row(0);
		forward["A"+to_string(i)] = relu(forward["Z"+to_string(i)]);
	}
	MatrixXf xW = forward.at("A"+to_string(L-1)) * params.at("W"+to_string(L));
	forward["Z"+to_string(L)] = xW.rowwise() + params.at("b"+to_string(L)).row(0);
	forward["A"+to_string(L)] = softmax(forward["Z"+to_string(L)]);
}

float accuracy(MatrixXf A, MatrixXi Y) {
	assert (A.rows() == Y.rows() && A.cols() == Y.cols());
	MatrixXi pred = MatrixXi::Zero(A.rows(),A.cols());
	Index maxIndex;
	for (int i=0; i<A.rows(); i++) {
		// Check to see if this works
		float max = A.row(i).maxCoeff(&maxIndex);
		pred(i, maxIndex) = 1;
	}

	VectorXi results_pre = (pred+Y).rowwise().maxCoeff();
	VectorXi results = (results_pre.array() == 2).cast<int>();
	assert (results.size() == A.rows());
	cout << "Matches: " << results.sum() << endl;
	return static_cast<float>(results.sum()) / results.size();
}

MatrixXf relu_deriv(MatrixXf A) {
	return (A.array() > 0).cast<float>();
}

MatrixXf softmax_crossentropy_deriv(MatrixXf A, MatrixXi Y) {
	return A - Y.cast<float>();
}

void back_pass(map<string,MatrixXf> &forward, map<string,MatrixXf> &params, map<string,MatrixXf> &grads, MatrixXi Y) {
	int L = params.size() / 2;

	int N = Y.rows();
	MatrixXf dZ = 1* softmax_crossentropy_deriv(forward.at("A"+to_string(L)), Y);
	grads["dW"+to_string(L)] = (1.0/N) * forward.at("A"+to_string(L-1)).transpose() * dZ;
	assert (grads.at("dW"+to_string(L)).rows() == params.at("W"+to_string(L)).rows());
	grads["db"+to_string(L)] = 1.0/N * dZ.colwise().sum();
	assert (grads.at("db"+to_string(L)).rows() == params.at("b"+to_string(L)).rows());

	MatrixXf dA;
	for (int i=L-1; i>0; i--) {
		dA = dZ * params.at("W"+to_string(i+1)).transpose();
		dZ = dA.array() * relu_deriv(forward.at("Z"+to_string(i))).array();
		grads.at("dW"+to_string(i))= 1.0/N * forward.at("A"+to_string(i-1)).transpose()* dZ;
		assert (grads.at("dW"+to_string(i)).rows() == params.at("W"+to_string(i)).rows());
		grads.at("db"+to_string(i)) = 1.0/N * dZ.colwise().sum();
		assert (grads.at("db"+to_string(i)).rows() == params.at("b"+to_string(i)).rows());
	}

}

// Cosine Annealing learning rate schedule
float get_lr(float rate, int i, int &iterations) {
	float lr = 0.005 + 1.0/2.0 * (rate-0.005)*(1+cos(static_cast<float>(i)/iterations*M_PI));
	return lr;
}

// Gradient Descent with Momentum
void learn(map<string,MatrixXf> &grads, map<string,MatrixXf> &params, map<string,MatrixXf>& vcs, float rate, float rho) {
	int L = params.size() / 2;
	for (int i=1; i<L+1; i++) {
		vcs["dW"+to_string(i)] = rho*vcs["dW"+to_string(i)] + grads["dW"+to_string(i)];
		vcs["db"+to_string(i)] = rho*vcs["db"+to_string(i)] + grads["db"+to_string(i)];
		params["W"+to_string(i)] = params["W"+to_string(i)] - rate*vcs["dW"+to_string(i)];
		params["b"+to_string(i)] = params["b"+to_string(i)] - rate*vcs["db"+to_string(i)];
	}
}


map<string, MatrixXf> model_train(MatrixXf &img_data, MatrixXi &label_data, vector<int> dims, int iterations, int batches, float rate, float rho) {
	cout << "Initializing variables... " << endl;
	map<string,MatrixXf> params, vcs, forward, grads;
	initialize(dims,params,vcs, grads);

	int batch_size = int(label_data.rows() / batches);
	float lr, acc;
	float sum_accs = 0;
	int L = params.size()/2;
	vector<float> accuracies;

	for (int i=0; i<iterations; i++) {
		for (int j=0; j<batches; j++) {
			cout << "Epoch " << i << ", Batch " << j << endl;
			cout << "Forward pass" << endl;
			forward_pass(forward, img_data(seqN(j*batch_size,batch_size), all), params);
			acc = accuracy(forward["A"+to_string(L)], label_data(seqN(j*batch_size,batch_size), all));
			cout << "Batch accuracy: " << acc << endl;
			sum_accs = sum_accs + acc;
			cout << "Backpropagating..." << endl;
			back_pass(forward, params, grads, label_data(seqN(j*batch_size,batch_size), all));
			lr = get_lr(rate,i,iterations);
			// lr = rate;
			cout << "learning rate: " << lr << endl;
			cout << "learning..." << endl;
			learn(grads,params,vcs, lr,rho);
		}
		cout << "Accuracy at epoch " << i+1 << ": " << sum_accs/batches << endl;
		accuracies.push_back(sum_accs/batches);
		sum_accs = 0;
	}
	cout << "Final Training Accuracy: " << sum_accs/batches << endl;

	return params;


}

double model_test(MatrixXf &img_data, MatrixXi &label_data, map<string,MatrixXf> &params, int batches) {
	map<string,MatrixXf> forward;
	int batch_size = int(label_data.rows() / batches);
	float acc;
	float sum_accs = 0;

	int L = params.size()/2;

	for (int j=0; j<batches; j++) {
		cout << "Batch " << j << endl;
		cout << "Forward pass" << endl;
		forward_pass(forward, img_data(seqN(j*batch_size,batch_size), all), params);
		acc = accuracy(forward["A"+to_string(L)], label_data(seqN(j*batch_size,batch_size), all));
		cout << "Batch accuracy: " << acc << endl;
		sum_accs = sum_accs + acc;
	}
	double test_acc = sum_accs/batches;
	cout << "Test accuracy: " << test_acc << endl;
	return test_acc;

}