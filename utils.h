// Utils file for functions

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <cassert>
#include <cmath>

using namespace std;
using namespace Eigen;


MatrixXi one_hot_encode(vector<int> labels) {
	int rows = labels.size();
	Matrix<int,Dynamic,Dynamic> one_hot(labels.size(),10);
	for (int i=0; i<labels.size(); i++) {
		vector<int> encoding(10,0);
		encoding[labels[i]] = 1;
		one_hot.row(i) = Map<VectorXi>(encoding.data(), 10);
	}

	return one_hot;
}


// Parse data from csv file
void parse(string path, MatrixXi &x, MatrixXi &y) {
	ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error: Could not open the CSV file." << endl;
        exit(1);
    }

    string line;
    vector<int> x_vector;
    vector<int> y_vector;
    bool skip = false;
	while (getline(file,line)) {
        if (!skip) {
            skip = true;
            continue;
        }

        istringstream iss(line);
        string token;
        bool label = false;
        vector<int> row;
        while (getline(iss, token, ',')) {
            // Convert each token to an integer and add it to the vector
            int num = stoi(token);
            if (label) {
                row.push_back(num);
            }
            else {
                label = true;
                y_vector.push_back(num);
            }
        }
        x_vector.insert(x_vector.end(), row.begin(),row.end());
    }
    file.close();
	x = Map<Matrix<int,Dynamic,Dynamic,RowMajor>> (x_vector.data(),x_vector.size()/784,784);
	y = one_hot_encode(y_vector);

}

// shuffle img and label data in sync
void shuffle(MatrixXi &x, MatrixXi &y) {
	vector<int> indices;
	indices.reserve(x.rows());
	for (int i = 0; i < x.rows(); i++) {
		indices.push_back(i);
	}
	random_shuffle(indices.begin(), indices.end());

	MatrixXi temp_x(x.rows(),x.cols());
	MatrixXi temp_y(y.rows(),y.cols());
	
	for (int i=0; i < indices.size(); i++) {
		temp_x.row(i) = x.row(indices[i]);
		temp_y.row(i) = y.row(indices[i]);
	}
	x = temp_x;
	y = temp_y;
}

void split(MatrixXi x, MatrixXi y, MatrixXd &train_x, 
			MatrixXi &train_y, MatrixXd &val_x, MatrixXi &val_y, double r) {
	int div = r*x.rows();
	train_x = x.topRows(div).cast<double>();
	train_y = y.topRows(div);
	val_x = x.bottomRows(x.rows()-div).cast<double>();
	val_y = y.bottomRows(y.rows()-div);
}

pair<double, double> normalize(MatrixXd &img_data) {
	double mean = img_data.mean();
	int num = img_data.size();

	
	MatrixXd dummy = img_data;
	dummy = (dummy.array() - mean).square();
	double std = sqrt(dummy.sum()/num);
	
	img_data = (img_data.array() - mean) / std;
	return make_pair(mean,std);
}

