#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <utility>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/EigenRand-0.5.0/EigenRand/EigenRand>
#include <ctime>
#include "utils.h"
#include "model_helper.h"

using namespace std;
using namespace Eigen;

// Change to float instead of double!!
int main()
{
    MatrixXf m1(60000,784);
    m1.setConstant(0.1);
    MatrixXf m2(784,128);
    m2.setConstant(0.2);
    cout << "try" << endl;
    auto t1 = chrono::system_clock::now();
    // cout << m1*m2 << endl;
    MatrixXf m = m1*m2;
    auto t2 = chrono::system_clock::now();
    auto seconds = chrono::duration_cast<chrono::seconds>(t2-t1).count();
    cout << "Time to do large float matrix multiplication: " << seconds << "s!" << endl;
    // Rand::P8_mt19937_64 urng{ 42 };
    // Rand::NormalGen<double> gen{0.0 ,1};
    // RowVectorXd row = (gen.template generate<RowVectorXd>(1,10,urng)).row(0);
    // MatrixXd m(3,3);
    // m << 0,2,0,
    //     0,1,1,
    //     1,0,1;
    // // MatrixXd mv(1,3);
    // // m << 1,1,1;
    // RowVectorXd v(3);
    // v << 1,1,1;
    // cout << m.rowwise() + v << endl;
    // // cout << m.rowwise() + mv << endl;
    // // VectorXf v(3);
    // // v << 0,1,2;
    // cout << (m.rowwise().maxCoeff().array() == 2) << endl;

    cout << "Parsing data..." << endl;
    // parse data
    filesystem::path cwd = filesystem::current_path();
    string path_train = cwd.string() + "\\dataset\\mnist_train.csv";
    string path_test = cwd.string() + "\\dataset\\mnist_test.csv";

    MatrixXi x_train, y_train, x_test, y_test;

    parse(path_train,x_train,y_train);
    cout<< "x_train shape: " << x_train.rows() << 'x' << x_train.cols() << endl;
    cout<< "y_train shape: " << y_train.rows() << 'x' << y_train.cols() << endl;
    parse(path_test,x_test,y_test);
    cout<< "x_test shape: " << x_test.rows() << 'x' << x_test.cols() << endl;
    cout<< "y_test shape: " << y_test.rows() << 'x' << y_test.cols() << endl;
    

    cout << "Shuffling data... " << endl;
    // shuffle data
    shuffle(x_train,y_train);
    shuffle(x_test,y_test);

    cout << "Organizing and reformatting data... " << endl;
    // make train, val, and test sets
    MatrixXd train_imgs, val_imgs, test_imgs;
    MatrixXi train_labels, val_labels, test_labels;
    double r = 5.0/6.0;
    split(x_train,y_train, train_imgs, train_labels, val_imgs, val_labels, r);
    test_imgs = x_test.cast<double>();
    test_labels = y_test;
    cout << "train_imgs shape " << train_imgs.rows() << 'x' << train_imgs.cols() << endl;
    cout << "train_labels shape " << train_labels.rows() << 'x' << train_labels.cols() << endl;
    cout << "val_imgs shape " << val_imgs.rows() << 'x' << val_imgs.cols() << endl;
    cout << "val_labels shape " << val_labels.rows() << 'x' << val_labels.cols() << endl;
    cout << "test_imgs shape " << test_imgs.rows() << 'x' << test_imgs.cols() << endl;
    cout << "test_labels shape " << test_labels.rows() << 'x' << test_labels.cols() << endl;

    pair<double, double> train_mean_std = normalize(train_imgs);
    pair<double, double> val_mean_std = normalize(val_imgs);
    pair<double, double> test_mean_std = normalize(test_imgs);


    vector<int> dims = {784,128,64,10};
    int batches = 1;
    int iterations = 10;
    double rate = 0.05;
    double rho = 0.9;
    
    cout << "Start Training!" << endl;
    auto start = chrono::system_clock::now();
    map<string,MatrixXd> params = model_train(train_imgs, train_labels, dims, iterations, batches, rate, rho);


    auto end = chrono::system_clock::now();
    // cout << end - start << endl;
    
    return 0;
}