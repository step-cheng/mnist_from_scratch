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
    
    cout << "Parsing data..." << endl;
    // parse data
    filesystem::path cwd = filesystem::current_path();
    string path_train = cwd.string() + "\\dataset\\mnist_train.csv";
    string path_test = cwd.string() + "\\dataset\\mnist_test.csv";

    MatrixXi x_train, y_train, x_test, y_test;

    parse(path_train,x_train,y_train);
    parse(path_test,x_test,y_test);

    cout << "Shuffling data... " << endl;
    // shuffle data
    shuffle(x_train,y_train);
    shuffle(x_test,y_test);

    cout << "Organizing and reformatting data... " << endl;
    // make train, val, and test sets
    MatrixXf train_imgs, val_imgs, test_imgs;
    MatrixXi train_labels, val_labels, test_labels;
    float r = 5.0/6.0;
    split(x_train,y_train, train_imgs, train_labels, val_imgs, val_labels, r);
    test_imgs = x_test.cast<float>();
    test_labels = y_test;
    
    pair<float, float> train_mean_std = normalize(train_imgs);
    pair<float, float> val_mean_std = normalize(val_imgs);
    pair<float, float> test_mean_std = normalize(test_imgs);


    vector<int> dims = {784,128,64,10};
    int batches = 1;
    int iterations = 20;
    float rate = 0.05;
    float rho = 0.9;
    
    cout << "Start Training!" << endl;
    auto start = chrono::system_clock::now();
    map<string,MatrixXf> params = model_train(train_imgs, train_labels, dims, iterations, batches, rate, rho);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end-start).count();
    cout << "Total Training Time: " << duration << "s" << endl;
    
    batches = 1;
    cout << "Start Testing!" << endl;
    double test_acc = model_test(test_imgs, test_labels, params, batches);

    return 0;
}