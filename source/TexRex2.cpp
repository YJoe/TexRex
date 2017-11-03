#include <iostream>
#include <ctime>
#include "..\include\NeuralNetwork.h"

using namespace std;

void print_vector_neatly(vector<double> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

int main() {
    srand(unsigned(time(nullptr)));
    vector<int> topology = {2, 2, 1, 1};
    int training_index = 0;
    double has_learnt_threshold = 0.01;
    bool log_basic_stuff = true;
	NeuralNetwork network = NeuralNetwork(&topology);

    vector<vector<double>> inputs = {
            {0.2, 0.4},
            {0.5, 0.3},
            {0.3, 0.5},
            {0.0, 0.0}
    };

    vector<vector<double>> targets = {
            {0.6},
            {0.8},
            {0.8},
            {0.0}
    };

    for(int i = 0; i < 1000; i++){

        // setting the training set
        training_index = random_int(0, inputs.size() - 1);

        if(log_basic_stuff){
            //printing the training set
            cout << "\niteration [" << i << "] training using set [" << training_index << "] ";
            print_vector_neatly(inputs[training_index]);
            cout << " = ";
            print_vector_neatly(targets[training_index]);
            cout << endl;
        }

        //feeding forward
        network.net_feed_forward(&inputs[training_index]);

        if(log_basic_stuff){
            //print the results
            cout << "network results ";
            network.print_results();
        }

        // back propagate to correct the network
        network.backwards_propagation(&targets[training_index]);

        if(log_basic_stuff){
            // how're we doin'?
            cout << "net recent average error [" << network.get_recent_average_error() << "] [";
            if(network.get_recent_average_error() < has_learnt_threshold){
                cout << "yayy]" << endl;
            } else {
                cout << "nope]" << endl;
            }
        }
    }

    cout << "\n-------------------------------------------------------" << endl;

    // see how well the network can mirror the training sets
    for(int i = 0; i < inputs.size(); i++) {
        cout << "\nthinking about the set [" << i << "] ";
        print_vector_neatly(inputs[i]);
        cout << " = ";
        print_vector_neatly(targets[i]);
        cout << endl;

        network.net_feed_forward(&inputs[i]);
        cout << "the network came up with ";
        network.print_results();
    }

	cin.get();

    return 0;
}