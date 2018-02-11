#include <stdio.h>
#include "..\include\SimpleCommandInterface.h"

int main() {
	SimpleCommandInterface sci;
	sci.evaluate_command("createnet data/cnn_json/demo_net.json");
	sci.evaluate_command("loadset data/MNIST/testing/ 100 0123456789");
	sci.evaluate_command("testnet");
	cin.get();

	sci.evaluate_command("loadset data/MNIST/training/ 100 0123456789");
	sci.evaluate_command("setiteration 500");
	sci.evaluate_command("trainnet test1/data1.dat 50");
	sci.evaluate_command("viewgraph test_plot.plt");
	
	sci.evaluate_command("loadset data/MNIST/testing/ 100 0123456789");
	sci.evaluate_command("testnet");

	sci.begin();
}
