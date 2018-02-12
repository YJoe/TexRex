#include <stdio.h>
#include "..\include\SimpleCommandInterface.h"
#include "..\include\ImageFun.h"

void auto_crop_folder(string folder, string ext, string dest_folder) {
	// variables that we will use to read directories
	HANDLE hFind;
	WIN32_FIND_DATA data;

	string search_file = folder+ ext;

	cout << folder << " " << dest_folder << endl;

	hFind = FindFirstFile(search_file.c_str(), &data);
	if (hFind != INVALID_HANDLE_VALUE) {

		// while there is another file
		do {
	
			//cout << folder << data.cFileName << endl;
			cv::Mat temp = cv::imread(folder + data.cFileName, cv::IMREAD_GRAYSCALE);
			crop_segment(temp, 5);
			cv::bitwise_not(temp, temp);
			cv::imwrite(dest_folder + data.cFileName, temp);
		} while (FindNextFile(hFind, &data));
	}
}

int main() {
	/*string abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	for (int i = 0; i < abc.size(); i++) {
		auto_crop_folder(string("data/NIST/testing/") + abc[i] + string("/"), "*.png", string("data/NIST2/testing/") + abc[i] + string("/"));
	}
*/
	SimpleCommandInterface sci;

	//
	sci.evaluate_command("loadnet data/cnn_json/char_net.json");
	sci.evaluate_command("loadset data/NIST2/training/ 100 ABCDEFGHIJKLMNOPQRSTUVWXYZ");
	sci.evaluate_command("setiteration 500");
	sci.evaluate_command("trainnet test1/data1.dat 50");
	sci.evaluate_command("viewgraph test_plot.plt");
	sci.evaluate_command("savenet data/cnn_json/char_net_trained2.json");
	
	//sci.evaluate_command("loadset data/NIST2/testing/ 10 ABCDEFGHIJKLMNOPQRSTUVWXYZ");
	sci.evaluate_command("testnet");
	
	sci.begin();

	//sci.evaluate_command("createnet data/cnn_json/demo_net.json");
	//sci.evaluate_command("loadset data/MNIST/testing/ 100 0123456789");
	//sci.evaluate_command("testnet");
	//cin.get();

	//sci.evaluate_command("loadset data/MNIST/training/ 100 0123456789");
	//sci.evaluate_command("setiteration 500");
	//sci.evaluate_command("trainnet test1/data1.dat 50");
	//sci.evaluate_command("viewgraph test_plot.plt");
	//
	//sci.evaluate_command("loadset data/MNIST/testing/ 100 0123456789");
	//sci.evaluate_command("testnet");
	//sci.begin();
	cin.get();
}
