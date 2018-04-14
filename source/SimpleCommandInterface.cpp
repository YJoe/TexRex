#include "..\include\SimpleCommandInterface.h"

int random_i(int min, int max) {
	return rand() % (max - min + 1) + min;
}

void view_data_set(vector<DataSample> samples) {
	for (int i = 0; i < samples.size(); i++) {
		cout << "image answer is [";
		for (int j = 0; j < samples[i].answer.size(); j++) {
			cout << samples[i].answer[j] << ", ";
		}
		cout << "]" << endl;
		cv::imshow("", samples[i].image_segment.m);
		cv::waitKey();
	}
}

void load_sized_data_sample(DataSample& destination, string directory, cv::Size size) {
	// construct the file name and load the image
	destination.image_segment.m = cv::imread(directory, cv::IMREAD_GRAYSCALE);

	// scale the image to meet the network input sized and convert to float
	cv::resize(destination.image_segment.m, destination.image_segment.m, size);
	destination.image_segment.m.convertTo(destination.image_segment.m, CV_32FC1, 1.0 / 255.0);

	// store the image as a vector of floats and release the image to save space
	get_vector(destination.image_segment.m, destination.image_segment.float_m_mini);
}

void load_sized_data_sample(DataSample& destination, cv::Mat& image, cv::Size size) {
	// construct the file name and load the image
	destination.image_segment.m = image;

	// scale the image to meet the network input sized and convert to float
	cv::resize(destination.image_segment.m, destination.image_segment.m, size);
	destination.image_segment.m.convertTo(destination.image_segment.m, CV_32FC1, 1.0 / 255.0);

	// store the image as a vector of floats and release the image to save space
	get_vector(destination.image_segment.m, destination.image_segment.float_m_mini);
}

vector<string> files_in_dir(string dir) {
	vector<string> files;

	HANDLE hFind;
	WIN32_FIND_DATA data;
	hFind = FindFirstFile(dir.c_str(), &data);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			files.emplace_back(data.cFileName);
		} while (FindNextFile(hFind, &data));
	}

	return files;
}

SimpleCommandInterface::SimpleCommandInterface() {
	function_help = {
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("help", "<function name>", "will provide some help on a function", &SimpleCommandInterface::help),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("loadnet", "<file location>", "loads an already defined json network from the given directory", &SimpleCommandInterface::loadnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("createnet", "<file location>", "creates and loads a json network from a simple template", &SimpleCommandInterface::createnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("formatset", "<input folder> <output folder> <image width> <image height>", "resizes a training set found in a folder", &SimpleCommandInterface::formatset),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("loadset", "<input folder> <testing||training> <sample size> (<target> <representation> || <representation>", "loads a data set from a given directory to the training or testing slot", &SimpleCommandInterface::loadset),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("trainnet", "<data output> <evaluation every n iterations> <number of unseen problems to test>", "begins the training proccess and logs into the given data file", &SimpleCommandInterface::trainnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("testnet", "", "begins the testing proccess for a loaded network and data set", &SimpleCommandInterface::testnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("setiteration", "<number of iterations>", "sets the network terminating condition to iteration with the number given", &SimpleCommandInterface::setiteration),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("viewgraph", "<plt file>", "displays a plot of the latest data file", &SimpleCommandInterface::viewgraph),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("setseed", "\"random\" or <number>", "sets the seed for which to make random choicess", &SimpleCommandInterface::setseed),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("evaluate", "<number of samples>", "shows the guesses made by the network for a given image", &SimpleCommandInterface::view_evaluations),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("savenet", "<save location>", "saves the network structure and weights", &SimpleCommandInterface::savenet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("setevaluation", "<softmax||threshold>", "the type of evaluation used to score the network, softmax for multiple, threshold or single", &SimpleCommandInterface::set_evaluation),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("testgroupnet", "<network directory> <network extension \"*.json\"> <network mapping>", "a test", &SimpleCommandInterface::group_net_test),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("demo", "<demo number>", "for presentation", &SimpleCommandInterface::demo)
	};
}

vector<string> SimpleCommandInterface::regex_split(const string& s, string rgx_str = "\\s+") {
	vector<string> elems;
	regex rgx(rgx_str);
	sregex_token_iterator iter(s.begin(), s.end(), rgx, -1);
	sregex_token_iterator end;

	while (iter != end) {
		elems.push_back(*iter);
		++iter;
	}

	return elems;
}

boolean SimpleCommandInterface::evaluate_command(string input) {
	cout << "evaluating [" << input << "]" << endl;

	if (input == "exit") {
		return false;
	}
	else if (input == "") {
		return true;
	}
	else {
		bool ret = handle_input(regex_split(input));
		cout << endl;
		return ret;
	}
}

void SimpleCommandInterface::begin() {

	// while the user has something to enter
	string input = "";
	boolean running = true;
	while (running) {

		// get the user input
		cout << "input >> ";
		getline(cin, input);

		running = evaluate_command(input);
	}

	// exit the interface
	cout << "bye :(" << endl;
}

void SimpleCommandInterface::test(){
	view_data_set(cnn.testingSamples);
}

char SimpleCommandInterface::get_type_code(string input) {
	
	// define regex rules to identify patterns in lines
	vector<pair<regex, char>> regex_patterns = {
		pair<regex, char>("(\\+|-)?[[:digit:]]+", 'I'),
		pair<regex, char>("((\\/?[a-zA-Z0-9_-])+\\.[a-zA-Z0-9]+)|(\\/?([a-zA-Z0-9].*\\.+|[a-zA-Z0-9].*\.\\/))", 'D'),
		pair<regex, char>("[A-Z|a-z|0-9]+", 'S')
	};

	// match and return codes
	for (int i = 0; i < regex_patterns.size(); i++) {
		if (regex_match(input, regex_patterns[i].first)) {
			return regex_patterns[i].second;
		}
	}
	
	// return a failure
	return '?';
}

bool SimpleCommandInterface::handle_input(vector<string>& input) {

	bool found = false;
	for (int i = 0; i < function_help.size(); i++) {
		if (input[0] == get<0>(function_help[i])) {
			(this->*get<3>(function_help[i]))(input);
			found = true;
			break;
		}
	}

	if (!found) {
		error_message(input[0]);
	}
	return found;
}

void SimpleCommandInterface::loadnet(vector<string>& input) {
	if (input.size() == 2) {
		if (file_exists(input[1])) {
			OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
			cnn = ConvolutionalNeuralNetwork(input[1], ocl, 0);
		}
		else {
			cout << "error - file [" << input[1] << "] doesn't exist" << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::createnet(vector<string>& input) {
	if (input.size() == 2) {
		create_template(input[1]);
		OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
		cnn = ConvolutionalNeuralNetwork(input[1], ocl, 0);
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::set_evaluation(vector<string>& input) {
	if (input[1] == "softmax") {
		cnn.set_softmax_evaluation(true);
	}
	else {
		cnn.set_softmax_evaluation(false);
	}
}

void SimpleCommandInterface::formatset(vector<string>& input) {
	cout << "formatting set not yet complete" << endl;
}

bool replace(string& str, const string& find, const string& put) {
	//https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
	size_t start_pos = str.find(find);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, find.length(), put);
	return true;
}

void SimpleCommandInterface::load_nist_binary(vector<DataSample>& data_samples, string folder, string target, string all_in_folder, int sample_count, cv::Size network_input_size) {

	// make sure we aren't loading images from the target set into the random set
	replace(all_in_folder, target, "");

	if (sample_count < all_in_folder.size()) {
		cout << "Not enough spread when selecting samples" << endl;
		cin.get();
		exit(0);
	}

	// load the correct answers that will just be images of the target labeled with the answer "1"
	load_nist(data_samples, folder, target, sample_count, false, network_input_size);

	// load a sample of all other images that arent the target all will be labeled with the answer "0"
	int per_char = sample_count / all_in_folder.size();
	load_nist(data_samples, folder, all_in_folder, per_char, true, network_input_size);

	//view_data_set(data_samples);
}

void SimpleCommandInterface::loadset(vector<string>& input) {

	if (input.size() == 5 || input.size() == 6) {
		if (is_number(input[3])) {
			// generate the training set
			vector<DataSample> data_samples;
			vector<char> mapping;

			// if we are using a multi problem network
			if (input.size() == 5) {
				load_nist(data_samples, input[2], input[4], atoi(input[3].c_str()), false, cnn.input_size);

				// creating mapping
				for (int i = 0; i < input[4].size(); i++) {
					mapping.emplace_back(input[4][i]);
				}
			}

			// we are using a single problem network
			else {

				/*
					EXAMPLE
					evaluating[loadset training data/NIST2/training/ 200 A BCD]
					input[0] is[loadset]
					input[1] is[training]
					input[2] is[data/NIST2/training/]
					input[3] is[200]
					input[4] is[A]
					input[5] is[BCD]
				*/

				// validate that we are only using one target
				if (input[4].size() > 1) {
					cout << "error - only one target can be selected -> [" << input[4] << "] is the issue" << endl;
					cin.get();
					exit(-1);
				}

				// input[4] in this case should be one char anyway so just take the first (and only index of the string)
				mapping = { input[4][0] };

				// load a set of correct inputs and several incorrect images
				// vector<data_sample>, folder, target, all_in_folder, sample_count, network_input_size 
				load_nist_binary(data_samples, input[2], input[4], input[5], atoi(input[3].c_str()), cnn.input_size);
			}

			if (input[1] == "training") {
				cnn.setTrainingSamples(data_samples);
			}
			else if (input[1] == "testing") {
				cnn.setTestingSamples(data_samples);
			}
			else {
				cout << "error - the data slot [" << input[1] << "] was not recognised, this should be either \"testing\" or \"training\"" << endl;
			}

			cnn.setMapping(mapping);
		}
		else {
			cout << "error - [" << input[3] << "] is not a number" << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::trainnet(vector<string>& input) {
	if (input.size() == 4) {
		if (cnn.is_defined) {
			if (is_number(input[2])) {
				ofstream data_file_clear(input[1]);
				data_file_clear.close();
				ofstream data_file(input[1], ios::app);
				if (!data_file.is_open()) {
					cout << "error - failed to create data logging file, try another location" << endl;
					return;
				}

				cout << "traing network - data will log to [" << input[1] << "] and will sample the classification rate every [" << input[2] << "] inputs" << endl;
				cout << "classification rates of the training set and [" << input[3] << "] random unseen problems will be recorded" << endl;
				cnn.train(data_file, atoi(input[2].c_str()), atoi(input[3].c_str()));
			}
			else {
				cout << "error - [" << input[1] << "] is not a number" << endl;
			}
		}
		else {
			cout << "error - try loading or creating a network first" << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::testnet(vector<string>& input) {
	if (input.size() == 2) {
		if(is_number(input[1])){
			cnn.test(atoi(input[1].c_str()));
		}
		else {
			cout << "error - [" << input[1] << "] is not a number" << endl;
		}
	}
	else {
		cnn.test(-1);
	}
}

void SimpleCommandInterface::setiteration(vector<string>& input) {
	if (input.size() == 2) {
		if (cnn.is_defined) {
			if (is_number(input[1])) {
				cnn.terminating_function = &ConvolutionalNeuralNetwork::iteration_check;
				cnn.iteration_target = atoi(input[1].c_str());
				cout << "the network will terminate training after [" << input[1] << "] iterations" << endl;
			}
			else {
				cout << "error - [" << input[1] << "] is not a number" << endl;
			}
		}
		else {
			cout << "error - try loading or creating a network first" << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::viewgraph(vector<string>& input) {
	if (input.size() == 2) {
		if (file_exists(input[1])) {
			ShellExecute(0, 0, input[1].c_str(), 0, 0, SW_SHOW);
		}
		else {
			cout << "the file [" << input[1].c_str() << "] does not exist" << endl; 
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::setseed(vector<string>& input) {
	if (input.size() == 2) {
		if (input[1] == "random") {
			srand(time(NULL));
			cout << "set network seed" << endl;
		}
		else if (is_number(input[1])) {
			srand(atoi(input[1].c_str()));
			cout << "set network seed" << endl;
		}
		else {
			cout << "try \"setseed random\" or \"setseed <number>\" " << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::savenet(vector<string>& input) {
	if (input.size() == 2) {
		cnn.json_dump_network(input[1]);
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::help(vector<string>& input) {
	if (input.size() == 1) {
		cout << "possible functions:" << endl;
		for (int i = 0; i < function_help.size(); i++) {
			cout << "[" << get<0>(function_help[i]) << "] " << get<1>(function_help[i]) << endl;
		}
	}
	else if (input.size() == 2) {
		bool found = false;
		for (int i = 0; i < function_help.size(); i++) {
			if (input[1] == get<0>(function_help[i])) {
				cout << "[" << get<0>(function_help[i]) << "] " << get<1>(function_help[i]) << " " << get<2>(function_help[i]) << endl;
				found = true;
				break;
			}
		}
		if (!found) {
			cout << "help for the function \"" << input[1] << "\" was not found" << endl;
			cout << "try the command \"help\" for a list of functions" << endl;
		}
	}
}

void SimpleCommandInterface::view_evaluations(vector<string>& input) {
	cnn.evaluate_random_set(atoi(input[1].c_str()));
}

void SimpleCommandInterface::group_net_test(vector<string>& input){

	int test_count = 13;

	// get all files within the network folder
	vector<string> files = files_in_dir(input[1] + input[2]);
	
	// create an object to use the GPU or CPU used by all networks in the group
	OCLFunctions ocl_functions = OCLFunctions(CL_DEVICE_TYPE_GPU);

	// create and store all networks
	ConvolutionalNeuralNetwork networks[26];
	for (int i = 0; i < files.size(); i++) {
		networks[i] = ConvolutionalNeuralNetwork(input[1] + files[i], ocl_functions, 0);
	}

	// load data samples to test the network group with
	vector<DataSample> data_samples;
	//string input_folder = "data/NIST2/testing/";
	string input_folder = "data/MNIST/testing/";
	load_nist(data_samples, input_folder, input[3], 30, false, networks[0].input_size);

	int correct_count = 0;

	// for the number of samples we want to test with
	for (int i = 0; i < test_count; i++) {

		// pick and random sample and get the scores from each network
		int random_index = random_i(0, data_samples.size());
		int highest_index = 0;
		float highest_probability = 0;

		cv::destroyAllWindows();
		for (int j = 0; j < files.size(); j++) {
			
			// get the probability that the image is the char of this network
			float current_probability = networks[j].highest_probability(data_samples[random_index].image_segment.float_m_mini);
			cout << input[3][j] << " -> " << current_probability << endl;

			// check to see if this new network has a higher score
			if (current_probability > highest_probability) {
				highest_probability = current_probability;
				highest_index = j;
			}
		}

		cout << "The highest score was [" << highest_probability << "] from index [" << highest_index << "] which is [" << input[3][highest_index] << "]" << endl;
		cout << "Corrent index is [" << data_samples[random_index].correct_index << "] which is [" << input[3][data_samples[random_index].correct_index] << "] so the network was [" << (data_samples[random_index].correct_index == highest_index ? "correct" : "incorrect") << "]" << endl << endl;
		if (data_samples[random_index].correct_index == highest_index) {
			correct_count++;
		}
		cv::Mat temp;
		cv::resize(data_samples[random_index].image_segment.m, temp, cv::Size(200, 200));
		cv::imshow("", temp);
		cv::waitKey();
	}
	cv::destroyAllWindows();

	cout << "Finished testing, network was correct [" << (float)correct_count / (float)test_count * 100 << "%] of the time" << endl;

	//////////////////////////////////////////////////////////////////////

	cout << "\nTesting network with noise problems" << endl;
	
	// define paths to files
	vector<string> noise_problem_dirs = {
		"data/MNIST/noise/2_noise.png",
		"data/MNIST/noise/3_noise.png",
		"data/MNIST/noise/4_noise.png",
		"data/MNIST/noise/5_noise.png",
		"data/MNIST/noise/7_noise.png"
	};

	vector<DataSample> samples;
	for (string dir : noise_problem_dirs) {
		samples.emplace_back(DataSample());
		load_sized_data_sample(samples.back(), dir, cv::Size(20, 20));
	}

	for (int i = 0; i < noise_problem_dirs.size(); i++) {

		cv::destroyAllWindows();
		float highest_probability = 0.0f;
		int highest_index = 0;
		for (int j = 0; j < files.size(); j++) {

			// get the probability that the image is the char of this network
			float current_probability = networks[j].highest_probability(samples[i].image_segment.float_m_mini);
			cout << input[3][j] << " -> " << current_probability << endl;

			// check to see if this new network has a higher score
			if (current_probability > highest_probability) {
				highest_probability = current_probability;
				highest_index = j;
			}

		}
		cout << "The highest score was [" << highest_probability << "] from index [" << highest_index << "] which is [" << input[3][highest_index] << "]" << endl << endl;

		cv::Mat temp;
		cv::resize(samples[i].image_segment.m, temp, cv::Size(200, 200));
		cv::imshow("", temp);
		cv::waitKey();
	}
	cv::destroyAllWindows();

}

void SimpleCommandInterface::demo(vector<string>& input){
	if(input.size() > 1){
		
		// trianing example A and !A
		if(input[1] == "1"){
			evaluate_command("setseed 0");
			evaluate_command("loadnet data/cnn_json/single_char_net.json");
			evaluate_command("loadset training data/NIST2/training/ 1000 A ABCDEFGHIJKLMNOPQRSTUVWXYZ");
			evaluate_command("loadset testing data/NIST2/testing/ 200 A ABCDEFGHIJKLMNOPQRSTUVWXYZ");
			evaluate_command("setiteration 500");
			evaluate_command("setevaluation threshold");
			evaluate_command("trainnet test1/DEMO_A.dat 100 200");
			evaluate_command("savenet data/DEMO_A_SAVE.json");
			evaluate_command("viewgraph demo_a_plot.plt");
		} 
		
		else if(input[1] == "2"){
			evaluate_command("setseed 0");
			evaluate_command("testgroupnet data/cnn_json/mnum_locmax2/ *.json 0123456789");
		}

		else if(input[1] == "3"){

			// load an image, clean and segment it
			cv::Mat image = cv::imread("data/WHITEBOARD/convolution.png", cv::IMREAD_GRAYSCALE);
			vector<ImageSegment> segments;
			cv::Mat source = image.clone();
			GaussianBlur(source, image, cv::Size(3, 3), 5);
			adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 101, 5);
			segment_image_islands(image, segments);
			//bitwise_not ( image, image );
			
			// create DataSample objects from segments
			vector<DataSample> samples;
			for(int i = 0; i < segments.size(); i++){
				samples.emplace_back(DataSample());
				bitwise_not ( segments[i].m, segments[i].m );
				load_sized_data_sample(samples.back(), segments[i].m, cv::Size(20, 20));
			}

			// get all files within the network folder
			vector<string> files = files_in_dir("data/cnn_json/mchar_locmax2/*.json");
	
			// create an object to use the GPU or CPU used by all networks in the group
			OCLFunctions ocl_functions = OCLFunctions(CL_DEVICE_TYPE_GPU);

			// create and store all networks
			ConvolutionalNeuralNetwork networks[26];
			for (int i = 0; i < files.size(); i++) {
				networks[i] = ConvolutionalNeuralNetwork("data/cnn_json/mchar_locmax2/" + files[i], ocl_functions, 0);
			}

			string mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
			string output = "";

			for(int i = 0; i < samples.size(); i++){
				float highest_probability = 0.0f;
				int highest_index = 0;
				for (int j = 0; j < files.size(); j++) {

					// get the probability that the image is the char of this network
					float current_probability = networks[j].highest_probability(samples[i].image_segment.float_m_mini);
					//cout << mapping[j] << " -> " << current_probability << endl;

					// check to see if this new network has a higher score
					if (current_probability > highest_probability) {
						highest_probability = current_probability;
						highest_index = j;
					}

				}
				//cout << "The highest score was [" << highest_probability << "] from index [" << highest_index << "] which is [" << mapping[highest_index] << "]" << endl << endl;
				output += mapping[highest_index];
				//cout << "so far... [" << output << "]\n" << endl;
			}

			cout << "network read [" << output << "]" << endl;
		}
	}
}

void SimpleCommandInterface::error_message(string function) {
	cout << "command for function \"" << function << "\" was not understood" << endl;
	cout << "try the command \"help\" or \"help " << function << "\"" << endl;
}

boolean SimpleCommandInterface::file_exists(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}

boolean SimpleCommandInterface::is_number(const std::string& s){
	string::const_iterator it = s.begin();
	while (it != s.end() && isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}

void SimpleCommandInterface::create_template(string file_name) {
	ifstream src("data/cnn_json/network_default_dont_delete.json", ios::binary);
	ofstream dst(file_name, ios::binary);
	dst << src.rdbuf();
}

void SimpleCommandInterface::load_nist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, bool random, cv::Size network_input_size) {

	// variables that we will use to read directories
	HANDLE hFind;
	WIN32_FIND_DATA data;

	// for how many numbers we should train on
	for (int i = 0; i < num_string.size(); i++) {

		// form a folder to look at
		string f = folder + num_string[i] + "/" + "*.png";
		int count = 0;

		// search for all files in this dir that are png
		cout << "loading [" << sample_count << "] images of the character [" << num_string[i] << "] searching [" << f << "]" << endl;

		hFind = FindFirstFile(f.c_str(), &data);
		if (hFind != INVALID_HANDLE_VALUE) {

			// while there is another file
			do {
				DataSample temp_data;

				// store which index is correct
				temp_data.correct_index = i;

				// if the call is from a single problem set then this data is the random data and all should have a 0 answer
				if (random) {
					temp_data.answer.emplace_back(0);
				}

				// the data being loaded will be used for a multiple problem network
				else {
					// create the answer for this image
					for (int k = 0; k < num_string.size(); k++) {
						if (k == i) {
							temp_data.answer.emplace_back(1);
						}
						else {
							temp_data.answer.emplace_back(0);
						}
					}
				}

				// create the data sample and load the image to the data sample add it to the pool of samples
				load_sized_data_sample(temp_data, folder + num_string[i] + "/" + data.cFileName, network_input_size);
				data_samples.emplace_back(temp_data);

				count++;

			} while (FindNextFile(hFind, &data) && count < sample_count);

			// close the handle on the file we found
			FindClose(hFind);
		}
	}
}
