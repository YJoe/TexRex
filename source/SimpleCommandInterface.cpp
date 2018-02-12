#include "..\include\SimpleCommandInterface.h"

SimpleCommandInterface::SimpleCommandInterface(){
	function_help = {
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("help", "<function name>", "will provide some help on a function", &SimpleCommandInterface::help),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("loadnet", "<file location>", "loads an already defined json network from the given directory", &SimpleCommandInterface::loadnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("createnet", "<file location>", "creates and loads a json network from a simple template", &SimpleCommandInterface::createnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("formatset", "<input folder> <output folder> <image width> <image height>", "resizes a training set found in a folder", &SimpleCommandInterface::formatset),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("loadset", "<input folder> <sample size> <representation>", "loads a training set from a given directory", &SimpleCommandInterface::loadset),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("trainnet", "<data output> <classification sample size>", "begins the training proccess and logs into the given data file", &SimpleCommandInterface::trainnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("testnet", "", "begins the testing proccess for a loaded network and data set", &SimpleCommandInterface::testnet),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("setiteration", "<number of iterations>", "sets the network terminating condition to iteration with the number given", &SimpleCommandInterface::setiteration),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("viewgraph", "<plt file>", "displays a plot of the latest data file", &SimpleCommandInterface::viewgraph),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("setseed", "\"random\" or <number>", "sets the seed for which to make random choicess", &SimpleCommandInterface::setseed),
		tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>("savenet", "<save location>", "saves the network structure and weights", &SimpleCommandInterface::savenet)
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
	else if(input != ""){
		handle_input(regex_split(input));
		cout << endl;
		return true;
	}

	// only in the case that input != ""
	cout << endl;
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

void SimpleCommandInterface::handle_input(vector<string>& input) {

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

void SimpleCommandInterface::formatset(vector<string>& input) {
	cout << "formatting set not yet complete" << endl;
}

void SimpleCommandInterface::loadset(vector<string>& input) {

	if (input.size() == 4) {
		if (is_number(input[2])) {
			// generate the training set
			vector<DataSample> data_samples;
			load_nist(data_samples, input[1], input[3], atoi(input[2].c_str()), cnn.input_size);

			// creating mapping
			vector<char> mapping;
			for (int i = 0; i < input[3].size(); i++) {
				mapping.emplace_back(input[3][i]);
			}

			cnn.setTrainingSamples(data_samples);
			cnn.setMapping(mapping);
		}
		else {
			cout << "error - [" << input[2] << "] is not a number" << endl;
		}
	}
	else {
		error_message(input[0]);
	}
}

void SimpleCommandInterface::trainnet(vector<string>& input) {
	if (input.size() == 3) {
		if (cnn.is_defined) {
			if (is_number(input[2])) {
				ofstream data_file_clear(input[1]);
				data_file_clear.close();
				ofstream data_file(input[1], ios::app);
				if (!data_file.is_open()) {
					cout << "error - failed to create data logging file, try another location" << endl;
					return;
				}

				cout << "traing network and data will log to [" << input[1] << "] and will sample the classification rate every [" << input[2] << "] inputs" << endl;
				cnn.train(data_file, atoi(input[2].c_str()));
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
	if (input.size() == 1) {
		cnn.test();
	}
	else {
		error_message(input[0]);
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

void SimpleCommandInterface::load_nist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, cv::Size network_input_size) {

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

				// create the answer for this image
				for (int k = 0; k < num_string.size(); k++) {
					if (k == i) {
						temp_data.answer.emplace_back(1);
					}
					else {
						temp_data.answer.emplace_back(0);
					}
				}

				// construct the file name and load the image
				temp_data.image_segment.m = cv::imread(folder + num_string[i] + "/" + data.cFileName, cv::IMREAD_GRAYSCALE);

				// scale the image to meet the network input sized and convert to float
				resize(temp_data.image_segment.m, temp_data.image_segment.m, network_input_size);
				temp_data.image_segment.m.convertTo(temp_data.image_segment.m, CV_32FC1, 1.0 / 255.0);

				// store the image as a vector of floats and release the image to save space
				get_vector(temp_data.image_segment.m, temp_data.image_segment.float_m_mini);
				data_samples.emplace_back(temp_data);

				count++;

			} while (FindNextFile(hFind, &data) && count < sample_count);

			// close the handle on the file we found
			FindClose(hFind);
		}
	}
}
