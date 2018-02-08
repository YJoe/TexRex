#include "..\include\SimpleCommandInterface.h"

SimpleCommandInterface::SimpleCommandInterface(){
}

void SimpleCommandInterface::begin() {
	cout << "========================" << endl;
	cout << "Simple Network Interface" << endl;
	cout << "========================" << endl;

	// while the user has something to enter
	string input = "";
	boolean running = true;
	while (running) {

		// get the user input
		cout << "input -> ";
		getline(cin, input);

		// split user data by regex and dsplay seperated command
		vector<string> split_data = regex_split(input, "[ ]");
		string pattern_code = "";
		for (const auto & e : split_data) {
			pattern_code += get_type_code(e);
		}

		// run the command or exit the interface
		input == "exit" ? running = false : handle_input(split_data, pattern_code);
		cout << endl;
	}

	// exit the interface
	cout << "closing" << endl;
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

char SimpleCommandInterface::get_type_code(string input) {
	
	// define regex rules to identify patterns in lines
	vector<pair<regex, char>> regex_patterns = {
		pair<regex, char>("(\\+|-)?[[:digit:]]+", 'I'),
		pair<regex, char>("((\\/?[a-zA-Z_-])+\\.[a-zA-Z]+)|(\\/?([a-zA-Z].*\\.+|[a-zA-Z].*\.\\/))", 'D'),
		pair<regex, char>("[A-Z|a-z]+", 'S')
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

void SimpleCommandInterface::handle_input(vector<string>& input, string pattern_code) {

	// all patterns and their respective function pointers
	vector<pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>> valid_patterns = {
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("S", &SimpleCommandInterface::handle_s),
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("SD", &SimpleCommandInterface::handle_sd),
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("SS", &SimpleCommandInterface::handle_ss),
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("SDIS", &SimpleCommandInterface::handle_sdis),
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("SDII", &SimpleCommandInterface::handle_sdii),
		pair<string, void (SimpleCommandInterface::*)(vector<string>& s)>("SDDII", &SimpleCommandInterface::handle_sddii)
	};

	// match the pattern to a function
	boolean matched = false;
	for (int i = 0; i < valid_patterns.size(); i++) {
		if (valid_patterns[i].first == pattern_code) {
			matched = true;
			(this->*(valid_patterns[i].second))(input);
			break;
		}
	}

	// we didn't find a match so do nothing
	if (!matched) {
		cout << "pattern template not found for [" << pattern_code << "] try \"help\"" << endl;
	}
}

void SimpleCommandInterface::handle_s(vector<string>& input) {
	if (input[0] == "help") {
		cout << "try \"help\" on any of the following function names" << endl;
		cout << "\nfunctions:" << endl;
		cout << "\tloadnet <file location>\n\t\tloads an already defined json network" << endl;
		cout << "\n\tcreatenet <file location>\n\t\tcreates a json network from a simple template" << endl;
		cout << "\n\tformatset <input folder> <output folder> <image width> <image height>\n\t\tresizes images found in a folder" << endl;
		cout << "\n\tloadset <input folder> <sample size> <representation>\n\t\tresizes images found in a folder" << endl;
		cout << "\n\ttrainnet <output data file>\n\t\tbegins the training proccess of a loaded network and set" << endl;
		cout << "\n\ttestnet\n\t\tbegins the testing proccess of a loaded network and set" << endl;
	}
	else if (input[0] == "about"){
		cout << "created by Joe Pauley, blah blah write more here" << endl;
	}
	else if (input[0] == "testnet"){
		cout << "testing network" << endl;
		cnn.test();
	} 
	else if (input[0] == "viewnet") {
		if (cnn.is_defined) {
			ShellExecute(0, 0, cnn.this_net_dir.c_str(), 0, 0, SW_SHOW);
		}
	}
	else if (input[0] == "hi") {
		cout << "howdy :)" << endl;
	}
	else {
		cout << "pattern [S] must have a first argument of [\"help\" | \"about\"] try \"help\"" << endl;
	}
}

void SimpleCommandInterface::handle_sd(vector<string>& input){ 
	if (input[0] == "loadnet") {
		if (file_exists(input[1])) {
			OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
			cnn = ConvolutionalNeuralNetwork(input[1], ocl, 1);
		}
		else {
			cout << "file [" << input[1] << "] doesn't exist" << endl;
		}
	}
	else if(input[0] == "createnet"){
		create_template(input[1]);
		OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
		cnn = ConvolutionalNeuralNetwork(input[1], ocl, 1);
	}
	else if(input[0] == "viewnet") {
		//TODO: make this work
		ShellExecute(0, 0, input[1].c_str(), "C:\\Program Files\\Sublime Text 3\\sublime_text.exe", 0, SW_SHOW);
	}
	else if (input[0] == "trainnet") {
		if (cnn.is_defined) {
			ofstream data_file_clear(input[1]);
			data_file_clear.close();
			ofstream data_file(input[1], ios::app);
			if (data_file.is_open()) {
				data_file << "# NETWORK LEARNING DATA\n";
			}
			else {
				cout << "failed to create data logging file, quitting" << endl;
				cin.get();
				exit(-1);
			}

			cout << "traing network and data will log to [" << input[1] << "]" << endl;
			cnn.train(data_file);
		}
	}
	else {
		cout << "pattern [SD] must have a firt argument of [\"loadnet\" | \"createnet\" | \"trainnet\"] try \"help <loadnet|createnet|trainnet>\" for info" << endl;
	}
}

void SimpleCommandInterface::handle_ss(vector<string>& input) {
	if (input[0] == "help") {
		if (input[1] == "loadnet") {
			cout << "loadnet <file location>\n\tloads an already defined json network" << endl;
			cout << "examples\n\t\"loadnet some/file/directory/file.json\"\n\t\"loadnet file.json\"" << endl;
		}
		else if (input[1] == "createnet") {
			cout << "createnet <file location>\n\tcreates a json network from a simple template" << endl;
			cout << "examples\n\t\"createnet some/file/directory/file.json\"\n\t\"loadnet file.json\"" << endl;
			cout << "help on createnet" << endl;
		}
		else if (input[1] == "formatset") {
			cout << "help on formatset" << endl;
		}
		else if (input[1] == "loadset") {
			cout << "help on loadset" << endl;
		} 
		else if (input[1] == "trainnet") {
			cout << "help on trainnet" << endl;
		}
		else if (input[1] == "help") {
			cout << "that's silly" << endl;
		}
		else if (input[1] == "about") {
			cout << "that's silly" << endl;
		}
		else {
			cout << "there is no function called [" << input[1] << "] try \"help\" for a list of functions" << endl;
		}
	}
	else if(input[0] == "setmap") {
		if (cnn.is_defined) {
			cout << "setting network mapping to [" << input[1] << "]" << endl;
			vector<char> mapping;
			for (int i = 0; i < input[1].size(); i++) {
				mapping.emplace_back(input[1][i]);
			}
			cnn.setMapping(mapping);
		}
		else {
			cout << "define the network first! try \"help createnet\" or \"help loadnet\"" << endl;
		}
	}
	else {
		cout << "pattern [SS] must have first argument of [\"help\"] try \"help\" or \"help <a function>\"" << endl;
	}
}

void SimpleCommandInterface::handle_sdii(vector<string>& input){
	if (input[0] == "loadset") {

		// generate the training set
		vector<DataSample> data_samples;
		load_mnist(data_samples, input[1], input[3], input[3].size(), cnn.input_size);

		// creating mapping
		vector<char> mapping;
		for (int i = 0; i < input[3].size(); i++) {
			mapping.emplace_back(input[3][i]);
		}

		cout << "Setting training samples" << endl;
		cnn.setTrainingSamples(data_samples);

		cout << "Setting training mapping" << endl;
		cnn.setMapping(mapping);
	}
}

void SimpleCommandInterface::handle_sdis(vector<string>& input) {

	if (input[0] == "loadset") {

		// generate the training set
		vector<DataSample> data_samples;
		load_mnist(data_samples, input[1], input[3], input[3].size(), cnn.input_size);

		// creating mapping
		vector<char> mapping;
		for (int i = 0; i < input[2].size(); i++) {
			mapping.emplace_back(input[2][i]);
		}

		cout << "Setting training samples" << endl;
		cnn.setTrainingSamples(data_samples);

		cout << "Setting training mapping" << endl;
		cnn.setMapping(mapping);
	}
}

void SimpleCommandInterface::handle_sddii(vector<string>& input) {
	if (input[0] == "formatset") {
		cout << "formatting set" << endl;
	}
}

boolean SimpleCommandInterface::file_exists(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}

void SimpleCommandInterface::create_template(string file_name) {
	ifstream src("data/cnn_json/network_default_dont_delete.json", ios::binary);
	ofstream dst(file_name, ios::binary);
	dst << src.rdbuf();
}

void SimpleCommandInterface::load_mnist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, cv::Size network_input_size) {

	// variables that we will use to read directories
	HANDLE hFind;
	WIN32_FIND_DATA data;

	// for how many numbers we should train on
	for (int i = 0; i < num_string.size(); i++) {

		cout << "loading [" << sample_count << "] letter [" << num_string[i] << "] files" << endl;

		// form a folder to look at
		string f = folder + num_string[i] + "/" + "*.png";
		int count = 0;

		// search for all files in this dir that are png
		cout << "searching [" << f << "]" << endl;
		hFind = FindFirstFile(f.c_str(), &data);
		if (hFind != INVALID_HANDLE_VALUE) {

			// while there is another file
			do {
				cout << "\t" << data.cFileName << endl;

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
