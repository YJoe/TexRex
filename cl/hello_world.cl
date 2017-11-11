__kernel void hello(__global char* str){
	str[0] = 'H';
	str[1] = 'e';
	str[2] = 'l';
	str[3] = 'l';
	str[4] = 'o';
	str[5] = ',';
	str[6] = ' ';
	str[7] = 'W';
	str[8] = 'o';
	str[9] = 'r';
	str[10] = 'l';
	str[11] = 'd';
	str[12] = '!';
	str[13] = '\0';
}