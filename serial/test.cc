#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "serialUtils.h"

using namespace std;



int main(int argc, char** argv) {

	int port = serialPortOpen();
	if(!start(port)) printf("failed to open port");
	if(!turn(port, "40")) printf("failed to turn");
	serialPortClose(port);

}

