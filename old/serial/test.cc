#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ncurses.h>
#include "serialUtils.h"

using namespace std;


#define KEY_UP 259
#define KEY_DOWN 258
#define KEY_LEFT 260
#define KEY_RIGHT 261

int main(int argc, char** argv) {
	/* curses init */
	initscr();
	keypad(stdscr, TRUE);
	noecho();

	int port = serialPortOpen();
	if(!start(port)) printf("failed to open port");
	int c = 0;
	while(1)
	{
		c = 0;

		switch((c=getch())) {
			case KEY_UP:
				set_speed(port, "128");
				break;
			case KEY_DOWN:
				set_speed(port, "0");
				break;
			case KEY_LEFT:
				turn(port, "135");
				break;
			case KEY_RIGHT:
				turn(port, "45");
				break;
			default:
				set_speed(port, "0");
				turn(port, "90");
				break;
		}

	}
	serialPortClose(port);
}
