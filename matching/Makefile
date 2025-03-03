# Compiler
CC = g++ 

# Compiler flags
CFLAGS = -Wall

#Load Libraries
LDLIBS = -ljpeg -lm

# Source files
SRCS = stereo.cc imageUtils.cc utils.cc matrixUtils.cc readParams.cc readData.cc

# Object files (replace .cc with .o)
OBJS = $(SRCS:.cc=.o)

# Output executable
TARGET = stereo 

# Rule to build the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)

# Rule to build object files
%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

