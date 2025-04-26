# Compiler
CC =gcc 
NVCC =/usr/local/cuda/bin/nvcc

# Compiler flags

#Load Libraries
LDLIBS = -I/usr/include/opencv4 -ljpeg -lm -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -llapacke -llapack -lblas -lcuda -lopencv_videoio -lncurses

# Source files
CCSRCS = test.cc serialUtils.cc 

# Object files (replace .cc with .o)
CCOBJS = $(CCSRCS:.cc=.o)

# Output executable
TARGET = test 

# Rule to build the final executable
$(TARGET): $(CCOBJS) $(CUOBJS)
	$(NVCC) $(CUFLAGS) -o $(TARGET) $(CCOBJS) $(CUOBJS) $(LDLIBS)

# Rule to build cc object files
%.o: %.cc 
	$(CC) $(CCFLAGS) -c $< -o $@

# Rule to build cu object files
%.o: %.cu 
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(CCOBJS) $(CUOBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

