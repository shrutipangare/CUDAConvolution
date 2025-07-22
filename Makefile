SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64
INCLUDE := -I$(SDK_INSTALL_PATH)/include
OPTIONS   :=  -lcudnn

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS := convolve
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt *.cpp

#######################################################################
convolve: convolve.cu
	${NVCC} -o convolve convolve.cu $(LIB) $(OPTIONS)