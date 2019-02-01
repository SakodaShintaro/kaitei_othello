CXX = g++
TARGET = kaitei
CXXFLAGS = -O3 -std=c++17 -march=native -I ../../eigen/ -w
LDFLAGS = -pthread -lstdc++fs
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

$(TARGET) : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

all : $(OBJS)

clean:
	rm -f $(TARGET) $(OBJS)
