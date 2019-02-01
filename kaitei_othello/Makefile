CXX = g++
TARGET = kaitei_othello
CXXFLAGS = -O3 -std=c++17 -march=native -I../eigen/
LDFLAGS = -pthread -lstdc++fs
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp = .o)

$(TARGET) : $(OBJS)
$(CXX) -o $@ $(OBJS) $(LDFLAGS)

all : $(OBJS)

clean :
	rm - f $(TARGET) $(OBJS)