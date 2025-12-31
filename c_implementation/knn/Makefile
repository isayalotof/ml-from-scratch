CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LIBS = -lm

# Source files
UTILS = ../utils/matrix.c ../utils/data_loader.c
SOURCES = knn.c main.c $(UTILS)

# Output executable
TARGET = knn_classifier

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run