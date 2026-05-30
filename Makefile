CC = gcc 
CFLAGS = -Wall -Wextra -pedantic -O3
SRCS = $(shell find src -name '*.c')
OBJS = $(SRCS:%.c=%.o)
INCLUDE_DIRS = $(shell find src -type d)
INCLUDES = $(addprefix -I,$(INCLUDE_DIRS))

LD = gcc 
LDFLAGS = -lm

TARGET = ./main

.PHONY: all
all: $(TARGET)

.PHONY: run
run: $(TARGET)
	$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(LD) $(OBJS) -o $(TARGET) $(LDFLAGS)
