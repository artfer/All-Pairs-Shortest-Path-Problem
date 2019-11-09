CC = mpicc
CFLAGS = -lm
SRCS = src/fox.c
EXEC = fox

all:
	$(CC) $(SRCS) $(CFLAGS) -o $(EXEC)

clean:
	rm -f $(EXEC)