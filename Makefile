.PHONY: run build clean

run:
	go run .

build:
	go build -o nano-agent .

clean:
	rm -f nano-agent
