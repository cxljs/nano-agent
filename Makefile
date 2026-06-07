.PHONY: run build clean test test-live

run:
	go run .

build:
	go build -o duck .

clean:
	rm -f duck

# Fast suite: tools + agent-with-mock-LLM. No tokens spent.
# -count=1 disables the test result cache so reruns actually rerun.
test:
	go test -race -count=1 -v ./...

# Live end-to-end test.
test-live:
	RUN_LIVE_AGENT_TESTS=1 go test -v -count=1 -run TestLive ./...
