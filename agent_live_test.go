package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Live end-to-end test against a real Anthropic-compatible API. It is gated
// on RUN_LIVE_AGENT_TESTS=1 because it costs tokens; running it in CI by
// accident would be both flaky (model variance) and noisy.
//
// Configuration (base URL, API key, model) is resolved the same way main()
// does — via loadSettings(): ~/.claude/settings.json, then
// ./.claude/settings.json, then ANTHROPIC_* environment variables override
// either. So locally just:
//
//	RUN_LIVE_AGENT_TESTS=1 go test -v -run TestLive
//
// will pick up whatever the binary itself would use.
//
// The test asks the agent to perform a small, deterministic task that
// requires reading a file. We assert on a sentinel string the agent must
// have actually read from disk to surface — guessing or hallucinating
// cannot produce it. We deliberately avoid asserting on the wording around
// the sentinel so the test is robust to normal model variance.

func TestLive_AgentReadsFileWithRealModel(t *testing.T) {
	if os.Getenv("RUN_LIVE_AGENT_TESTS") != "1" {
		t.Skip("set RUN_LIVE_AGENT_TESTS=1 to run the live agent test")
	}

	// Resolve settings BEFORE chdir: loadSettings looks for a project-local
	// ./.claude/settings.json relative to cwd, and we don't want the tmp
	// sandbox to mask the real project's config.
	baseURL, apiKey, model := loadSettings()
	if apiKey == "" {
		t.Skip("no API key found via loadSettings (~/.claude/settings.json, .claude/settings.json, or ANTHROPIC_AUTH_TOKEN); skipping")
	}

	dir := t.TempDir()
	chdir(t, dir)

	// Plant a unique sentinel so we can detect the model actually read the
	// file rather than guessing the token shape.
	const sentinel = "HIPPO-7392-XYZZY"
	mustWrite(t, filepath.Join(dir, "secret.txt"), "the secret token is "+sentinel+"\n")

	opts := []option.RequestOption{}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	opts = append(opts, option.WithAPIKey(apiKey))
	client := anthropic.NewClient(opts...)

	// Single-shot user input then EOF — the agent loop exits cleanly once
	// the model stops issuing tool_use blocks.
	input := scriptedUserInput("Read secret.txt and tell me the token it contains. Output just the token.")

	captured := newStdoutCapture(t)
	defer captured.restore()

	agent := NewAgent(&client, input, model, Tools)
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()

	if err := agent.Run(ctx); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	out := captured.flush()
	if !strings.Contains(out, sentinel) {
		t.Errorf("model output does not contain the planted sentinel %q; got:\n%s", sentinel, out)
	}
}

// stdoutCapture replaces os.Stdout with a pipe so we can read what the
// agent printed (Claude responses, tool invocations). It also tees the
// stream back to the original stdout so a human watching the test still
// sees the conversation in real time.
type stdoutCapture struct {
	orig     *os.File
	writer   *os.File
	done     chan struct{}
	buf      strings.Builder
	restored bool
}

func newStdoutCapture(t *testing.T) *stdoutCapture {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	c := &stdoutCapture{
		orig:   os.Stdout,
		writer: w,
		done:   make(chan struct{}),
	}
	os.Stdout = w
	go func() {
		defer close(c.done)
		var chunk [4096]byte
		for {
			n, rerr := r.Read(chunk[:])
			if n > 0 {
				c.buf.Write(chunk[:n])
				_, _ = c.orig.Write(chunk[:n])
			}
			if rerr != nil {
				return
			}
		}
	}()
	return c
}

func (c *stdoutCapture) restore() {
	if c.restored {
		return
	}
	c.restored = true
	// Close the writer so the goroutine sees EOF and exits before callers
	// observe c.buf.
	_ = c.writer.Close()
	<-c.done
	os.Stdout = c.orig
}

func (c *stdoutCapture) flush() string {
	c.restore()
	return c.buf.String()
}
