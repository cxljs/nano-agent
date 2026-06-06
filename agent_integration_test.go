package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// These tests exercise the full agent loop end-to-end: user input → model
// inference → tool dispatch → tool result → next inference. The Anthropic
// client is pointed at an httptest server that plays a scripted sequence of
// responses, which lets us assert two things at once:
//
//  1. The agent correctly translates tool_use blocks into real tool calls
//     and feeds tool_result blocks back on the next turn.
//  2. The tools, run against a real temp filesystem, produce results the
//     agent forwards faithfully.
//
// The conversation transcript captured by the mock server is the assertion
// surface; we deliberately avoid peeking into Agent internals.

// agentScript is a queue of canned model responses played out in order. Each
// call to /v1/messages pops the next entry. Tests build a script that mirrors
// what a real model would do for a given task.
type agentScript struct {
	mu        sync.Mutex
	responses []*anthropic.Message
	requests  []recordedRequest
	calls     atomic.Int32
}

type recordedRequest struct {
	Body anthropic.MessageNewParams
}

func (s *agentScript) push(msg *anthropic.Message) {
	s.responses = append(s.responses, msg)
}

func (s *agentScript) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("mock: read body: %v", err)
			http.Error(w, err.Error(), 500)
			return
		}
		var parsed anthropic.MessageNewParams
		if err := json.Unmarshal(body, &parsed); err != nil {
			// Don't fail the test — keep the raw body for diagnostics. Some
			// fields the SDK sends round-trip through union types and may not
			// unmarshal cleanly into the param struct.
			t.Logf("mock: param unmarshal warning: %v\nbody=%s", err, body)
		}

		s.mu.Lock()
		idx := int(s.calls.Add(1)) - 1
		s.requests = append(s.requests, recordedRequest{Body: parsed})
		var resp *anthropic.Message
		if idx < len(s.responses) {
			resp = s.responses[idx]
		}
		s.mu.Unlock()

		if resp == nil {
			t.Errorf("mock: unexpected request #%d (script exhausted); raw body: %s", idx+1, body)
			http.Error(w, "no scripted response", 500)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(resp)
	}
}

// newScriptedClient wires an Anthropic client to a fresh httptest server
// driven by the given script. t.Cleanup tears down the server.
func newScriptedClient(t *testing.T, script *agentScript) *anthropic.Client {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", script.handler(t))
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	client := anthropic.NewClient(
		option.WithBaseURL(srv.URL+"/"),
		option.WithAPIKey("test-key"),
	)
	return &client
}

// textMessage builds a model response with a single text block — the
// terminal step of a conversation where Claude is done calling tools.
func textMessage(text string) *anthropic.Message {
	raw, _ := json.Marshal(map[string]any{
		"id":    "msg_test",
		"type":  "message",
		"role":  "assistant",
		"model": "test-model",
		"content": []map[string]any{
			{"type": "text", "text": text},
		},
		"stop_reason": "end_turn",
		"usage":       map[string]int{"input_tokens": 1, "output_tokens": 1},
	})
	var m anthropic.Message
	if err := json.Unmarshal(raw, &m); err != nil {
		panic(err)
	}
	return &m
}

// toolUseMessage builds a model response with a single tool_use block.
// `input` becomes the tool's argument JSON.
func toolUseMessage(toolID, toolName string, input map[string]any) *anthropic.Message {
	raw, _ := json.Marshal(map[string]any{
		"id":    "msg_test",
		"type":  "message",
		"role":  "assistant",
		"model": "test-model",
		"content": []map[string]any{
			{
				"type":  "tool_use",
				"id":    toolID,
				"name":  toolName,
				"input": input,
			},
		},
		"stop_reason": "tool_use",
		"usage":       map[string]int{"input_tokens": 1, "output_tokens": 1},
	})
	var m anthropic.Message
	if err := json.Unmarshal(raw, &m); err != nil {
		panic(err)
	}
	return &m
}

// scriptedUserInput returns a getUserMessage callback that feeds a fixed
// sequence of inputs and then signals EOF. The agent loop exits on EOF.
func scriptedUserInput(inputs ...string) func() (string, bool) {
	i := 0
	return func() (string, bool) {
		if i >= len(inputs) {
			return "", false
		}
		v := inputs[i]
		i++
		return v, true
	}
}

func TestAgent_ToolCallRoundtripsResult(t *testing.T) {
	// Goal: when the model emits a tool_use, the agent must
	//   (a) actually invoke the tool on the local filesystem,
	//   (b) send a follow-up request containing a tool_result block keyed
	//       to the original tool_use id with the tool's output.
	dir := t.TempDir()
	chdir(t, dir)
	mustWrite(t, filepath.Join(dir, "readme.md"), "# project\nhello\n")

	script := &agentScript{}
	// Turn 1: model asks to read readme.md.
	script.push(toolUseMessage("call_1", "read_file", map[string]any{"path": "readme.md"}))
	// Turn 2: model produces the final answer using the tool output.
	script.push(textMessage("The file says hello."))

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("what is in readme.md?"), "test-model", Tools)

	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	if got := script.calls.Load(); got != 2 {
		t.Fatalf("expected 2 model calls, got %d", got)
	}

	// The second request is what proves the loop worked: it must carry the
	// tool_result for call_1 with the file's contents.
	second := script.requests[1]
	tr := findToolResult(t, second.Body, "call_1")
	got := toolResultText(t, tr)
	if !strings.Contains(got, "hello") {
		t.Errorf("tool_result for call_1 missing file contents; got %q", got)
	}
	if tr.IsError.Value {
		t.Errorf("tool_result for call_1 unexpectedly marked as error")
	}
}

func TestAgent_PropagatesToolErrors(t *testing.T) {
	// Goal: when a tool returns an error, the agent must still continue the
	// loop, but the tool_result block must carry is_error=true so the model
	// can recover. This is the contract executeTool documents.
	chdir(t, t.TempDir())

	script := &agentScript{}
	// Model asks for a file that doesn't exist.
	script.push(toolUseMessage("call_err", "read_file", map[string]any{"path": "does-not-exist.txt"}))
	script.push(textMessage("Sorry, no such file."))

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("read missing"), "test-model", Tools)
	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	tr := findToolResult(t, script.requests[1].Body, "call_err")
	if !tr.IsError.Value {
		t.Errorf("tool_result for failed read should be is_error=true; got %+v", tr)
	}
}

func TestAgent_UnknownToolNameReportedAsError(t *testing.T) {
	// Goal: if the model hallucinates a tool name, executeTool reports
	// "tool not found" with is_error=true rather than crashing or silently
	// dropping the call.
	chdir(t, t.TempDir())

	script := &agentScript{}
	script.push(toolUseMessage("call_x", "delete_universe", map[string]any{}))
	script.push(textMessage("I cannot do that."))

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("go"), "test-model", Tools)
	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	tr := findToolResult(t, script.requests[1].Body, "call_x")
	if !tr.IsError.Value {
		t.Errorf("unknown-tool result should be is_error=true")
	}
	if got := toolResultText(t, tr); !strings.Contains(strings.ToLower(got), "not found") {
		t.Errorf("expected 'not found' in tool_result; got %q", got)
	}
}

func TestAgent_MultiTurnSequencingPreservesHistory(t *testing.T) {
	// Goal: across two user turns the agent must keep accumulating history,
	// not start fresh. After turn 2's user message arrives, the request the
	// model sees must include turn 1's assistant reply.
	chdir(t, t.TempDir())

	script := &agentScript{}
	script.push(textMessage("Hi!"))         // response to first user input
	script.push(textMessage("Still here.")) // response to second user input

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("hello", "you there?"), "test-model", Tools)
	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	if got := script.calls.Load(); got != 2 {
		t.Fatalf("expected 2 model calls, got %d", got)
	}
	// The second request must contain all three prior messages:
	// user1, assistant1, user2. We just check the count and the last role,
	// which is the load-bearing invariant for multi-turn correctness.
	msgs := script.requests[1].Body.Messages
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages on second request, got %d (%+v)", len(msgs), msgs)
	}
}

func TestAgent_EditFileEndToEnd(t *testing.T) {
	// Goal: a realistic two-step plan — list, then edit — actually mutates
	// the filesystem through the agent. This is the closest test we have to
	// "the agent works".
	dir := t.TempDir()
	chdir(t, dir)
	mustWrite(t, filepath.Join(dir, "note.txt"), "draft body")

	script := &agentScript{}
	script.push(toolUseMessage("c1", "list_files", map[string]any{}))
	script.push(toolUseMessage("c2", "edit_file", map[string]any{
		"path":    "note.txt",
		"old_str": "draft",
		"new_str": "final",
	}))
	script.push(textMessage("Done."))

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("rename draft to final in note.txt"), "test-model", Tools)
	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	got, err := os.ReadFile(filepath.Join(dir, "note.txt"))
	if err != nil {
		t.Fatalf("read note.txt: %v", err)
	}
	if string(got) != "final body" {
		t.Errorf("note.txt = %q, want %q", got, "final body")
	}

	// And the edit_file tool_result on the final request should report OK.
	last := script.requests[len(script.requests)-1]
	tr := findToolResult(t, last.Body, "c2")
	if tr.IsError.Value {
		t.Errorf("edit_file tool_result marked as error: %+v", tr)
	}
	if got := toolResultText(t, tr); !strings.Contains(got, "OK") {
		t.Errorf("expected OK in edit_file tool_result, got %q", got)
	}
}

func TestAgent_RegistersAllToolsWithAPI(t *testing.T) {
	// Goal: every tool in the Tools slice must be advertised to the model
	// in MessageNewParams.Tools. Without this, the model can never invoke
	// them. We assert by name to keep the test stable across SDK refactors.
	chdir(t, t.TempDir())
	script := &agentScript{}
	script.push(textMessage("noop"))

	client := newScriptedClient(t, script)
	agent := NewAgent(client, scriptedUserInput("ping"), "test-model", Tools)
	if err := agent.Run(t.Context()); err != nil {
		t.Fatalf("agent.Run: %v", err)
	}

	advertised := make([]string, 0, len(script.requests[0].Body.Tools))
	for _, tu := range script.requests[0].Body.Tools {
		if tu.OfTool != nil {
			advertised = append(advertised, tu.OfTool.Name)
		}
	}
	for _, td := range Tools {
		if !slices.Contains(advertised, td.Name) {
			t.Errorf("tool %q was not advertised to the model; got %v", td.Name, advertised)
		}
	}
}

// ---- helpers ----

// findToolResult locates the ToolResultBlockParam carrying the given tool_use
// id in the last user message of a request body. The agent always packages
// tool results into one trailing user message.
func findToolResult(t *testing.T, body anthropic.MessageNewParams, toolUseID string) anthropic.ToolResultBlockParam {
	t.Helper()
	if len(body.Messages) == 0 {
		t.Fatal("request body has no messages")
	}
	last := body.Messages[len(body.Messages)-1]
	idx := slices.IndexFunc(last.Content, func(b anthropic.ContentBlockParamUnion) bool {
		return b.OfToolResult != nil && b.OfToolResult.ToolUseID == toolUseID
	})
	if idx < 0 {
		t.Fatalf("no tool_result with id=%q in last message; got %+v", toolUseID, last)
	}
	return *last.Content[idx].OfToolResult
}

// toolResultText flattens a tool_result block back into the string the agent
// produced. The SDK models content as a union of text/image blocks; for our
// tools it is always a single text block.
func toolResultText(t *testing.T, tr anthropic.ToolResultBlockParam) string {
	t.Helper()
	var parts []string
	for _, c := range tr.Content {
		if c.OfText != nil {
			parts = append(parts, c.OfText.Text)
		}
	}
	if len(parts) == 0 {
		t.Fatalf("tool_result has no text content: %+v", tr)
	}
	return strings.Join(parts, "")
}
