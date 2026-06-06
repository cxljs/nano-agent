package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests exercise the tools through the same entry point the agent uses
// (ToolDefinition.Function with a json.RawMessage input). They run against a
// real, isolated filesystem via t.TempDir() so we exercise the full read /
// list / write paths rather than mocked stand-ins.

// chdir switches the working directory for the duration of the test and
// restores it on cleanup. The tools take relative paths, so each test needs
// its own sandbox that "looks like" the project root.
func chdir(t *testing.T, dir string) {
	t.Helper()
	old, err := os.Getwd()
	require.NoError(t, err, "getwd")
	require.NoError(t, os.Chdir(dir), "chdir %s", dir)
	t.Cleanup(func() { _ = os.Chdir(old) })
}

// callTool marshals a typed input and invokes the tool's Function the same
// way the agent does, so we cover the json.Unmarshal path as well.
func callTool(t *testing.T, tool ToolDefinition, input any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(input)
	require.NoError(t, err, "marshal tool input")
	return tool.Function(raw)
}

func TestReadFile_ReturnsFileContents(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	want := "hello\nworld\n"
	require.NoError(t, os.WriteFile(filepath.Join(dir, "greeting.txt"), []byte(want), 0644), "seed file")

	got, err := callTool(t, ReadFileDefinition, ReadFileInput{Path: "greeting.txt"})
	require.NoError(t, err, "read_file")
	assert.Equal(t, want, got, "read_file content mismatch")
}

func TestReadFile_MissingFileReturnsError(t *testing.T) {
	chdir(t, t.TempDir())

	_, err := callTool(t, ReadFileDefinition, ReadFileInput{Path: "nope.txt"})
	assert.Error(t, err, "expected error for missing file")
}

func TestListFiles_IncludesNestedEntriesWithDirSuffix(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	// Build a small tree so we can assert both files and directories show up,
	// and that directories carry the trailing "/" the tool adds.
	mustMkdir(t, filepath.Join(dir, "sub"))
	mustWrite(t, filepath.Join(dir, "a.txt"), "a")
	mustWrite(t, filepath.Join(dir, "sub", "b.txt"), "b")

	out, err := callTool(t, ListFilesDefinition, ListFileInput{})
	require.NoError(t, err, "list_files")

	var entries []string
	require.NoError(t, json.Unmarshal([]byte(out), &entries), "list_files output is not a JSON array: %q", out)

	for _, want := range []string{"a.txt", "sub/", filepath.Join("sub", "b.txt")} {
		assert.Contains(t, entries, want, "missing entry")
	}
}

func TestListFiles_HonorsExplicitPath(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	mustMkdir(t, filepath.Join(dir, "pkg"))
	mustWrite(t, filepath.Join(dir, "pkg", "x.go"), "package pkg")
	// File outside "pkg" should not appear when we list "pkg".
	mustWrite(t, filepath.Join(dir, "outside.txt"), "x")

	out, err := callTool(t, ListFilesDefinition, ListFileInput{Path: "pkg"})
	require.NoError(t, err, "list_files")
	var entries []string
	require.NoError(t, json.Unmarshal([]byte(out), &entries), "output not JSON array")
	assert.Contains(t, entries, "x.go")
	assert.NotContains(t, entries, "outside.txt", "outside.txt should be scoped out")
}

func TestEditFile_ReplacesExistingSubstring(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	mustWrite(t, filepath.Join(dir, "f.txt"), "before middle after")

	res, err := callTool(t, EditFileDefinition, EditFileInput{
		Path: "f.txt", OldStr: "middle", NewStr: "MIDDLE",
	})
	require.NoError(t, err, "edit_file")
	assert.Equal(t, "OK", res)

	got, _ := os.ReadFile(filepath.Join(dir, "f.txt"))
	assert.Equal(t, "before MIDDLE after", string(got), "file not updated")
}

func TestEditFile_CreatesFileWhenMissingAndOldStrEmpty(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	// edit_file doubles as "create" when path is missing and old_str is empty.
	// We assert it also creates the parent directory, which is the only
	// non-trivial behavior in createNewFile.
	target := filepath.Join("nested", "new.txt")
	res, err := callTool(t, EditFileDefinition, EditFileInput{
		Path: target, OldStr: "", NewStr: "fresh",
	})
	require.NoError(t, err, "edit_file create")
	assert.Contains(t, res, "Successfully created", "unexpected create message")

	got, err := os.ReadFile(filepath.Join(dir, target))
	require.NoError(t, err, "created file unreadable")
	assert.Equal(t, "fresh", string(got), "created file content")
}

func TestEditFile_RejectsInvalidInput(t *testing.T) {
	chdir(t, t.TempDir())

	cases := []struct {
		name string
		in   EditFileInput
	}{
		{"empty path", EditFileInput{Path: "", OldStr: "a", NewStr: "b"}},
		{"old equals new", EditFileInput{Path: "x.txt", OldStr: "same", NewStr: "same"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := callTool(t, EditFileDefinition, tc.in)
			assert.Error(t, err, "expected error")
		})
	}
}

func TestEditFile_OldStrNotFoundReturnsError(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)
	mustWrite(t, filepath.Join(dir, "f.txt"), "hello")

	_, err := callTool(t, EditFileDefinition, EditFileInput{
		Path: "f.txt", OldStr: "absent", NewStr: "x",
	})
	assert.Error(t, err, "expected error when old_str is not in file")
}

func TestBash_ExecutesEcho(t *testing.T) {
	chdir(t, t.TempDir())

	got, err := callTool(t, BashDefinition, BashInput{Command: "echo hello world"})
	require.NoError(t, err)
	assert.Equal(t, "hello world\n", got)
}

func TestBash_PipesAndRedirectsWork(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	got, err := callTool(t, BashDefinition, BashInput{Command: "echo foo | tr a-z A-Z > out.txt && cat out.txt"})
	require.NoError(t, err, "bash pipe+touch")
	assert.Equal(t, "FOO\n", got)

	b, err := os.ReadFile(filepath.Join(dir, "out.txt"))
	require.NoError(t, err)
	assert.Equal(t, "FOO\n", string(b), "file written by shell redirect")
}

func TestBash_ReturnsStderrOnError(t *testing.T) {
	chdir(t, t.TempDir())

	got, err := callTool(t, BashDefinition, BashInput{Command: "ls nonexistent_dir"})
	assert.Error(t, err)
	// Both the command error and the tool's own wrapping should be visible,
	// and the output should contain the actual stderr from ls.
	assert.Contains(t, got, "No such file or directory", "stderr should be returned")
}

func TestBash_EmptyCommandRejected(t *testing.T) {
	chdir(t, t.TempDir())

	_, err := callTool(t, BashDefinition, BashInput{Command: ""})
	assert.ErrorContains(t, err, "command is required")
	_, err = callTool(t, BashDefinition, BashInput{Command: "   "})
	assert.ErrorContains(t, err, "command is required")
}

func TestBash_HonorsTimeout(t *testing.T) {
	chdir(t, t.TempDir())

	_, err := callTool(t, BashDefinition, BashInput{Command: "sleep 10", TimeoutMs: 100})
	assert.ErrorContains(t, err, "timed out")
}

func TestBash_CapsTimeoutAtMax(t *testing.T) {
	chdir(t, t.TempDir())

	// 700 seconds > 600 (10 minute) cap. The tool should cap internally and
	// the sleep should finish well within the capped timeout since we only
	// sleep 0.01s — the point is the cap doesn't error.
	got, err := callTool(t, BashDefinition, BashInput{Command: "sleep 0.01 && echo ok", TimeoutMs: 700_000})
	require.NoError(t, err)
	assert.Contains(t, got, "ok")
}

// TestToolsRegistry pins the wiring used by main: the agent receives the
// exact tools advertised here, by name. If a tool is renamed or dropped, the
// agent code that constructs ToolUnionParam (main.go) silently goes out of
// sync with prompts and docs, so we lock the contract.
func TestToolsRegistry(t *testing.T) {
	want := []string{"read_file", "list_files", "edit_file", "bash"}
	got := make([]string, 0, len(Tools))
	for _, td := range Tools {
		got = append(got, td.Name)
		assert.NotNil(t, td.Function, "tool %q has nil Function", td.Name)
		assert.NotEmpty(t, td.Desc, "tool %q has empty Desc", td.Name)
		assert.NotNil(t, td.InputSchema.Properties, "tool %q has nil InputSchema.Properties", td.Name)
	}
	// Order-insensitive equality: registration order is not part of the
	// contract, but the set of tools is.
	slices.Sort(want)
	slices.Sort(got)
	assert.Equal(t, want, got, "Tools mismatch")
}

// ---- helpers ----

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	require.NoError(t, os.WriteFile(path, []byte(content), 0644), "write %s", path)
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(path, 0755), "mkdir %s", path)
}
