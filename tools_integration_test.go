package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
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
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir %s: %v", dir, err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
}

// callTool marshals a typed input and invokes the tool's Function the same
// way the agent does, so we cover the json.Unmarshal path as well.
func callTool(t *testing.T, tool ToolDefinition, input any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal tool input: %v", err)
	}
	return tool.Function(raw)
}

func TestReadFile_ReturnsFileContents(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	want := "hello\nworld\n"
	if err := os.WriteFile(filepath.Join(dir, "greeting.txt"), []byte(want), 0644); err != nil {
		t.Fatalf("seed file: %v", err)
	}

	got, err := callTool(t, ReadFileDefinition, ReadFileInput{Path: "greeting.txt"})
	if err != nil {
		t.Fatalf("read_file: %v", err)
	}
	if got != want {
		t.Errorf("read_file content mismatch:\n got: %q\nwant: %q", got, want)
	}
}

func TestReadFile_MissingFileReturnsError(t *testing.T) {
	chdir(t, t.TempDir())

	_, err := callTool(t, ReadFileDefinition, ReadFileInput{Path: "nope.txt"})
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
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
	if err != nil {
		t.Fatalf("list_files: %v", err)
	}

	var entries []string
	if err := json.Unmarshal([]byte(out), &entries); err != nil {
		t.Fatalf("list_files output is not a JSON array: %v (%q)", err, out)
	}

	for _, want := range []string{"a.txt", "sub/", filepath.Join("sub", "b.txt")} {
		if !slices.Contains(entries, want) {
			t.Errorf("missing entry %q in %v", want, entries)
		}
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
	if err != nil {
		t.Fatalf("list_files: %v", err)
	}
	var entries []string
	if err := json.Unmarshal([]byte(out), &entries); err != nil {
		t.Fatalf("output not JSON array: %v", err)
	}
	if !slices.Contains(entries, "x.go") {
		t.Errorf("expected x.go in %v", entries)
	}
	if slices.Contains(entries, "outside.txt") {
		t.Errorf("outside.txt should be scoped out, got %v", entries)
	}
}

func TestEditFile_ReplacesExistingSubstring(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)

	mustWrite(t, filepath.Join(dir, "f.txt"), "before middle after")

	res, err := callTool(t, EditFileDefinition, EditFileInput{
		Path: "f.txt", OldStr: "middle", NewStr: "MIDDLE",
	})
	if err != nil {
		t.Fatalf("edit_file: %v", err)
	}
	if res != "OK" {
		t.Errorf("expected OK, got %q", res)
	}

	got, _ := os.ReadFile(filepath.Join(dir, "f.txt"))
	if string(got) != "before MIDDLE after" {
		t.Errorf("file not updated, got %q", got)
	}
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
	if err != nil {
		t.Fatalf("edit_file create: %v", err)
	}
	if !strings.Contains(res, "Successfully created") {
		t.Errorf("unexpected create message: %q", res)
	}

	got, err := os.ReadFile(filepath.Join(dir, target))
	if err != nil {
		t.Fatalf("created file unreadable: %v", err)
	}
	if string(got) != "fresh" {
		t.Errorf("created file content = %q, want %q", got, "fresh")
	}
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
			if _, err := callTool(t, EditFileDefinition, tc.in); err == nil {
				t.Fatal("expected error, got nil")
			}
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
	if err == nil {
		t.Fatal("expected error when old_str is not in file")
	}
}

// TestToolsRegistry pins the wiring used by main: the agent receives the
// exact tools advertised here, by name. If a tool is renamed or dropped, the
// agent code that constructs ToolUnionParam (main.go) silently goes out of
// sync with prompts and docs, so we lock the contract.
func TestToolsRegistry(t *testing.T) {
	want := []string{"read_file", "list_files", "edit_file"}
	got := make([]string, 0, len(Tools))
	for _, td := range Tools {
		got = append(got, td.Name)
		if td.Function == nil {
			t.Errorf("tool %q has nil Function", td.Name)
		}
		if td.Desc == "" {
			t.Errorf("tool %q has empty Desc", td.Name)
		}
		if td.InputSchema.Properties == nil {
			t.Errorf("tool %q has nil InputSchema.Properties", td.Name)
		}
	}
	// Order-insensitive equality: registration order is not part of the
	// contract, but the set of tools is.
	slices.Sort(want)
	slices.Sort(got)
	if !slices.Equal(want, got) {
		t.Errorf("Tools mismatch: got %v, want %v", got, want)
	}
}

// ---- helpers ----

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}
