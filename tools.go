package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
)

var Tools = []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition, BashDefinition}

type ToolDefinition struct {
	Name        string                         `json:"name"`
	Desc        string                         `json:"description"`
	InputSchema anthropic.ToolInputSchemaParam `json:"input_schema"`
	Function    func(input json.RawMessage) (string, error)
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Desc:        "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		log.Panicf("read file err: %s\n", err.Error())
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)

	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Desc:        "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFileInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFileInput]()

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFileInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		log.Panicf("list files err: %s\n", err.Error())
	}

	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}
	return string(result), nil
}

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Desc: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: EditFileInputSchema,
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

var EditFileInputSchema = GenerateSchema[EditFileInput]()

func EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		log.Panicf("edit file err: %s\n", err.Error())
	}

	if editFileInput.Path == "" || editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, -1)

	if oldContent == newContent && editFileInput.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(editFileInput.Path, []byte(newContent), 0644)
	if err != nil {
		return "", err
	}

	return "OK", nil
}

var BashDefinition = ToolDefinition{
	Name: "bash",
	// The model needs to know this is a non-interactive shell so it does not
	// reach for commands like `vim` or `less` that would hang waiting for a TTY.
	Desc: `Execute a bash command in a non-interactive shell and return the combined stdout and stderr.

Commands run from the agent's current working directory. Avoid interactive programs (editors, pagers, prompts) — they will hang until timeout. Long-running commands are killed at the timeout (default 30 seconds, max 600).`,
	InputSchema: BashInputSchema,
	Function:    Bash,
}

type BashInput struct {
	Command   string `json:"command" jsonschema_description:"The bash command line to execute. Passed to 'bash -c' as a single string, so shell features like pipes and redirection work."`
	TimeoutMs int    `json:"timeout_ms,omitempty" jsonschema_description:"Optional timeout in milliseconds. Defaults to 30000 (30s) if omitted; capped at 600000 (10min)."`
}

var BashInputSchema = GenerateSchema[BashInput]()

func Bash(input json.RawMessage) (string, error) {
	bashInput := BashInput{}
	err := json.Unmarshal(input, &bashInput)
	if err != nil {
		log.Panicf("bash err: %s\n", err.Error())
	}

	if strings.TrimSpace(bashInput.Command) == "" {
		return "", fmt.Errorf("command is required")
	}

	// Bound the runtime so a runaway command can't wedge the agent loop.
	// 30s matches typical CLI tool defaults; the 10min cap mirrors the upper
	// end of what most shells consider "reasonable" for an interactive task.
	timeout := 30 * time.Second
	if bashInput.TimeoutMs > 0 {
		timeout = time.Duration(bashInput.TimeoutMs) * time.Millisecond
		if max := 10 * time.Minute; timeout > max {
			timeout = max
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// `bash -c` so the model can use pipes, globs, redirection, etc. without
	// us having to parse a command line ourselves.
	cmd := exec.CommandContext(ctx, "bash", "-c", bashInput.Command)
	// Combined output: the model usually wants stderr too (errors, progress),
	// and separating the two streams would lose interleaving order.
	out, runErr := cmd.CombinedOutput()

	if ctx.Err() == context.DeadlineExceeded {
		return string(out), fmt.Errorf("command timed out after %s", timeout)
	}
	if runErr != nil {
		// Return output alongside the error so the model can see *why* the
		// command failed (stack trace, missing file, etc.), not just the exit code.
		return string(out), fmt.Errorf("command failed: %w", runErr)
	}
	return string(out), nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}
