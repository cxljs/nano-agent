package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/chzyer/readline"
)

func main() {
	baseURL, apiKey, model := loadSettings()

	opts := []option.RequestOption{}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	client := anthropic.NewClient(opts...)

	rl, err := readline.New("\x1b[94mYou\x1b[0m: ")
	if err != nil {
		log.Fatalf("readline init failed: %s", err)
	}
	defer rl.Close()

	getUserMessage := func() (string, bool) {
		input, err := rl.Readline()
		if err != nil {
			return "", false
		}
		if strings.TrimSpace(input) == "/exit" {
			return "", false
		}
		return input, true
	}

	agent := NewAgent(&client, getUserMessage, model, Tools)
	err = agent.Run(context.TODO())
	if err != nil {
		log.Printf("Error: %s\n", err.Error())
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	model          string
	tools          []ToolDefinition
	toolParams     []anthropic.ToolUnionParam
}

func NewAgent(
	client *anthropic.Client,
	getUserMessage func() (string, bool),
	model string,
	tools []ToolDefinition,
) *Agent {
	anthropicTools := []anthropic.ToolUnionParam{}
	for _, tool := range tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Desc),
				InputSchema: tool.InputSchema,
			},
		})
	}

	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		model:          model,
		tools:          tools,
		toolParams:     anthropicTools,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with Claude (use 'ctrl-c' or '/exit' to quit)")

	readUserInput := true
	for {
		if readUserInput {
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}
			if strings.TrimSpace(userInput) == "" {
				continue
			}

			userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
			conversation = append(conversation, userMessage)
		}

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		// log.Printf("message.Content: %v\n", message.Content)

		toolResults := []anthropic.ContentBlockParamUnion{}
		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\x1b[93mClaude\x1b[0m: %s\n", content.Text)
			case "tool_use":
				result := a.executeTool(content.ID, content.Name, content.Input)
				toolResults = append(toolResults, result)
			}
		}
		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		readUserInput = false
		conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     a.model,
		MaxTokens: int64(1024),
		Messages:  conversation,
		Tools:     a.toolParams,
	})
	return message, err
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) anthropic.ContentBlockParamUnion {
	var toolDef ToolDefinition
	found := false
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return anthropic.NewToolResultBlock(id, "tool not found", true)
	}

	fmt.Printf("\x1b[92mtool\x1b[0m: %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return anthropic.NewToolResultBlock(id, err.Error(), true)
	}
	return anthropic.NewToolResultBlock(id, response, false)
}
