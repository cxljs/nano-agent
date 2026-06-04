package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
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

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(&client, getUserMessage, model)
	err := agent.Run(context.TODO())
	if err != nil {
		log.Printf("Error: %s\n", err.Error())
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	model          string
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool), model string) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		model:          model,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with Claude (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\x1b[94mYou\x1b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\x1b[93mClaude\x1b[0m: %s\n", content.Text)
			}
		}
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     a.model,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})
	return message, err
}
