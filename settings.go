package main

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/anthropics/anthropic-sdk-go"
)

type settings struct {
	Env map[string]string `json:"env"`
}

func loadSettings() (baseURL, apiKey, model string) {
	globalPath := filepath.Join(os.Getenv("HOME"), ".claude", "settings.json")
	readSettingsFile(globalPath, &baseURL, &apiKey, &model)

	readSettingsFile(".claude/settings.json", &baseURL, &apiKey, &model)

	if v := os.Getenv("ANTHROPIC_BASE_URL"); v != "" {
		baseURL = v
	}
	if v := os.Getenv("ANTHROPIC_AUTH_TOKEN"); v != "" {
		apiKey = v
	}
	if v := os.Getenv("ANTHROPIC_MODEL"); v != "" {
		model = v
	}

	if model == "" {
		model = string(anthropic.ModelClaudeOpus4_8)
	}

	return baseURL, apiKey, model
}

func readSettingsFile(path string, baseURL, apiKey, model *string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var s settings
	if err := json.Unmarshal(data, &s); err != nil {
		return
	}

	if v, ok := s.Env["ANTHROPIC_BASE_URL"]; ok && v != "" {
		*baseURL = v
	}
	if v, ok := s.Env["ANTHROPIC_AUTH_TOKEN"]; ok && v != "" {
		*apiKey = v
	}
	if v, ok := s.Env["ANTHROPIC_MODEL"]; ok && v != "" {
		*model = v
	}
}
