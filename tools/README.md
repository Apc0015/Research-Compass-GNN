# Tools Directory

This directory contains utility scripts for testing and diagnostics.

## Scripts

### check_llm_endpoint.py
Quick diagnostic tool for checking local LLM provider endpoints (LM Studio, Ollama).

**Purpose:** Helps debug connectivity issues with local LLM providers by probing common endpoints.

**Usage:**
```bash
python tools/check_llm_endpoint.py --hosts localhost --ports 11434,1234,8080
```

**What it does:**
- Tests common endpoints for LM Studio and Ollama
- Reports connection status and model availability
- Helps troubleshoot why the Gradio UI can't fetch available models

### run_feature_tests.py
Runs feature tests for the Research Compass system.

**Purpose:** Validates that core components are working correctly.

**Usage:**
```bash
python tools/run_feature_tests.py
```

**What it tests:**
- LLM Providers: Tests LM Studio and Ollama connections and model listing
- GraphManager: Tests Neo4j connection and basic graph operations
- Does not mutate external resources (read-only tests)

## When to use these tools

- **check_llm_endpoint.py**: Use when you're having trouble connecting to local LLM providers
- **run_feature_tests.py**: Use after installation or configuration changes to verify everything works

## Dependencies

Both scripts use standard dependencies from the main project. Ensure you have:
- requests (for endpoint checking)
- Access to configured LLM providers and Neo4j database
