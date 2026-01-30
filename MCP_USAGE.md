# MCP (Model Context Protocol) Usage Guide

The AI Workbench supports connecting to external MCP servers to extend the chatbot's capabilities with additional tools.

## Quick Start

### Connect to MCP Server During Chat

Use the `--mcp-server` option when starting the interactive chat:

```bash
# Connect to filesystem MCP server
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp

# Connect multiple MCP servers
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp \
  --mcp-server weather:python:weather_server.py
```

### MCP Server Format

The `--mcp-server` parameter uses the format: `name:command:args`

- **name**: Unique identifier for the server (e.g., `filesystem`, `weather`)
- **command**: Command to run the server (e.g., `npx`, `python3`, `node`)
- **args**: Comma-separated arguments for the command

**Examples:**

```bash
# Filesystem server
--mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp

# Python-based server
--mcp-server myserver:python3:server.py,--port,8080

# Node.js server
--mcp-server nodejs-server:node:server.js
```

## Chat Commands

Once connected, use these commands in the interactive chat:

### `/mcp-tools` - List Available Tools

Shows all tools from connected MCP servers:

```
[You] > /mcp-tools

Available MCP Tools (14):

filesystem:
  • filesystem:read_text_file
    Read the complete contents of a file from the file system
  • filesystem:write_file
    Create a new file or overwrite existing file
  • filesystem:list_directory
    Get detailed listing of files and directories
  ...
```

### `/mcp-call` - Call an MCP Tool

Call a specific MCP tool with JSON arguments:

```
[You] > /mcp-call filesystem:list_directory {"path": "/tmp"}

✓ filesystem:list_directory succeeded:

[DIR] test-folder/
[FILE] example.txt
[FILE] data.json
```

**JSON Arguments Format:**

```bash
# Simple argument
/mcp-call filesystem:read_text_file {"path": "/tmp/test.txt"}

# Multiple arguments
/mcp-call filesystem:edit_file {"path": "/tmp/test.txt", "edits": [...]}

# No arguments (empty object)
/mcp-call weather:get_current {}
```

## Common MCP Servers

### 1. Filesystem Server

Access local filesystem (read, write, list files):

```bash
# Install and run
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp
```

**Available Tools:**
- `read_text_file` - Read file contents
- `write_file` - Write to file
- `list_directory` - List directory contents
- `search_files` - Search for files by pattern
- `get_file_info` - Get file metadata
- And more...

### 2. GitHub MCP Server

Interact with GitHub repositories:

```bash
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --mcp-server github:npx:-y,@modelcontextprotocol/server-github
```

### 3. PostgreSQL MCP Server

Query PostgreSQL databases:

```bash
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --mcp-server postgres:npx:-y,@modelcontextprotocol/server-postgres
```

## Programmatic Usage

### Using the Async Client

```python
import asyncio
from ai_workbench.mcp.client import MCPClient

async def main():
    client = MCPClient()

    # Connect to server
    success = await client.connect_server(
        server_name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    if success:
        # List available tools
        tools = client.list_tools()
        for tool in tools:
            print(f"{tool.server_name}:{tool.name} - {tool.description}")

        # Call a tool
        result = await client.call_tool(
            tool_name="filesystem:list_directory",
            arguments={"path": "/tmp"}
        )

        if result.success:
            print(result.content)
        else:
            print(f"Error: {result.error}")

        # Disconnect
        await client.disconnect_all()

asyncio.run(main())
```

### Using the Synchronous Wrapper

```python
from ai_workbench.mcp.client import MCPClientManager

# Use context manager for automatic cleanup
with MCPClientManager() as manager:
    # Connect to server
    success = manager.connect_server(
        server_name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    if success:
        # List tools
        tools = manager.list_tools()
        print(f"Connected with {len(tools)} tools")

        # Call tool
        result = manager.call_tool(
            tool_name="filesystem:list_directory",
            arguments={"path": "/tmp"}
        )

        if result.success:
            print(result.content)
```

## Troubleshooting

### Connection Fails

**Issue:** `Failed to connect to <server>`

**Solutions:**
1. Check if required command is installed (`npx`, `python3`, etc.)
2. Verify the server package is available
3. Check server arguments are correct
4. Try running the server command manually first

```bash
# Test filesystem server manually
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### Tool Call Fails

**Issue:** `MCP error -32602: Input validation error`

**Solution:** Verify you're passing the correct arguments in JSON format:

```bash
# Wrong (missing quotes)
/mcp-call filesystem:read_text_file {path: /tmp/test.txt}

# Correct
/mcp-call filesystem:read_text_file {"path": "/tmp/test.txt"}
```

### No Tools Available

**Issue:** `/mcp-tools` shows "No MCP tools available"

**Solutions:**
1. Make sure you connected with `--mcp-server` when starting chat
2. Check connection was successful (look for "✓ Connected to ..." message)
3. Try reconnecting or restarting the chat

## Architecture

The MCP client implementation consists of:

- **`MCPClient`** (async): Core async client for connecting to MCP servers
- **`MCPClientManager`** (sync): Synchronous wrapper using background thread
- **`MCPTool`**: Tool metadata (name, description, schema)
- **`MCPToolResult`**: Result from tool execution

The client uses:
- `AsyncExitStack` to maintain persistent connections
- Background thread for synchronous interface (avoids event loop conflicts)
- stdio-based communication with MCP servers

## Learn More

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Building MCP Servers](https://modelcontextprotocol.io/docs/building-servers)
