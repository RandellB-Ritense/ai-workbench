"""
MCP (Model Context Protocol) client for connecting to external MCP servers.
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, StdioServerParameters as StdioParams
from contextlib import AsyncExitStack


@dataclass
class MCPTool:
    """Information about an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    tool_name: str
    success: bool
    content: Any
    error: Optional[str] = None


class MCPClient:
    """
    Client for connecting to MCP servers and using their tools.

    Supports stdio-based MCP servers (most common for local tools).
    """

    def __init__(self):
        """Initialize MCP client."""
        self.servers: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._exit_stack = AsyncExitStack()
        self._initialized = False

    async def connect_server(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Connect to an MCP server.

        Args:
            server_name: Unique name for this server
            command: Command to run the server (e.g., "npx", "python")
            args: Arguments for the command (e.g., ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
            env: Optional environment variables

        Returns:
            True if connection successful
        """
        try:
            # Initialize exit stack if not done
            if not self._initialized:
                await self._exit_stack.__aenter__()
                self._initialized = True

            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env,
            )

            # Connect to server and keep connection alive using exit stack
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize session
            await session.initialize()

            # Store session
            self.servers[server_name] = session

            # Discover tools
            await self._discover_tools(server_name, session)

            return True

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            return False

    async def _discover_tools(self, server_name: str, session: ClientSession):
        """
        Discover available tools from an MCP server.

        Args:
            server_name: Name of the server
            session: Client session
        """
        try:
            # List tools
            result = await session.list_tools()

            # Store tools
            for tool in result.tools:
                tool_info = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                    server_name=server_name,
                )
                # Use server_name prefix to avoid conflicts
                full_name = f"{server_name}:{tool.name}"
                self.tools[full_name] = tool_info

        except Exception as e:
            print(f"Failed to discover tools from {server_name}: {e}")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPToolResult:
        """
        Call an MCP tool.

        Args:
            tool_name: Full tool name (server_name:tool_name)
            arguments: Tool arguments

        Returns:
            MCPToolResult with the result
        """
        # Parse tool name
        if ":" not in tool_name:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                error=f"Invalid tool name format. Use server_name:tool_name",
            )

        server_name, actual_tool_name = tool_name.split(":", 1)

        # Check if server exists
        if server_name not in self.servers:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                error=f"Server {server_name} not connected",
            )

        # Check if tool exists
        if tool_name not in self.tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                error=f"Tool {tool_name} not found",
            )

        try:
            # Call tool
            session = self.servers[server_name]
            result = await session.call_tool(actual_tool_name, arguments or {})

            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                content=result.content,
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                error=str(e),
            )

    def list_tools(self) -> List[MCPTool]:
        """
        List all available tools from all connected servers.

        Returns:
            List of MCPTool objects
        """
        return list(self.tools.values())

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get information about a specific tool.

        Args:
            tool_name: Full tool name (server_name:tool_name)

        Returns:
            MCPTool object or None if not found
        """
        return self.tools.get(tool_name)

    def is_connected(self, server_name: str) -> bool:
        """
        Check if a server is connected.

        Args:
            server_name: Name of the server

        Returns:
            True if connected
        """
        return server_name in self.servers

    async def disconnect(self, server_name: str):
        """
        Disconnect from an MCP server.

        Args:
            server_name: Name of the server
        """
        if server_name in self.servers:
            # Remove tools from this server
            self.tools = {
                name: tool
                for name, tool in self.tools.items()
                if tool.server_name != server_name
            }
            # Remove server
            del self.servers[server_name]
            # Note: Context will be cleaned up when disconnect_all is called

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        self.servers.clear()
        self.tools.clear()

        # Clean up all connections
        if self._initialized:
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
        return False


class MCPClientManager:
    """
    Manager for MCP client with synchronous interface.

    Wraps async MCPClient for easier use in non-async code.
    Uses a background thread to run the event loop to avoid conflicts.
    """

    def __init__(self):
        """Initialize MCP client manager."""
        import threading

        self.client = MCPClient()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._setup_done = False
        self._lock = threading.Lock()

    def _ensure_loop(self):
        """Ensure event loop is running in background thread."""
        import threading

        with self._lock:
            if self.loop is None or not self.loop.is_running():
                # Create new event loop in background thread
                self.loop = asyncio.new_event_loop()

                def run_loop():
                    asyncio.set_event_loop(self.loop)
                    self.loop.run_forever()

                self._thread = threading.Thread(target=run_loop, daemon=True)
                self._thread.start()

                # Wait for loop to be ready
                import time
                while not self.loop.is_running():
                    time.sleep(0.01)

    def _run_coro(self, coro):
        """Run a coroutine in the background loop."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)

    def connect_server(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Connect to an MCP server (synchronous)."""
        return self._run_coro(
            self.client.connect_server(server_name, command, args, env)
        )

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPToolResult:
        """Call an MCP tool (synchronous)."""
        return self._run_coro(self.client.call_tool(tool_name, arguments))

    def list_tools(self) -> List[MCPTool]:
        """List all available tools."""
        return self.client.list_tools()

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get tool information."""
        return self.client.get_tool(tool_name)

    def is_connected(self, server_name: str) -> bool:
        """Check if server is connected."""
        return self.client.is_connected(server_name)

    def disconnect(self, server_name: str):
        """Disconnect from server (synchronous)."""
        return self._run_coro(self.client.disconnect(server_name))

    def disconnect_all(self):
        """Disconnect from all servers (synchronous)."""
        return self._run_coro(self.client.disconnect_all())

    def cleanup(self):
        """Clean up resources and stop event loop."""
        try:
            if self.client.servers:
                self.disconnect_all()

            # Stop the event loop
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)

            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except:
            pass  # Ignore errors during cleanup

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'client') and hasattr(self.client, 'servers'):
                self.cleanup()
        except:
            pass  # Ignore errors during cleanup
