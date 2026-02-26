#!/usr/bin/env python3
"""Filter Claude Code stream-json output into readable terminal output."""
import json
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue

    msg_type = msg.get("type", "")

    if msg_type == "assistant":
        # Assistant text output
        message = msg.get("message", {})
        for block in message.get("content", []):
            if block.get("type") == "text":
                print(block["text"], end="", flush=True)
            elif block.get("type") == "tool_use":
                tool = block.get("name", "?")
                inp = block.get("input", {})
                print(f"\n{CYAN}{BOLD}[{tool}]{RESET} ", end="", flush=True)
                if tool == "Edit":
                    path = inp.get("file_path", "")
                    print(f"{DIM}{path}{RESET}", flush=True)
                elif tool == "Write":
                    path = inp.get("file_path", "")
                    print(f"{DIM}{path}{RESET}", flush=True)
                elif tool == "Bash":
                    cmd = inp.get("command", "")
                    desc = inp.get("description", "")
                    label = desc if desc else cmd[:80]
                    print(f"{DIM}{label}{RESET}", flush=True)
                elif tool == "Read":
                    path = inp.get("file_path", "")
                    print(f"{DIM}{path}{RESET}", flush=True)
                else:
                    print(flush=True)

    elif msg_type == "tool_result":
        # Brief summary of tool results
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    lines = text.strip().split("\n")
                    if len(lines) > 5:
                        for l in lines[:3]:
                            print(f"  {DIM}{l}{RESET}")
                        print(f"  {DIM}... ({len(lines) - 3} more lines){RESET}")
                    elif text.strip():
                        for l in lines:
                            print(f"  {DIM}{l}{RESET}")
        elif isinstance(content, str) and content.strip():
            lines = content.strip().split("\n")
            if len(lines) > 5:
                for l in lines[:3]:
                    print(f"  {DIM}{l}{RESET}")
                print(f"  {DIM}... ({len(lines) - 3} more lines){RESET}")
            else:
                for l in lines:
                    print(f"  {DIM}{l}{RESET}")

    elif msg_type == "result":
        # Final result
        text = msg.get("result", "")
        if text:
            print(f"\n{GREEN}{BOLD}[Result]{RESET} {text[:200]}", flush=True)
