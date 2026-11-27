"""Command-line chat.   python -m hotel_agent.cli            (interactive)
                         python -m hotel_agent.cli --ask "..."  (one question)"""
from __future__ import annotations

import argparse

from hotel_agent.agent import AgentResult, build_agent
from hotel_agent.config import ConfigError


def _show(result: AgentResult, verbose: bool) -> None:
    if verbose:
        for call in result.tool_calls:
            status = "ok" if call.result.get("ok") else f"ERROR: {call.result.get('error')}"
            detail = call.arguments.get("sql") or call.arguments
            print(f"  [tool] {call.name} ({call.duration_s:.2f}s) -> {status}\n         {detail}")
    print(f"\n{result.answer}\n")
    for path in result.charts:
        print(f"Chart saved: {path}")
    print(f"({result.steps} step(s), {len(result.tool_calls)} tool call(s), "
          f"{result.prompt_tokens + result.completion_tokens} tokens, {result.latency_s:.1f}s)")
    if result.error:
        print(f"Note: {result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hotel Analytics Agent")
    parser.add_argument("--ask", help="ask a single question and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    args = parser.parse_args()

    try:
        agent = build_agent()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    if args.ask:
        _show(agent.ask(args.ask), args.verbose)
        return

    print("Hotel Analytics Agent. Ask about bookings (type 'exit' to quit, 'reset' to clear memory).")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("Memory cleared.")
            continue
        if question:
            _show(agent.ask(question), args.verbose)


if __name__ == "__main__":
    main()
