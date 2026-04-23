from app.chat_service import create_initial_state, run_chat_turn

state = create_initial_state()


def run_once(message: str):
    global state

    print(
        "\n===========================================================================#############============================================================"
    )
    print("USER:", message)

    result = run_chat_turn(state, message)
    answer = result.get("answer", "")
    print("\nFINAL ANSWER:", answer)

    state = {
        "messages": result.get("messages", []),
        "summary": result.get("summary", ""),
    }


if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            print("answer：请输入内容。")
            continue

        run_once(user_input)
