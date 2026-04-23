# python -m scripts.scratch_test
from app.chat_service import create_initial_state, run_chat_turn

state = create_initial_state()

result = run_chat_turn(state, "北京气候怎么样")
print("answer1:", result["answer"])

state = {
    "messages": result["messages"],
    "summary": result["summary"],
}

result = run_chat_turn(state, "那上海呢")
print("answer2:", result["answer"])
print("routes:", result.get("routes"))
print("summary:", result.get("summary"))
