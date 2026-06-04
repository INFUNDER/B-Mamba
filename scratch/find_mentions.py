import json

log_path = "/home/ronit.28010/.gemini/antigravity-ide/brain/54f780f6-1cbf-4534-a9d1-24591cd0acd7/.system_generated/logs/transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content_str = str(data)
        if "main.tex" in content_str:
            print(f"Line {i+1}: type={data.get('type')}, source={data.get('source')}, status={data.get('status')}")
            # If there's a tool call to write/create, print it
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    if "main.tex" in str(tc):
                        print("  Tool call:", tc["name"], tc["args"].keys() if isinstance(tc["args"], dict) else tc["args"])
