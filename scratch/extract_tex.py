import json
import re

log_path = "/home/ronit.28010/.gemini/antigravity-ide/brain/54f780f6-1cbf-4534-a9d1-24591cd0acd7/.system_generated/logs/transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "CODE_ACTION" and "main.tex" in data.get("content", ""):
            print("Found CODE_ACTION content length:", len(data["content"]))
            # Let's inspect the content
            content = data["content"]
            # Look for the diff pattern: @@ -1,0 +1,353 @@ or similar
            # Or just save the whole content to a text file for inspection
            with open("/home/ronit.28010/COD_Project/scratch/extracted_content.txt", "w") as out:
                out.write(content)
            print("Saved content to scratch/extracted_content.txt")
            break
