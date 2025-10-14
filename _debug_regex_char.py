from pathlib import Path
import re
text = Path("autogern.py").read_text(encoding="utf-8")
match = re.search(r"re\.split\(r'\[(.*?)\]'", text)
if match:
    snippet = match.group(1)
    print(snippet)
    for ch in snippet:
        print(repr(ch), ord(ch))
