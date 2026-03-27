# Transmogrifier

Register-aware prompt translation. Normalizes linguistic register to maximize LLM output quality.

## Install

```bash
pip install transmogrifier
```

## Usage

```python
from transmogrifier.core import Transmogrifier

t = Transmogrifier()
result = t.translate("yo what's the deal with TCP", model="claude-opus-4")
print(result.output_text)       # Rewritten prompt
print(result.system_prompt)     # Level 1 system prompt to prepend
print(result.detected_register) # "casual"
print(result.target_register)   # "direct"
```
