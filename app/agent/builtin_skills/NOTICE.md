# Bundled skills

Skills the Agent loads on demand. They are copied from public repositories under their licenses, with these changes only:
cross-references to skills that are not bundled were reworded, the default plan folder is `docs/plans`, Codex's `agents/openai.yaml`
files and `mcp-builder`'s TypeScript guide were left out. Project or user skills with the same name take priority.

| Skill | Source | License |
|---|---|---|
| systematic-debugging, verification-before-completion, test-driven-development, writing-plans, receiving-code-review, requesting-code-review, dispatching-parallel-agents | [obra/superpowers](https://github.com/obra/superpowers) @ 8ca22dba9a94 | MIT, Jesse Vincent |
| grilling, handoff, research, diagnosing-bugs, codebase-design, improve-codebase-architecture, prototype, pr | [mattpocock/skills](https://github.com/mattpocock/skills) @ 4588b32ecab9 | MIT, Matt Pocock (`pr` credits Dex Horthy's show-me, see its CREDITS.md) |
| frontend-design, webapp-testing, mcp-builder | [anthropics/skills](https://github.com/anthropics/skills) @ 683bc88e56f3 | Apache-2.0, each folder keeps its LICENSE.txt |

## MIT licenses

### obra/superpowers

MIT License

Copyright (c) 2025 Jesse Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### mattpocock/skills

MIT License

Copyright (c) 2026 Matt Pocock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
