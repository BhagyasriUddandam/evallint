# CLAUDE.md — evalcheck

## Who I am
Bhagya Sri Uddandam, sole builder. MacBook M4 Pro (arm64). I need to genuinely understand
this code for interviews. Explain non-obvious decisions in plain language as you go.
Never hand me code I can't explain.

## What this is
evalcheck: a CLI tool + library that audits LLM eval DATASETS for flaws that make
evaluations silently lie. Read SPEC.md for the full design before doing anything.

It is NOT an eval runner. It does not compete with promptfoo/DeepEval. It audits the set.

## Scope discipline
v1 = exactly three checks: discrimination failure, near-duplicate detection, class imbalance.
Do not add a fourth check, a web UI, a database, or a cloud anything. If tempted to add
architecture, STOP and ask me first. Over-engineering is the main failure mode here.

## Tech
- Python 3.12+, managed with uv. `uv pip install`, never bare pip.
- numpy, pandas, sentence-transformers (local embeddings), pytest, click, rich.
- No Node, no JS, no database, no server.

## Workflow
1. For any task with 3+ steps, write a short plan to tasks/todo.md first and check it with me.
2. Build ONE module at a time. Show me it works before moving on.
3. Never mark done without running it and showing real output.
4. Write the test alongside each check, not after.
5. When I correct you, append the lesson to tasks/lessons.md and read that file at session start.

## Non-negotiables
- Every check must report its own limitations, not just its findings. A tool that states
  its uncertainty is more trustworthy.
- No fabricated example data presented as real results.
- Tests must prove each check fires on known-bad input AND stays quiet on known-good input.
