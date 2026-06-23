# decisions/ — the user inbox

The single async channel for anything that needs the user. One file per open decision:

```
coordination/decisions/<short-slug>.md
```

Each file states the question, the options with a recommendation, and what is blocked on it. When
an agent adds one, it also fires `python harness/tools/notify.py --level gate "<one-line>"`. The
dashboard **Inbox** lists these; the user answers in-session or by editing the file. Resolved
decisions are deleted (the LOG records the outcome).

Rule (see `CLAUDE.md` → Autonomy): never bury a question for the user inside a design or technical
doc. It goes here, or it does not exist.
