# Finetune Data Collector for VS Code

Collect real human feedback from prose and code work as reusable SFT and preference/DPO records.

## Principles

- Works for code edits and prose edits.
- The active VS Code agent/model can call the `#finetuneFeedback` language-model tool.
- Tracks editor changes locally; it does not send every keystroke to a model.
- SFT stores the human-accepted/corrected result.
- Preference/DPO is emitted only when a real AI candidate and a real human-chosen result both exist.
- Never fabricates a rejected candidate.
- One independent decision should become one record.

## Output directory

`finetuneCollector.outputDirectory` is configurable in VS Code Settings.

Default:

```
./finetune-data/
```

For this novels repository, add this to workspace settings:

```json
{
  "finetuneCollector.outputDirectory": "./data"
}
```

## Real editing trial (v0.1.1)

Open the target document, run **Finetune Collector: Start Collection Session**,
and enter the editing request before asking AI to edit. Save the document to
automatically collect a draft. Once you approve the result, run **Finetune
Collector: Capture Current Edit** to write an SFT-eligible record.

Drafts are not training-eligible. File changes cannot establish who authored an
edit or prove a DPO preference, so this workflow does not create preference pairs.
Sessions live in memory: restarting the extension requires starting a new session.
Use the **Finetune Collector** Output channel to see saved record paths.

Build with `npm ci && npm run compile`. Package using Node 22 or newer:
`npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license`.

## Agent tool and manual capture

1. Registers an agent tool named `finetuneCollector_recordFeedback`.
2. The current VS Code agent can invoke it after explicit correction/acceptance/rejection.
3. Tracks text-document edits so a human edit session can be captured.
4. Command **Finetune Collector: Capture Current Edit** saves a tracked before/after pair.
5. Writes schema `3.0-draft` records while retaining separate `sft` and `preference` sections.

## Development

```bash
cd vscode-finetune-collector
npm install
npm run compile
```

Then open the extension folder in VS Code and press F5 to launch an Extension Development Host.

## Next implementation steps

- Correlate an AI-applied patch with subsequent human edits, instead of treating arbitrary before/after edits as DPO.
- Add debounce/session boundaries for saves and commits.
- Add Git evidence (commit, changed hunks, tests) to quality metadata.
- Read an optional project schema/config and validate records before writing.
- Add deduplication and exporters for model-specific SFT/DPO JSONL.
- Add tests and Marketplace packaging.
