import * as vscode from 'vscode';
import * as path from 'path';
import { randomUUID } from 'crypto';

type FeedbackInput = {
  taskType: string;
  userRequest?: string;
  context?: string;
  original?: string;
  aiCandidate?: string;
  humanFinal: string;
  feedbackReason?: string;
  accepted?: boolean;
  sourceFile?: string;
};

type EditSession = {
  uri: string;
  before: string;
  after: string;
  languageId: string;
  changedAt: string;
};

const sessions = new Map<string, EditSession>();
const armed = new Map<string, { request: string; before: string; lastSaved: string }>();
let log: vscode.OutputChannel;

function workspaceRoot(): vscode.Uri | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri;
}

function outputDir(): vscode.Uri | undefined {
  const root = workspaceRoot();
  if (!root) return undefined;
  const configured = vscode.workspace.getConfiguration('finetuneCollector').get<string>('outputDirectory', './finetune-data');
  return vscode.Uri.joinPath(root, configured.replace(/^\.\//, ''));
}

async function writeRecord(input: FeedbackInput, source: string) {
  const dir = outputDir();
  if (!dir) throw new Error('Open a workspace before collecting finetune data.');
  await vscode.workspace.fs.createDirectory(dir);

  // A changed file alone is not evidence of a same-prompt author preference.
  const hasRealRejected = Boolean(input.accepted === true && input.userRequest && input.aiCandidate && input.humanFinal && input.aiCandidate !== input.humanFinal);
  const record = {
    schema_version: '3.0-draft',
    task_type: input.taskType,
    prompt: {
      user_request: input.userRequest ?? null,
      context: input.context ?? null,
      original: input.original ?? null
    },
    feedback: {
      reason: input.feedbackReason ?? null,
      accepted: input.accepted ?? null
    },
    sft: {
      eligible: input.accepted === true && Boolean(input.userRequest),
      response: input.humanFinal,
      target_type: hasRealRejected ? 'human_corrected_revision' : 'accepted_result'
    },
    preference: {
      eligible: hasRealRejected,
      candidates: hasRealRejected ? [
        { candidate_id: 'ai', response: input.aiCandidate, origin: 'ai_candidate', status: 'rejected' },
        { candidate_id: 'human', response: input.humanFinal, origin: 'human_edit', status: 'chosen' }
      ] : [],
      chosen_candidate_id: hasRealRejected ? 'human' : null,
      preference_reason: input.feedbackReason ?? null,
      selection_source: hasRealRejected ? 'human_edit' : null
    },
    source: {
      type: source,
      file: input.sourceFile ?? null,
      captured_at: new Date().toISOString()
    },
    quality: {
      real_rejected_candidate: hasRealRejected,
      human_accepted: input.accepted ?? false
    }
  };

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const safeType = input.taskType.replace(/[^a-zA-Z0-9_-]/g, '_');
  const target = vscode.Uri.joinPath(dir, `${stamp}-${safeType}-${randomUUID()}.json`);
  await vscode.workspace.fs.writeFile(target, Buffer.from(JSON.stringify(record, null, 2) + '\n', 'utf8'));
  log?.appendLine(`Saved ${source}: ${target.fsPath}`);
  return target;
}

class RecordFeedbackTool implements vscode.LanguageModelTool<FeedbackInput> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<FeedbackInput>): Promise<vscode.LanguageModelToolResult> {
    const target = await writeRecord(options.input, 'vscode_agent_tool');
    return new vscode.LanguageModelToolResult([
      new vscode.LanguageModelTextPart(`Saved training record to ${vscode.workspace.asRelativePath(target)}. No rejected candidate was fabricated.`)
    ]);
  }

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<FeedbackInput>) {
    return {
      invocationMessage: 'Preparing finetune feedback record',
      confirmationMessages: {
        title: 'Save finetune feedback?',
        message: new vscode.MarkdownString(`Save this **${options.input.taskType}** feedback as training data?`)
      }
    };
  }
}

export function activate(context: vscode.ExtensionContext) {
  log = vscode.window.createOutputChannel('Finetune Collector');
  context.subscriptions.push(log);
  log.appendLine('Finetune Collector activated');
  for (const doc of vscode.workspace.textDocuments) {
    if (doc.uri.scheme === 'file') sessions.set(doc.uri.toString(), {
      uri: doc.uri.toString(), before: doc.getText(), after: doc.getText(),
      languageId: doc.languageId, changedAt: new Date().toISOString()
    });
  }
  context.subscriptions.push(vscode.commands.registerCommand('finetuneCollector.startSession', async () => {
    const doc = vscode.window.activeTextEditor?.document;
    if (!doc || doc.uri.scheme !== 'file') return;
    const request = await vscode.window.showInputBox({ prompt: '本次编辑要求（先开始采集，再让 AI 修改）', ignoreFocusOut: true });
    if (!request?.trim()) return;
    armed.set(doc.uri.toString(), { request, before: doc.getText(), lastSaved: doc.getText() });
    log.appendLine(`Session started: ${doc.uri.fsPath}`);
    vscode.window.showInformationMessage('采集已开始：保存时自动记录待确认样本；认可最终结果后执行 Capture Current Edit。');
  }));
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(async doc => {
    const session = armed.get(doc.uri.toString());
    if (!session || session.lastSaved === doc.getText()) return;
    const text = doc.getText();
    try {
      await writeRecord({ taskType: doc.languageId === 'markdown' ? 'prose_edit' : 'code_edit',
        userRequest: session.request, original: session.before, humanFinal: text,
        accepted: false, sourceFile: vscode.workspace.asRelativePath(doc.uri)
      }, 'vscode_saved_draft');
      session.lastSaved = text;
    } catch (error) { vscode.window.showErrorMessage(`采集失败：${String(error)}`); }
  }));
  context.subscriptions.push(
    vscode.lm.registerTool('finetuneCollector_recordFeedback', new RecordFeedbackTool())
  );

  context.subscriptions.push(vscode.workspace.onDidOpenTextDocument(doc => {
    if (doc.uri.scheme === 'file') {
      sessions.set(doc.uri.toString(), {
        uri: doc.uri.toString(),
        before: doc.getText(),
        after: doc.getText(),
        languageId: doc.languageId,
        changedAt: new Date().toISOString()
      });
    }
  }));

  context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(event => {
    if (!vscode.workspace.getConfiguration('finetuneCollector').get<boolean>('autoTrackEdits', true)) return;
    if (event.document.uri.scheme !== 'file' || event.contentChanges.length === 0) return;
    const key = event.document.uri.toString();
    const existing = sessions.get(key);
    sessions.set(key, {
      uri: key,
      before: existing?.before ?? '',
      after: event.document.getText(),
      languageId: event.document.languageId,
      changedAt: new Date().toISOString()
    });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('finetuneCollector.captureEdit', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const key = editor.document.uri.toString();
    const session = sessions.get(key);
    const collection = armed.get(key);
    if (!session || (collection?.before ?? session.before) === editor.document.getText()) {
      vscode.window.showInformationMessage('Finetune Collector: no tracked edit to capture.');
      return;
    }
    const request = collection?.request ?? await vscode.window.showInputBox({ prompt: '本次编辑要求', ignoreFocusOut: true });
    if (!request?.trim()) return;
    const target = await writeRecord({
      taskType: editor.document.languageId === 'markdown' ? 'prose_edit' : 'code_edit',
      userRequest: request,
      original: collection?.before ?? session.before,
      humanFinal: editor.document.getText(),
      accepted: true,
      sourceFile: vscode.workspace.asRelativePath(editor.document.uri)
    }, 'vscode_manual_edit_capture');
    sessions.set(key, { ...session, before: editor.document.getText(), after: editor.document.getText() });
    armed.delete(key);
    vscode.window.showInformationMessage(`Finetune Collector: saved ${vscode.workspace.asRelativePath(target)}`);
  }));
}

export function deactivate() {}
