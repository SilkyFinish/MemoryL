'use strict';

const vscode = require('vscode');
const path = require('node:path');
const { createVideoHtml } = require('./playerHtml');

const maximumVideoBytes = 32 * 1024 * 1024;

function activate(context) {
  const provider = {
    openCustomDocument(uri) {
      return { uri, dispose() {} };
    },
    async resolveCustomEditor(document, panel, cancellation) {
      panel.webview.options = { enableScripts: true, localResourceRoots: [] };
      try {
        const metadata = await vscode.workspace.fs.stat(document.uri);
        if (metadata.size > maximumVideoBytes) {
          throw new Error('Embedded preview is limited to videos of 32 MiB or less.');
        }
        const videoBytes = await vscode.workspace.fs.readFile(document.uri);
        if (cancellation.isCancellationRequested) return;
        const extension = path.extname(document.uri.path).toLowerCase();
        const mimeType = extension === '.webm' ? 'video/webm' : 'video/mp4';
        panel.webview.html = createVideoHtml(videoBytes, mimeType, path.basename(document.uri.path));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        void vscode.window.showErrorMessage(`Offline Video Preview: ${message}`);
        panel.webview.html = '<!doctype html><html><body><p>Video could not be loaded. See the editor notification for details.</p></body></html>';
      }
    },
  };
  context.subscriptions.push(vscode.window.registerCustomEditorProvider(
    'memoryOfflineVideo.preview', provider,
    { supportsMultipleEditorsPerDocument: true, webviewOptions: { retainContextWhenHidden: true } },
  ));
}

module.exports = { activate };
