'use strict';

const { randomBytes } = require('node:crypto');

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
}

function createVideoHtml(videoBytes, mimeType, fileName) {
  if (!['video/mp4', 'video/webm'].includes(mimeType)) {
    throw new Error('Unsupported video container');
  }
  const nonce = randomBytes(16).toString('hex');
  const dataSource = `data:${mimeType};base64,${Buffer.from(videoBytes).toString('base64')}`;
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src data:; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
<title>${escapeHtml(fileName)}</title>
<style nonce="${nonce}">
body { margin: 24px; color: #eee; background: #181818; font-family: sans-serif; }
video { display: block; width: min(100%, 960px); max-height: 75vh; background: #000; }
button { margin: 12px 0; padding: 8px 18px; cursor: pointer; }
pre { white-space: pre-wrap; font-size: 13px; }
</style></head><body>
<h2>${escapeHtml(fileName)}</h2>
<video id="video" controls playsinline preload="auto"></video>
<button id="play" type="button">Play / Pause</button>
<pre id="status">Loading embedded video...</pre>
<p>Read-only. No localhost server, remote resource URL, network request, or codec installation.</p>
<script nonce="${nonce}">
'use strict';
const video = document.getElementById('video');
const status = document.getElementById('status');
window.offlinePreviewState = { presentedFrames: 0 };
function updateStatus() {
  const quality = video.getVideoPlaybackQuality();
  const state = {
    currentTime: video.currentTime, duration: Number.isFinite(video.duration) ? video.duration : null,
    paused: video.paused, ended: video.ended, readyState: video.readyState,
    width: video.videoWidth, height: video.videoHeight,
    decodedFrames: video.webkitDecodedFrameCount ?? quality.totalVideoFrames,
    presentedFrames: window.offlinePreviewState.presentedFrames,
    error: video.error ? { code: video.error.code, message: video.error.message } : null,
    support: {
      vp8: video.canPlayType('video/webm; codecs="vp8"'),
      vp9: video.canPlayType('video/webm; codecs="vp9"'),
      h264: video.canPlayType('video/mp4; codecs="avc1.42E01E"')
    }
  };
  window.offlinePreviewState = state;
  status.textContent = JSON.stringify(state, null, 2);
}
for (const eventName of ['loadedmetadata', 'loadeddata', 'canplay', 'timeupdate', 'play', 'pause', 'seeked', 'ended', 'error']) {
  video.addEventListener(eventName, updateStatus);
}
if (video.requestVideoFrameCallback) {
  const countFrame = () => {
    window.offlinePreviewState.presentedFrames += 1;
    video.requestVideoFrameCallback(countFrame);
  };
  video.requestVideoFrameCallback(countFrame);
}
document.getElementById('play').addEventListener('click', async () => {
  try {
    if (video.paused) await video.play(); else video.pause();
  } catch (error) {
    status.textContent = error.message;
  }
});
video.src = '${dataSource}';
video.load();
</script></body></html>`;
}

module.exports = { createVideoHtml };
