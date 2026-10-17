import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';

const outputDirectory = dirname(fileURLToPath(import.meta.url));
const runIdentifier = new Date().toISOString().replace(/[:.]/g, '-');
const originalPath = '/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/eval_qwen35_ditL_host64_55000_8gpu3workers_20261014/swap_T/seed100000/episode0.mp4';
const expectedOriginalHash = 'af4f717c100080ed09e15ffb0e946cab53cacc80b8e327c85fc7b2f2ebc77911';
const extensionDirectory = '/root/.cursor-server/extensions/yutabee.unmute-video-0.3.0-universal';
const builtinPreviewScript = '/root/.cursor-server/bin/linux-x64/cd6d2a1f2e56e9841f0ed9c7c24542087b4e69b0/extensions/media-preview/media/videoPreview.js';
const testModes = process.argv.includes('--builtin-only')
  ? ['builtin_preview'] : ['http_range', 'blob', 'file'];
const require = createRequire(import.meta.url);
const { StreamServer } = require(join(extensionDirectory, 'out/server/streamServer.js'));

const cases = [
  { name: 'original_mp4', path: originalPath, expectedFrames: 600, expectedDuration: 60 },
  { name: 'faststart_mp4', path: join(outputDirectory, 'episode0_faststart.mp4'), expectedFrames: 600, expectedDuration: 60 },
  { name: 'baseline_mp4', path: join(outputDirectory, 'episode0_baseline.mp4'), expectedFrames: 600, expectedDuration: 60 },
  { name: 'vp8_webm', path: join(outputDirectory, 'episode0_vp8.webm'), expectedFrames: 600, expectedDuration: 60 },
  { name: 'existing_vp9_webm', path: '/mnt/data/dylan/d2/memory/TTTchoiceMem/video_previews/20261014_55k_swap_T_seed100000_episode0.webm', expectedFrames: 315, expectedDuration: 31.5 },
  { name: 'libero_reference', path: '/mnt/data/dylan/d2/memory/RMBench/attention-test-memoryvla/run_full_long10_20261007/full_suite_report/media/task_00/initial_state_000/original_7df1055086ef9b3b.mp4', expectedFrames: 32, expectedDuration: 8 },
];

async function computeFileHash(filePath) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) digest.update(chunk);
  return digest.digest('hex');
}

function inspectVideo(testCase) {
  const metadata = JSON.parse(execFileSync('ffprobe', [
    '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'format=duration,size:stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,nb_read_frames',
    '-of', 'json', testCase.path,
  ], { encoding: 'utf8', timeout: 30000 }));
  assert.equal(Number(metadata.streams[0].nb_read_frames), testCase.expectedFrames);
  assert.ok(Math.abs(Number(metadata.format.duration) - testCase.expectedDuration) < 0.11);
  const decodeErrors = execFileSync('ffmpeg', [
    '-hide_banner', '-v', 'error', '-xerror', '-threads', '1', '-i', testCase.path,
    '-map', '0:v:0', '-an', '-f', 'null', '-',
  ], { encoding: 'utf8', timeout: 30000 });
  assert.equal(decodeErrors, '');
  return { metadata, fullDecodePassed: true };
}

async function inspectPlayer(page) {
  return page.evaluate(() => {
    const video = document.querySelector('video');
    const quality = video.getVideoPlaybackQuality();
    return {
      currentTime: video.currentTime, duration: video.duration,
      paused: video.paused, ended: video.ended, readyState: video.readyState,
      width: video.videoWidth, height: video.videoHeight,
      decodedFrames: video.webkitDecodedFrameCount ?? quality.totalVideoFrames,
      presentedFrames: window.presentedFrames,
      error: video.error ? { code: video.error.code, message: video.error.message } : null,
    };
  });
}

async function initializePlayer(page, mediaUrl, mode) {
  if (mode === 'builtin_preview') {
    const settings = JSON.stringify({ src: mediaUrl, autoplay: false, loop: false })
      .replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    const previewHtml = `<!doctype html><html><head>
      <meta id="settings" data-settings="${settings}"></head>
      <body class="loading"><a class="open-file-link" href="#">Open as text</a></body></html>`;
    // Use a same-origin HTML navigation instead of an opaque about:blank origin.
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      if (request.isNavigationRequest() && request.frame() === page.mainFrame()) {
        void request.respond({ status: 200, contentType: 'text/html', body: previewHtml });
      } else {
        void request.continue();
      }
    });
    await page.goto(mediaUrl, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await page.evaluate(() => {
      // The host bridge is stubbed; the installed preview script is unchanged.
      window.acquireVsCodeApi = () => ({ postMessage() {} });
    });
    await page.addScriptTag({ path: builtinPreviewScript });
    await page.waitForFunction(() => document.body.classList.contains('ready')
      || document.body.classList.contains('error'), { timeout: 15000 });
    assert.equal(await page.evaluate(() => document.body.classList.contains('error')), false);
  } else {
    await page.goto(mediaUrl, { waitUntil: 'domcontentloaded', timeout: 20000 });
  }
  await page.waitForSelector('video', { timeout: 10000 });
  if (mode === 'blob') {
    await page.evaluate(async (sourceUrl) => {
      const response = await fetch(sourceUrl);
      const video = document.querySelector('video');
      video.src = URL.createObjectURL(await response.blob());
      video.load();
    }, mediaUrl);
  }
  await page.evaluate(() => {
    const video = document.querySelector('video');
    video.muted = true;
    video.loop = false;
    window.presentedFrames = 0;
    const countPresentedFrames = () => {
      window.presentedFrames += 1;
      video.requestVideoFrameCallback(countPresentedFrames);
    };
    video.requestVideoFrameCallback(countPresentedFrames);
  });
  await page.waitForFunction(() => {
    const video = document.querySelector('video');
    return video.error || (video.readyState >= 2 && Number.isFinite(video.duration));
  }, { timeout: 15000 });
  const initial = await inspectPlayer(page);
  assert.equal(initial.error, null, JSON.stringify(initial.error));
  assert.ok(initial.width > 0 && initial.height > 0);
  return initial;
}

async function playAndCheckProgress(page) {
  await page.evaluate(async () => {
    const video = document.querySelector('video');
    video.currentTime = 0;
    video.playbackRate = 1;
    await video.play();
  });
  await page.waitForFunction(() => {
    const video = document.querySelector('video');
    return video.error || (video.currentTime >= 0.7 && window.presentedFrames >= 3);
  }, { timeout: 15000 });
  const progressed = await inspectPlayer(page);
  assert.equal(progressed.error, null);
  assert.ok(progressed.currentTime >= 0.7 && progressed.presentedFrames >= 3);
  await page.evaluate(() => document.querySelector('video').pause());
  return progressed;
}

async function seekAndCheckFrame(page, targetTime) {
  await page.evaluate((position) => {
    const video = document.querySelector('video');
    window.framesBeforeSeek = window.presentedFrames;
    video.currentTime = position;
  }, targetTime);
  await page.waitForFunction((position) => {
    const video = document.querySelector('video');
    return video.error || (!video.seeking && Math.abs(video.currentTime - position) < 0.15
      && window.presentedFrames > window.framesBeforeSeek);
  }, { timeout: 15000 }, targetTime);
  const seeked = await inspectPlayer(page);
  assert.equal(seeked.error, null);
  assert.ok(Math.abs(seeked.currentTime - targetTime) < 0.15);
  return seeked;
}

async function verifyBrowserMode(browser, testCase, mediaUrl, mode) {
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  const mediaResponses = [];
  page.on('response', (response) => {
    if (response.url() === mediaUrl) mediaResponses.push({ status: response.status(), headers: response.headers() });
  });
  try {
    await page.setViewport({ width: 900, height: 650 });
    const initial = await initializePlayer(page, mediaUrl, mode);
    assert.ok(Math.abs(initial.duration - testCase.expectedDuration) < 0.11);
    const progress = await playAndCheckProgress(page);
    const result = { mode, passed: true, initial, progress, mediaResponses };
    if (mode === 'http_range' || mode === 'builtin_preview') {
      result.middle = await seekAndCheckFrame(page, testCase.expectedDuration / 2);
      if (testCase.name === 'vp8_webm') {
        result.screenshot = join(outputDirectory, `vp8_${mode}_middle_${runIdentifier}.png`);
        await page.screenshot({ path: result.screenshot });
      }
      result.tail = await seekAndCheckFrame(page, testCase.expectedDuration - 1);
    }
    if (mode === 'http_range') {
      // Run from the beginning to the actual ended event, not just metadata load.
      await page.evaluate(async () => {
        const video = document.querySelector('video');
        video.currentTime = 0;
        video.playbackRate = 4;
        await video.play();
      });
      await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return video.ended || video.error;
      }, { timeout: 90000 });
      result.finished = await inspectPlayer(page);
      assert.equal(result.finished.error, null);
      assert.equal(result.finished.ended, true);
      assert.ok(result.finished.decodedFrames >= testCase.expectedFrames);
      assert.ok(mediaResponses.some((response) => response.status === 206));
    }
    return result;
  } catch (error) {
    return { mode, passed: false, reason: error.message, player: await inspectPlayer(page).catch(() => null), mediaResponses };
  } finally {
    await page.close();
  }
}

async function main() {
  assert.equal(await computeFileHash(originalPath), expectedOriginalHash, 'Original video changed before verification');
  const server = new StreamServer();
  let browser;
  const report = {
    scope: 'Headless Chrome on remote host; NOT the local Cursor webview. Uses the installed extension streaming server unchanged.',
    startedAt: new Date().toISOString(), originalSha256: expectedOriginalHash,
    installedExtension: 'yutabee.unmute-video@0.3.0', testModes,
    builtinPreviewScript, builtinPreviewScriptSha256: await computeFileHash(builtinPreviewScript), cases: [],
  };
  try {
    await server.start();
    browser = await puppeteer.launch({
      executablePath: '/root/.cache/puppeteer/chrome/linux-152.0.7977.42/chrome-linux64/chrome',
      headless: true,
      protocolTimeout: 30000,
      args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--autoplay-policy=no-user-gesture-required'],
    });
    report.browserVersion = await browser.version();
    for (const testCase of cases) {
      console.log(`Testing ${testCase.name}`);
      const record = { ...testCase, sha256: await computeFileHash(testCase.path), ...inspectVideo(testCase), browser: [] };
      const token = server.register(testCase.path);
      const mediaUrl = server.urlFor(token);
      const rangeResponse = await fetch(mediaUrl, {
        headers: { Range: 'bytes=0-31', Connection: 'close' },
      });
      assert.equal(rangeResponse.status, 206);
      assert.equal((await rangeResponse.arrayBuffer()).byteLength, 32);
      record.httpRangePassed = true;
      for (const mode of testModes) {
        const sourceUrl = mode === 'file' ? pathToFileURL(testCase.path).href : mediaUrl;
        const result = await verifyBrowserMode(browser, testCase, sourceUrl, mode);
        record.browser.push(result);
        console.log(JSON.stringify({ case: testCase.name, mode, passed: result.passed, reason: result.reason ?? null }));
      }
      server.unregister(token);
      report.cases.push(record);
    }
    const computePacketHash = (filePath) => execFileSync('ffmpeg', [
      '-v', 'error', '-i', filePath, '-map', '0:v:0', '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-',
    ], { encoding: 'utf8', timeout: 30000 }).trim();
    report.faststartEncodedPacketsUnchanged = computePacketHash(originalPath) === computePacketHash(join(outputDirectory, 'episode0_faststart.mp4'));
    assert.equal(report.faststartEncodedPacketsUnchanged, true);
    report.originalUnchanged = await computeFileHash(originalPath) === expectedOriginalHash;
    assert.equal(report.originalUnchanged, true);
  } finally {
    if (browser) await browser.close();
    server.dispose();
    report.finishedAt = new Date().toISOString();
    const reportPath = join(outputDirectory, `playback_results_${runIdentifier}.json`);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
    console.log(`Playback evidence: ${reportPath}`);
  }
  const vp8Result = report.cases.find((record) => record.name === 'vp8_webm');
  assert.ok(vp8Result?.browser.every((result) => result.passed), 'VP8 WebM did not pass every browser loading mode');
}

await main();
