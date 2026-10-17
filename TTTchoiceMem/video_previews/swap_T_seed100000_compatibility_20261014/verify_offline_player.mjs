import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const directory = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { createVideoHtml } = require(join(directory, 'offline-video-preview/playerHtml.js'));
const runIdentifier = new Date().toISOString().replace(/[:.]/g, '-');
const candidates = ['episode0_vp8.webm', 'episode0_faststart.mp4', 'episode0_baseline.mp4'];
const report = { scope: 'Embedded extension HTML under its restrictive CSP in headless Chrome, not the local Cursor runtime.', cases: [] };

const browser = await puppeteer.launch({
  executablePath: '/root/.cache/puppeteer/chrome/linux-152.0.7977.42/chrome-linux64/chrome',
  headless: true, protocolTimeout: 30000,
  args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});
try {
  report.browserVersion = await browser.version();
  for (const fileName of candidates) {
    console.log(`Testing offline ${fileName}`);
    const bytes = await readFile(join(directory, fileName));
    const mimeType = fileName.endsWith('.webm') ? 'video/webm' : 'video/mp4';
    const html = createVideoHtml(bytes, mimeType, fileName);
    const page = await browser.newPage();
    const networkRequests = [];
    page.on('request', (request) => {
      if (/^https?:/.test(request.url())) networkRequests.push(request.url());
    });
    const record = { fileName, networkRequests };
    try {
      await page.setViewport({ width: 960, height: 900 });
      await page.setContent(html, { waitUntil: 'domcontentloaded', timeout: 20000 });
      await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return video.error || video.readyState >= 2;
      }, { timeout: 15000 });
      record.loaded = await page.evaluate(() => window.offlinePreviewState);
      assert.equal(record.loaded.error, null);
      assert.equal(record.loaded.duration, 60);
      await page.click('#play');
      await page.waitForFunction(() => document.querySelector('video').currentTime >= 1, { timeout: 15000 });
      record.playing = await page.evaluate(() => window.offlinePreviewState);
      assert.ok(record.playing.presentedFrames >= 3);
      await page.evaluate(() => {
        const video = document.querySelector('video');
        video.pause();
        video.currentTime = 30;
      });
      await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return !video.seeking && video.currentTime === 30;
      }, { timeout: 15000 });
      record.middle = await page.evaluate(() => window.offlinePreviewState);
      if (fileName.endsWith('.webm')) {
        record.screenshot = join(directory, `offline_vp8_${runIdentifier}.png`);
        await page.screenshot({ path: record.screenshot });
        const htmlPath = join(directory, 'episode0_offline.html');
        await writeFile(htmlPath, html, { flag: 'wx' });
        record.standaloneHtml = htmlPath;
      }
      await page.evaluate(async () => {
        const video = document.querySelector('video');
        video.currentTime = 0;
        video.playbackRate = 4;
        await video.play();
      });
      await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return video.ended || video.error;
      }, { timeout: 45000 });
      record.finished = await page.evaluate(() => window.offlinePreviewState);
      assert.equal(record.finished.ended, true);
      assert.equal(record.finished.error, null);
      assert.ok(record.finished.decodedFrames >= 600);
      assert.equal(networkRequests.length, 0);
      record.passed = true;
    } catch (error) {
      record.passed = false;
      record.reason = error.message;
    } finally {
      await page.close();
    }
    report.cases.push(record);
    console.log(JSON.stringify({ fileName, passed: record.passed, reason: record.reason ?? null }));
  }
} finally {
  await browser.close();
  const reportPath = join(directory, `offline_playback_results_${runIdentifier}.json`);
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  console.log(`Evidence: ${reportPath}`);
}
assert.ok(report.cases.every((record) => record.passed), 'An embedded playback candidate failed');
