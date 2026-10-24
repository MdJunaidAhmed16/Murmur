// Records the README demo: the real page in headless Chrome, a scripted voice as the microphone.
//
// Video: puppeteer's screencast. Audio: recorded inside the page from its own audio graph, the
// agent's playback and the microphone mixed, so both voices stay in sync with what is shown.
// ffmpeg then joins them.
//
//   node scripts/record_demo/record.cjs <url> <mic.wav> <out-dir> <seconds> <chrome> <ffmpeg>
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const puppeteer = require("puppeteer-core");

const [url, micWav, outDir, seconds, chrome, ffmpeg] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const video = path.join(outDir, "screen.webm");
  const audio = path.join(outDir, "audio.webm");
  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: "new",
    defaultViewport: { width: 1280, height: 900 },
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${micWav}`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load" });

  const screen = await page.screencast({ path: video, ffmpegPath: ffmpeg });
  const videoStart = Date.now();
  await page.click("#start");
  // The page creates its audio contexts once the server's config arrives.
  await page.waitForFunction(
    () => typeof playbackNode !== "undefined" && playbackNode && micStream,
    { timeout: 30000 },
  );
  const audioStart = await page.evaluate(() => {
    const mix = playbackCtx.createMediaStreamDestination();
    playbackNode.connect(mix);
    playbackCtx.createMediaStreamSource(micStream).connect(mix);
    window.__chunks = [];
    window.__rec = new MediaRecorder(mix.stream, { mimeType: "audio/webm;codecs=opus" });
    window.__rec.ondataavailable = (e) => window.__chunks.push(e.data);
    window.__rec.start(1000);
    return Date.now();
  });

  await sleep(Number(seconds) * 1000);
  const b64 = await page.evaluate(async () => {
    window.__rec.stop();
    await new Promise((r) => setTimeout(r, 1000));
    const bytes = new Uint8Array(await new Blob(window.__chunks).arrayBuffer());
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(s);
  });
  await page.screenshot({ path: path.join(outDir, "poster.png") });
  await screen.stop();
  await browser.close();
  fs.writeFileSync(audio, Buffer.from(b64, "base64"));

  const offset = ((audioStart - videoStart) / 1000).toFixed(3);
  const out = path.join(outDir, "demo.mp4");
  execFileSync(ffmpeg, [
    "-y", "-loglevel", "error",
    "-i", video,
    "-itsoffset", offset, "-i", audio,
    "-map", "0:v", "-map", "1:a",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
    "-vf", "scale=1280:-2",
    "-c:a", "aac", "-b:a", "96k",
    "-shortest", out,
  ]);
  console.log(`wrote ${out} (audio offset ${offset}s)`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
