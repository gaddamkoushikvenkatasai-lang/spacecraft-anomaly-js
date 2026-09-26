const fs = require("fs-extra");
const path = require("path");

const DATA_DIR = path.join(__dirname, "data");
const TRAIN_DIR = path.join(DATA_DIR, "train");
const TEST_DIR = path.join(DATA_DIR, "test");
const LABELS_PATH = path.join(DATA_DIR, "labeled_anomalies.json");
const NUM_FEATURES = 25;

class SeededRandom {
  constructor(seed) { this.seed = seed; }
  next() {
    this.seed = (this.seed * 16807) % 2147483647;
    return this.seed / 2147483647;
  }
  randInt(min, max) { return Math.floor(this.next() * (max - min + 1)) + min; }
  randFloat(min, max) { return this.next() * (max - min) + min; }
  gaussRand() {
    const u1 = this.next(), u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
  choice(arr) { return arr[this.randInt(0, arr.length - 1)]; }
}

function generateChannelNames() {
  const prefixes = ["P","S","E","A","D","G","T","F","C","M","R"];
  const counts = [7,2,13,9,9,4,9,7,2,7,2];
  const channels = [];
  for (let p = 0; p < prefixes.length; p++)
    for (let i = 1; i <= counts[p]; i++)
      channels.push({ id: `${prefixes[p]}-${i}`, spacecraft: "SMAP" });
  for (let i = 1; i <= 27; i++)
    channels.push({ id: `MSL-${i}`, spacecraft: "MSL" });
  return channels;
}

function generateSignal(length, rng) {
  const data = [];
  const baseFreq = rng.randFloat(0.5, 2.5);
  const amplitude = rng.randFloat(0.3, 1.0);
  for (let t = 0; t < length; t++) {
    const row = [];
    const tNorm = (t / length) * 4 * Math.PI;
    for (let f = 0; f < NUM_FEATURES; f++) {
      const freqMod = baseFreq * (1 + 0.1 * f);
      const phase = rng.randFloat(0, 2 * Math.PI);
      row.push(+(amplitude * Math.sin(freqMod * tNorm + phase) +
        0.1 * Math.sin(3 * freqMod * tNorm + phase) +
        0.02 * tNorm + rng.gaussRand() * 0.05).toFixed(6));
    }
    data.push(row);
  }
  return data;
}

function injectAnomalies(data, rng) {
  const ranges = [];
  const numA = rng.randInt(1, 3);
  for (let a = 0; a < numA; a++) {
    const start = rng.randInt(100, data.length - 500);
    const end = Math.min(start + rng.randInt(50, 300), data.length - 1);
    ranges.push([start, end]);
    const type = rng.choice(["spike","drift","dropout","noise"]);
    const feats = [];
    while (feats.length < rng.randInt(1, 5)) {
      const f = rng.randInt(0, NUM_FEATURES - 1);
      if (!feats.includes(f)) feats.push(f);
    }
    for (const af of feats) {
      for (let t = start; t <= end; t++) {
        const amp = 0.3 + rng.next() * 0.7;
        if (type === "spike") data[t][af] += (rng.next() > 0.5 ? 1 : -1) * 3 * amp;
        else if (type === "drift") data[t][af] += ((t - start) / (end - start)) * 4 * amp;
        else if (type === "dropout") data[t][af] = 0;
        else data[t][af] += rng.gaussRand() * 2 * amp;
        data[t][af] = +data[t][af].toFixed(6);
      }
    }
  }
  return ranges;
}

async function generateDataset() {
  console.log("Generating NASA-format telemetry dataset...");
  fs.ensureDirSync(TRAIN_DIR);
  fs.ensureDirSync(TEST_DIR);
  const channels = generateChannelNames();
  const rng = new SeededRandom(42);
  const labels = {};
  for (let i = 0; i < channels.length; i++) {
    const ch = channels[i];
    const trainLen = rng.randInt(6000, 9000);
    const testLen = rng.randInt(4000, 7000);
    process.stdout.write(`\r  Channel ${i + 1}/${channels.length}: ${ch.id}   `);
    const trainData = generateSignal(trainLen, rng);
    const testData = generateSignal(testLen, rng);
    const anomalyRanges = injectAnomalies(testData, rng);
    fs.writeFileSync(path.join(TRAIN_DIR, `${ch.id}.json`), JSON.stringify(trainData));
    fs.writeFileSync(path.join(TEST_DIR, `${ch.id}.json`), JSON.stringify(testData));
    labels[ch.id] = { spacecraft: ch.spacecraft, trainLength: trainLen, testLength: testLen, features: NUM_FEATURES, anomalyRanges };
  }
  fs.writeFileSync(LABELS_PATH, JSON.stringify(labels, null, 2));
  console.log(`\nDone! ${channels.length} channels generated in ${DATA_DIR}`);
}

if (require.main === module) generateDataset().catch(console.error);
module.exports = { generateDataset, DATA_DIR, TRAIN_DIR, TEST_DIR, LABELS_PATH };