const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs-extra");
const { LABELS_PATH, TEST_DIR, generateDataset } = require("./data_loader");
const { runChannel, PAPER_RESULTS, PAPER_CHANNEL_F1, DEFAULT_CONFIG } = require("./pipeline");

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.static(path.join(__dirname, "public")));

async function ensureData() {
  if (!fs.existsSync(LABELS_PATH)) {
    console.log("Generating dataset...");
    await generateDataset();
  }
}

app.get("/api/channels", async (req, res) => {
  try {
    await ensureData();
    const labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
    const channels = Object.entries(labels).map(([id, info]) => {
      let anomalyPct = 0;
      try {
        const test = JSON.parse(fs.readFileSync(path.join(TEST_DIR, `${id}.json`), "utf-8"));
        let ac = 0;
        for (const [s, e] of info.anomalyRanges) ac += Math.min(e, test.length - 1) - Math.max(s, 0) + 1;
        anomalyPct = +((ac / test.length) * 100).toFixed(2);
      } catch (e) {}
      return {
        id,
        spacecraft: info.spacecraft,
        trainLength: info.trainLength,
        testLength: info.testLength,
        features: info.features,
        anomalyRanges: info.anomalyRanges.length,
        anomalyPct,
        paperF1: PAPER_CHANNEL_F1[id] || null,
      };
    });
    res.json({ channels, paperResults: PAPER_RESULTS });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/channels/:id/data", async (req, res) => {
  try {
    const cid = req.params.id;
    const labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
    const testRaw = JSON.parse(fs.readFileSync(path.join(TEST_DIR, `${cid}.json`), "utf-8"));
    const chInfo = labels[cid];
    const labelArr = new Array(testRaw.length).fill(0);
    if (chInfo) for (const [s, e] of chInfo.anomalyRanges) for (let i = Math.max(0, s); i <= Math.min(testRaw.length - 1, e); i++) labelArr[i] = 1;
    res.json({
      chanId: cid,
      spacecraft: chInfo?.spacecraft,
      testData: testRaw,
      labels: labelArr,
      anomalyRanges: chInfo?.anomalyRanges || [],
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/channels/:id/run", async (req, res) => {
  try {
    const cid = req.params.id;
    const { modelType = "autoencoder", config = {} } = req.body;
    await ensureData();
    const labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
    if (!labels[cid]) return res.status(404).json({ error: "Channel not found" });

    const result = await runChannel(cid, labels[cid], { ...DEFAULT_CONFIG, ...config }, modelType);

    fs.ensureDirSync(path.join(__dirname, "results"));
    fs.writeFileSync(path.join(__dirname, "results", "latest_result.json"), JSON.stringify(result));
    res.json(result);
  } catch (err) {
    console.error("Run error:", err);
    res.status(500).json({ error: err.message || "Detection failed" });
  }
});

async function start() {
  await ensureData();
  app.listen(PORT, () => {
    console.log("\n========================================");
    console.log("  Spacecraft Anomaly Detection Dashboard");
    console.log("========================================");
    console.log(`\n  Open: http://localhost:${PORT}\n`);
  });
}

start().catch(console.error);