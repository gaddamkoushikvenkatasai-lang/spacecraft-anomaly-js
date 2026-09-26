const tf = require("@tensorflow/tfjs");

function buildAutoencoder(inputDim, config = {}) {
  const { windowSize = 15, hiddenDim = 16, latentDim = 8, learningRate = 0.01 } = config;

  const input = tf.input({ shape: [windowSize, inputDim] });
  const encoded = tf.layers.lstm({ units: latentDim, returnSequences: false }).apply(input);
  const repeated = tf.layers.repeatVector({ n: windowSize }).apply(encoded);
  const decoded = tf.layers.lstm({ units: hiddenDim, returnSequences: true }).apply(repeated);
  const output = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: inputDim }) }).apply(decoded);

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({ optimizer: tf.train.adam(learningRate), loss: "meanSquaredError" });
  return model;
}

function buildPredictor(inputDim, config = {}) {
  const { windowSize = 15, hiddenDim = 16, learningRate = 0.01 } = config;

  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: hiddenDim, returnSequences: false, inputShape: [windowSize, inputDim] }));
  model.add(tf.layers.dense({ units: inputDim, activation: "linear" }));

  model.compile({ optimizer: tf.train.adam(learningRate), loss: "meanSquaredError" });
  return model;
}

function buildModel(type, inputDim, config) {
  return type === "predictor" ? buildPredictor(inputDim, config) : buildAutoencoder(inputDim, config);
}

module.exports = { buildModel, buildAutoencoder, buildPredictor };