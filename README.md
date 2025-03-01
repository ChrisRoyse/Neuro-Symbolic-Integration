# Neuro-Symbolic AI with ONNX and Preact

This repository provides a comprehensive guide and sample code for building a neuro-symbolic AI system that integrates a tiny distilled Large Language Model (LLM) with symbolic reasoning in a Preact web application using Deno. The system leverages ONNX Runtime Web to run the neural model in the browser, ensuring low latency and privacy, while symbolic logic handles rule-based reasoning.

---

## Table of Contents

- [Architectural Design](#architectural-design)
  - [Neuro-Symbolic Integration](#neuro-symbolic-integration)
  - [Client-Server or In-Browser Setup](#client-server-or-in-browser-setup)
  - [Sample Code](#sample-code)
    - [Loading and Running an ONNX Model](#loading-and-running-an-onnx-model)
    - [Using Neuro-Symbolic Reasoning](#using-neuro-symbolic-reasoning)
    - [Integrating into a Preact Component](#integrating-into-a-preact-component)
- [Implementation Guide](#implementation-guide)
  - [1. Prepare or Obtain a Distilled LLM ONNX Model](#1-prepare-or-obtain-a-distilled-llm-onnx-model)
  - [2. Set Up ONNX Runtime in Deno/Preact Environment](#2-set-up-onnx-runtime-in-denopreact-environment)
  - [3. Initialize the ONNX Inference Session](#3-initialize-the-onnx-inference-session)
  - [4. Data Preprocessing](#4-data-preprocessing)
  - [5. Running Inference](#5-running-inference)
  - [6. Symbolic Post-processing and Reasoning](#6-symbolic-post-processing-and-reasoning)
  - [7. Enhancing Reasoning in Tiny Models](#7-enhancing-reasoning-in-tiny-models)
  - [8. Deployment Considerations (Browser Efficiency)](#8-deployment-considerations-browser-efficiency)
- [Optimization Techniques](#optimization-techniques)
  - [Enhancing Model Intelligence](#enhancing-model-intelligence)
  - [Performance Optimizations (ONNX in Browser)](#performance-optimizations-onnx-in-browser)

---

## Architectural Design

### Neuro-Symbolic Integration

A neuro-symbolic AI system combines neural network models (e.g., a tiny LLM) with symbolic reasoning (rule-based logic) to leverage the strengths of both approaches. In practice, this means structuring your application into two complementary layers:

- **Neural Inference Layer**: Powered by an ONNX model, this layer handles pattern recognition or language understanding tasks, such as processing free-form text or extracting meaning from unstructured data.
- **Symbolic Reasoning Layer**: This layer applies explicit logical rules, constraints, or queries against a knowledge base to ensure reliability, transparency, and domain-specific control.

For example, a modular architecture like the MRKL system uses an LLM for language tasks but delegates factual checks or tool invocations to symbolic modules. This separation allows the neural model to focus on what it excels at—handling unstructured data—while the symbolic layer enforces precision and logic.

### Client-Server or In-Browser Setup

In a Preact web app with Deno, you have flexibility in where computation occurs:

- **In-Browser Execution**: The ONNX model runs entirely in the browser using ONNX Runtime Web (via WebAssembly or WebGL), delivering low latency and enhanced privacy since data never leaves the client. The user’s browser downloads the model and executes it locally.
- **Server-Side Logic**: Deno can serve content or handle additional logic on the server. Symbolic reasoning can run either in the browser (using JavaScript alongside the model) or on the server (via Deno), depending on your needs—for instance, if you’re working with a heavy knowledge base or want to keep rules private.

#### Common Architecture Workflow

1. **Input Processing**: User input is parsed or preprocessed. Simple symbolic rules might route the input—e.g., deciding which model to use or checking if it matches a known command or fact.
2. **Neural Inference**: The ONNX model (a tiny distilled LLM) is invoked in the browser via ONNX Runtime. The input is converted into the model’s expected tensor format and fed into an `InferenceSession` for prediction.
3. **Symbolic Reasoning Post-processing**: The raw model output is passed through rule-based logic to apply business rules, enforce constraints, or interpret results logically. For example, if the model proposes an answer, the symbolic layer might verify it against a knowledge graph or use if/then rules to determine the final response.
4. **Output Generation**: The combined neural and symbolic outputs produce the final answer or action, which is then presented to the user via the Preact UI.

This design enables the neural model to tackle unstructured tasks (e.g., understanding free-form text) while the symbolic system provides reliability, transparency, and domain-specific control. The ONNX Runtime in the browser acts as a callable function: the symbolic logic can invoke the model’s `session.run()` method with inputs and use the results in logical expressions. The neural and symbolic components interact by exchanging data—e.g., the symbolic layer might format a prompt for the LLM or interpret its output. In code, this could be organized with an “AI service” module that wraps model calls and applies rules, which Preact components can invoke. Keeping interfaces clear is key: the symbolic layer should define what it needs from the model (e.g., a classification or extracted facts), treating the model as a black-box predictor. This separation enhances maintainability.

### Sample Code

Below are simplified code snippets demonstrating how to load and run an ONNX model in Deno/the browser, and how to integrate rule-based reasoning and model inference into a Preact app.

#### Loading and Running an ONNX Model

First, load the ONNX Runtime Web library and the model. In a Deno environment, use ES module imports (Deno supports npm packages or URLs):

```javascript
// Import ONNX Runtime Web (for Deno, use npm specifier or CDN URL)
import * as ort from 'npm:onnxruntime-web';  // Deno's npm compatibility

// Optional: set path to WASM assets if not using default CDN
ort.env.wasm.wasmPaths = "./onnxruntime-web/";
ort.env.wasm.numThreads = 1;  // e.g., use single-thread (or more if supported)

console.log("Loading model...");
const session = await ort.InferenceSession.create("model.onnx", {
  executionProviders: ["wasm"]  // Use WebAssembly backend (CPU)
});
console.log("Model loaded");
```

In a browser app, `"model.onnx"` could be a URL or local path to a model file shipped with your app. Deno requires `--allow-net` permission if loading from a URL. Alternatively, include ONNX Runtime via a script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

After initialization, run inference:

```javascript
// Prepare input data for the model
let inputData = Float32Array.from(inputArray);  // Example for numeric data
const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);  // e.g., image tensor 1x3x224x224

// The feeds object keys should match the ONNX model's input names
const feeds = { input: inputTensor };

// Run inference
const results = await session.run(feeds);
// Assume model has output named 'output'; get output tensor data
const outputData = results.output.data;

console.log("Model output:", outputData);
```

This creates a tensor for the model’s input and feeds it into the session. Input and output names ("input" and "output" here) must match your model’s spec. For an LLM, `inputData` might be token IDs instead of pixel data.

#### Using Neuro-Symbolic Reasoning

Apply symbolic rules to the model’s output. Suppose the tiny LLM outputs a JSON string with fields (e.g., for a mortgage chatbot):

```javascript
// Example: symbolic post-processing of model output
const resultText = decodeOutput(outputData);  // Decode tensor to text if needed
let finalAnswer;
try {
  const answerObj = JSON.parse(resultText);
  // Apply rules: e.g., ensure age >= 18 and income >= 50000 for a loan
  if (answerObj.age < 18) {
    finalAnswer = "Sorry, the applicant is too young for a mortgage.";
  } else if (answerObj.income < 50000) {
    finalAnswer = "Sorry, income is insufficient for a mortgage.";
  } else {
    finalAnswer = "Mortgage can be granted to " + answerObj.firstName + " " + answerObj.lastName;
  }
} catch (e) {
  // Fallback to raw model output if parsing fails
  finalAnswer = resultText;
}
```

Here, the neural model extracts fields like `firstName`, `lastName`, `age`, and `income` from unstructured text, and the symbolic layer enforces business rules.

#### Integrating into a Preact Component

Wire the model into a Preact UI using hooks:

```jsx
import { useState, useEffect } from 'preact/hooks';
import * as ort from 'npm:onnxruntime-web';

function Chatbot() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [answer, setAnswer] = useState("");

  // Load ONNX model once on mount
  useEffect(() => {
    ort.InferenceSession.create('/models/tiny-llm.onnx', { executionProviders: ['wasm'] })
      .then(sess => {
        setSession(sess);
        setLoading(false);
      })
      .catch(err => console.error("Failed to load model", err));
  }, []);

  const handleAsk = async (userQuestion) => {
    if (!session) return;
    // Preprocess: e.g., tokenize userQuestion to model input tensor
    const inputIds = tokenize(userQuestion);  // Implement tokenize for your model
    const inputTensor = new ort.Tensor('int64', inputIds, [1, inputIds.length]);
    // Run the model
    const outputs = await session.run({ input_ids: inputTensor });
    const rawOutput = outputs.logits.data;  // Example output
    const text = decodeText(rawOutput);  // Convert tokens to string
    const finalAnswer = applyRulesToAnswer(text);  // Symbolic reasoning on text
    setAnswer(finalAnswer);
  };

  return (
    <div>
      {loading ? <p>Loading AI model...</p> :
        <div>
          {/* UI for user to input question */}
          <button onClick={() => handleAsk("Can I get a mortgage?")}>Ask</button>
          <p>Answer: {answer}</p>
        </div>
      }
    </div>
  );
}
```

The model loads once at startup, and inference runs on user interaction. For responsiveness, consider offloading heavy inference to a Web Worker.

---

## Implementation Guide

Follow these steps to implement the neuro-symbolic AI system in your Preact web app with Deno.

### 1. Prepare or Obtain a Distilled LLM ONNX Model

- Use a pre-trained tiny LLM (e.g., DistilBERT, MiniLM, or Microsoft’s Phi-3 mini).
- Convert it to ONNX format using tools like [Hugging Face Optimum](https://github.com/huggingface/optimum):
  ```bash
  optimum-cli export onnx --model <model-name> ./model_dir/
  ```
- Optionally, apply quantization to reduce size (see [Optimization Techniques](#optimization-techniques)).

### 2. Set Up ONNX Runtime in Deno/Preact Environment

- In your Deno/Preact project, import ONNX Runtime Web:
  ```javascript
  import * as ort from 'npm:onnxruntime-web';
  ```
- Or include it via a script tag:
  ```html
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
  ```
- Ensure WASM files are accessible (set `ort.env.wasm.wasmPaths` if self-hosting).

### 3. Initialize the ONNX Inference Session

- Load the model at app startup:
  ```javascript
  const session = await ort.InferenceSession.create('model.onnx', {
    executionProviders: ['wasm'],  // Use WebAssembly backend (CPU)
  });
  ```
- Specify execution providers (e.g., `['wasm', 'webgl']`) to leverage GPU if available.

### 4. Data Preprocessing

- Prepare inputs in the format expected by the model (e.g., tokenization for LLMs). Use libraries like [transformers.js](https://github.com/xenova/transformers.js) for browser-based tokenization.
- Symbolic logic might append special tokens or context based on rules.

### 5. Running Inference

- Create an input tensor and run the model:
  ```javascript
  const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const outputData = results.output.data;
  ```
- For LLMs, outputs might be logits or token IDs requiring further processing.

### 6. Symbolic Post-processing and Reasoning

- Apply rule-based logic to the model’s output (see [Using Neuro-Symbolic Reasoning](#using-neuro-symbolic-reasoning)).
- Use iterative neural-symbolic interplay—e.g., refine inputs and rerun the model if rules detect issues.

### 7. Enhancing Reasoning in Tiny Models

- **Prompt Engineering**: Include context or examples in the input to guide reasoning.
- **Chain-of-Thought Decomposition**: Break complex queries into subqueries handled by the model or rules.
- **Fine-Tuning**: Train on domain-specific data or distill from a larger model.
- **External Knowledge**: Fetch facts or context to enrich inputs.

### 8. Deployment Considerations (Browser Efficiency)

- **Model Size**: Use distillation and quantization to minimize size.
- **WASM Optimizations**: Enable SIMD and multi-threading (e.g., `ort.env.wasm.numThreads = navigator.hardwareConcurrency`).
- **WebGL/WebGPU**: Use GPU backends for larger models if supported.
- **Caching**: Cache the model with service workers.
- **Web Workers**: Offload inference to keep the UI responsive.
- **Memory Management**: Avoid retaining large tensors unnecessarily.

---

## Optimization Techniques

### Enhancing Model Intelligence

- **Knowledge Distillation & Fine-Tuning**: Train the tiny model on data from a larger model to preserve reasoning abilities.
- **Rule-Guided Generation**: Apply symbolic constraints during output generation.
- **External Knowledge and Memory**: Fetch context to provide richer inputs.
- **Step-by-Step Reasoning**: Use symbolic logic to break down complex queries into manageable steps.

### Performance Optimizations (ONNX in Browser)

- **Quantization**: Reduce model size and speed up inference (e.g., INT8 or 4-bit).
- **Graph Optimizations and Fusion**: Use tools like Olive to streamline the model graph.
- **Execution Providers**: Test WASM, WebGL, and WebGPU for optimal performance.
- **Memory and Compute Efficiency**: Limit input lengths and avoid unnecessary batching.
- **Lazy Loading**: Load the model on demand to reduce initial load times.
- **Profiling and Monitoring**: Use browser dev tools to identify bottlenecks.

---

By following this guide, you can build a robust neuro-symbolic AI system that combines neural networks and symbolic reasoning, delivering intelligent, reliable features directly in the browser using ONNX Runtime Web, Preact, and Deno.
```
