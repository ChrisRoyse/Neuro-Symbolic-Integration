// Chatbot.jsx
import { useState, useEffect } from 'preact/hooks';
// Note: In a real implementation, import 'onnxruntime-web' for ONNX model inference
// import * as ort from 'onnxruntime-web';

// Mock function to simulate LLM inference (replace with real ONNX model in production)
function mockModelInference(message) {
  let age = null;
  let income = null;

  // Extract age (e.g., "I am 30 years old")
  const ageMatch = message.match(/I am (\d+) years old/i);
  if (ageMatch) {
    age = parseInt(ageMatch[1], 10);
  }

  // Extract income (e.g., "my income is $60000")
  const incomeMatch = message.match(/my income is \$(\d+)/i);
  if (incomeMatch) {
    income = parseInt(incomeMatch[1], 10);
  }

  if (age !== null && income !== null) {
    return JSON.stringify({ age, income });
  } else {
    return JSON.stringify({ error: "Could not extract age and income" });
  }
}

// Symbolic logic to determine mortgage eligibility
function applyMortgageRules(data) {
  try {
    const { age, income } = JSON.parse(data);
    if (age >= 18 && income >= 50000) {
      return "Congratulations! You are eligible for a mortgage.";
    } else {
      return "Sorry, you do not meet the requirements for a mortgage.";
    }
  } catch (e) {
    return "Sorry, I couldn't understand your request.";
  }
}

function Chatbot() {
  const [session, setSession] = useState(null);
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(true);

  // Simulate loading the ONNX model when the component mounts
  useEffect(() => {
    // In a real scenario, load the ONNX model here:
    /*
    const loadModel = async () => {
      try {
        const modelSession = await ort.InferenceSession.create('/path/to/model.onnx');
        setSession(modelSession);
        setLoading(false);
      } catch (error) {
        console.error('Failed to load model', error);
        setLoading(false);
      }
    };
    loadModel();
    */
    // Mock model loading for demo
    setTimeout(() => {
      setSession({}); // Simulate successful load
      setLoading(false);
    }, 1000);
  }, []);

  // Handle form submission
  const handleSubmit = async () => {
    if (!session) return;
    setLoading(true);

    // In a real scenario, run ONNX model inference here:
    /*
    const inputIds = tokenize(message); // Tokenize input (model-specific)
    const inputTensor = new ort.Tensor('int64', inputIds, [1, inputIds.length]);
    const feeds = { input_ids: inputTensor };
    const outputs = await session.run(feeds);
    const outputData = outputs.output.data; // Adjust based on model output name
    const generatedText = decodeOutput(outputData); // Decode (model-specific)
    const modelOutput = generatedText;
    */
    // Use mock inference for demo
    const modelOutput = mockModelInference(message);

    const finalAnswer = applyMortgageRules(modelOutput);
    setResponse(finalAnswer);
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '500px', margin: '0 auto' }}>
      <h1>Mortgage Chatbot</h1>
      {loading ? (
        <p>Loading model...</p>
      ) : (
        <>
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="E.g., 'I am 30 years old and my income is $60000'"
            style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
            disabled={loading}
          />
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{ padding: '10px 20px' }}
          >
            {loading ? 'Processing...' : 'Ask'}
          </button>
          {response && <p style={{ marginTop: '20px' }}>{response}</p>}
        </>
      )}
    </div>
  );
}

export default Chatbot;