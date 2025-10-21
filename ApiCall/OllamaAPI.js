const fs = require('fs');
const https = require('https');
const http = require('http');
const { URL } = require('url');

// Default parameters (can be overridden by environment variables or CLI args)
const hostApi = process.env.HOST_API || "http://localhost:11434";
const model = process.env.MODEL || "phi3:mini";
const prompt = process.env.PROMPT || "What is the capital of France?";
const printRaw = process.env.PRINT_RAW === "true";
const enableLogging = process.env.ENABLE_LOGGING === "true";

const logFile = `${__dirname}/llm.log`;

function writeLog(message) {
    const timestamp = new Date().toISOString().replace('T', ' ').replace('Z', '');
    fs.appendFileSync(logFile, `${timestamp} ${message}\n`);
}

function postJson(url, data, callback) {
    const parsedUrl = new URL(url);
    const isHttps = parsedUrl.protocol === 'https:';
    const options = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port,
        path: parsedUrl.pathname + parsedUrl.search,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(data)
        }
    };
    const reqModule = isHttps ? https : http;
    const req = reqModule.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => callback(null, res.statusCode, body));
    });
    req.on('error', (e) => callback(e));
    req.write(data);
    req.end();
}

const body = JSON.stringify({ model, prompt });

if (enableLogging) writeLog(`Run: hostApi='${hostApi}', model='${model}', prompt='${prompt}', printRaw=${printRaw}`);

postJson(`${hostApi}/api/generate`, body, (err, statusCode, responseString) => {
    if (err) {
        console.error("ERROR:", err);
        if (enableLogging) writeLog(`ERROR: ${err}`);
        return;
    }
    if (enableLogging) writeLog(`STATUS CODE: ${statusCode}`);
    console.log(`STATUS CODE: ${statusCode}`);

    let fullResponse = '';
    const lines = responseString.split('\n');
    for (const line of lines) {
        if (line.trim()) {
            try {
                const json = JSON.parse(line);
                if (json.response) fullResponse += json.response;
            } catch {}
        }
    }
    if (printRaw) {
        console.log(responseString);
        if (enableLogging) writeLog(`RAW RESPONSE: ${responseString.substring(0, 200)}...`);
    } else {
        console.log(fullResponse);
        if (enableLogging) writeLog(`FULL RESPONSE: ${fullResponse.substring(0, 200)}...`);
    }
});