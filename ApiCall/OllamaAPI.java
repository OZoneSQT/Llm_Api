import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.*;

import org.json.JSONObject;

public class OllamaAPI {
    public static void main(String[] args) {
        // Default parameters
        String hostApi = "http://localhost:11434";
        String model = "phi3:mini";
        String prompt = "What is the capital of France?";
        boolean printRaw = false;
        boolean enableLogging = false;

        // Parse command-line arguments
        for (String arg : args) {
            if (arg.startsWith("--hostApi=")) hostApi = arg.substring(10);
            else if (arg.startsWith("--model=")) model = arg.substring(8);
            else if (arg.startsWith("--prompt=")) prompt = arg.substring(9);
            else if (arg.equals("--printRaw")) printRaw = true;
            else if (arg.equals("--enableLogging")) enableLogging = true;
        }

        String logFile = "llm.log";

        // Logging function
        Runnable logInit = () -> {
            File f = new File(logFile);
            if (!f.exists()) try { f.createNewFile(); } catch (IOException ignored) {}
        };
        logInit.run();

        voidLog("Run: hostApi='" + hostApi + "', model='" + model + "', prompt='" + prompt + "', printRaw=" + printRaw, enableLogging, logFile);

        try {
            // Prepare JSON body
            JSONObject bodyJson = new JSONObject();
            bodyJson.put("model", model);
            bodyJson.put("prompt", prompt);
            String body = bodyJson.toString();

            // HTTP POST
            URL url = new URL(hostApi + "/api/generate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = body.getBytes(StandardCharsets.UTF8);
                os.write(input, 0, input.length);
            }

            int statusCode = conn.getResponseCode();
            voidLog("STATUS CODE: " + statusCode, enableLogging, logFile);
            System.out.println("STATUS CODE: " + statusCode);

            InputStream is = (statusCode >= 200 && statusCode < 400) ? conn.getInputStream() : conn.getErrorStream();
            String responseString = new String(is.readAllBytes(), StandardCharsets.UTF8);

            StringBuilder fullResponse = new StringBuilder();
            for (String line : responseString.split("\n")) {
                if (!line.trim().isEmpty()) {
                    try {
                        JSONObject json = new JSONObject(line);
                        if (json.has("response")) {
                            fullResponse.append(json.getString("response"));
                        }
                    } catch (Exception ignored) {}
                }
            }

            if (printRaw) {
                System.out.println(responseString);
                voidLog("RAW RESPONSE: " + safeSub(responseString, 200) + "...", enableLogging, logFile);
            } else {
                System.out.println(fullResponse);
                voidLog("FULL RESPONSE: " + safeSub(fullResponse.toString(), 200) + "...", enableLogging, logFile);
            }
        } catch (Exception e) {
            System.out.println("ERROR:");
            e.printStackTrace();
            voidLog("ERROR: " + e, enableLogging, logFile);
        }
    }

    private static void voidLog(String message, boolean enableLogging, String logFile) {
        if (!enableLogging) return;
        try (FileWriter fw = new FileWriter(logFile, true)) {
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
            fw.write(timestamp + " " + message + "\n");
        } catch (IOException ignored) {}
    }

    private static String safeSub(String s, int max) {
        return s.length() > max ? s.substring(0, max) : s;
    }
}