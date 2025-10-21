using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class OllamaAPI
{
    static async Task Main(string[] args)
    {
        // Default parameters
        string hostApi = "http://localhost:11434";
        string model = "phi3:mini";
        string prompt = "What is the capital of France?";
        bool printRaw = false;
        bool enableLogging = false;

        // Parse command-line arguments (optional)
        foreach (var arg in args)
        {
            if (arg.StartsWith("-hostApi=")) hostApi = arg.Substring(9);
            else if (arg.StartsWith("-model=")) model = arg.Substring(7);
            else if (arg.StartsWith("-prompt=")) prompt = arg.Substring(8);
            else if (arg == "-printRaw") printRaw = true;
            else if (arg == "-enableLogging") enableLogging = true;
        }

        string logFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "llm.log");
        void WriteLog(string message)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            File.AppendAllText(logFile, $"{timestamp} {message}{Environment.NewLine}");
        }

        if (enableLogging)
            WriteLog($"Run: hostApi='{hostApi}', model='{model}', prompt='{prompt}', printRaw={printRaw}");

        var bodyObj = new { model, prompt };
        string body = JsonSerializer.Serialize(bodyObj);

        try
        {
            using var client = new HttpClient();
            var content = new StringContent(body, Encoding.UTF8, "application/json");
            var response = await client.PostAsync($"{hostApi}/api/generate", content);
            string responseString = await response.Content.ReadAsStringAsync();

            if (enableLogging)
                WriteLog($"STATUS CODE: {(int)response.StatusCode}");

            Console.WriteLine($"STATUS CODE: {(int)response.StatusCode}");

            var fullResponse = new StringBuilder();
            var lines = responseString.Split('\n');
            foreach (var line in lines)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    try
                    {
                        using var doc = JsonDocument.Parse(line);
                        if (doc.RootElement.TryGetProperty("response", out var respProp))
                            fullResponse.Append(respProp.GetString());
                    }
                    catch { }
                }
            }

            if (printRaw)
            {
                Console.WriteLine(responseString);
                if (enableLogging)
                    WriteLog($"RAW RESPONSE: {responseString.Substring(0, Math.Min(200, responseString.Length))}...");
            }
            else
            {
                Console.WriteLine(fullResponse.ToString());
                if (enableLogging)
                    WriteLog($"FULL RESPONSE: {fullResponse.ToString().Substring(0, Math.Min(200, fullResponse.Length))}...");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("ERROR:");
            Console.WriteLine(ex);
            if (enableLogging)
                WriteLog($"ERROR: {ex}");
        }
    }
}