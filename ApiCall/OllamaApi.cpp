#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Compile:
// sudo apt install libcurl4-openssl-dev nlohmann-json3-dev
// g++ -o OllamaAPI OllamaAPI.cpp -lcurl


std::string get_timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

void write_log(const std::string& message) {
    std::ofstream log("llm.log", std::ios::app);
    log << get_timestamp() << " " << message << std::endl;
}

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string hostApi = "http://localhost:11434";
    std::string model = "phi3:mini";
    std::string prompt = "What is the capital of France?";
    bool printRaw = false;
    bool enableLogging = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("-hostApi=", 0) == 0) hostApi = arg.substr(9);
        else if (arg.rfind("-model=", 0) == 0) model = arg.substr(7);
        else if (arg.rfind("-prompt=", 0) == 0) prompt = arg.substr(8);
        else if (arg == "-printRaw") printRaw = true;
        else if (arg == "-enableLogging") enableLogging = true;
    }

    if (enableLogging) {
        write_log("Run: hostApi='" + hostApi + "', model='" + model + "', prompt='" + prompt + "', printRaw=" + (printRaw ? "true" : "false"));
    }

    // Prepare JSON body
    json body_json = { {"model", model}, {"prompt", prompt} };
    std::string body = body_json.dump();

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to init curl" << std::endl;
        return 1;
    }

    std::string url = hostApi + "/api/generate";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    std::cout << "STATUS CODE: " << status << std::endl;
    if (enableLogging) {
        write_log("STATUS CODE: " + std::to_string(status));
    }

    if (res != CURLE_OK) {
        std::cerr << "ERROR: " << curl_easy_strerror(res) << std::endl;
        if (enableLogging) write_log(std::string("ERROR: ") + curl_easy_strerror(res));
    } else {
        std::string fullResponse;
        std::istringstream iss(response_string);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                try {
                    auto j = json::parse(line);
                    if (j.contains("response") && j["response"].is_string())
                        fullResponse += j["response"].get<std::string>();
                } catch (...) {}
            }
        }
        if (printRaw) {
            std::cout << response_string << std::endl;
            if (enableLogging) write_log("RAW RESPONSE: " + response_string.substr(0, 200) + "...");
        } else {
            std::cout << fullResponse << std::endl;
            if (enableLogging) write_log("FULL RESPONSE: " + fullResponse.substr(0, 200) + "...");
        }
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    return 0;
}
