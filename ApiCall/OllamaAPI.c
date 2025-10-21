#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#define LOG_FILE "llm.log"

typedef struct {
    char *memory;
    size_t size;
} MemoryStruct;

void write_log(const char *message) {
    FILE *log = fopen(LOG_FILE, "a");
    if (!log) return;
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
    fprintf(log, "%s %s\n", buf, message);
    fclose(log);
}

size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryStruct *mem = (MemoryStruct *)userp;
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(ptr == NULL) return 0; // out of memory!
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    return realsize;
}

int main(int argc, char *argv[]) {
    // Default parameters
    char *hostApi = "http://localhost:11434";
    char *model = "phi3:mini";
    char *prompt = "What is the capital of France?";
    int printRaw = 0;
    int enableLogging = 0;

    // Parse command-line arguments (very basic)
    for(int i=1; i<argc; i++) {
        if(strncmp(argv[i], "-hostApi=", 9)==0) hostApi = argv[i]+9;
        else if(strncmp(argv[i], "-model=", 7)==0) model = argv[i]+7;
        else if(strncmp(argv[i], "-prompt=", 8)==0) prompt = argv[i]+8;
        else if(strcmp(argv[i], "-printRaw")==0) printRaw = 1;
        else if(strcmp(argv[i], "-enableLogging")==0) enableLogging = 1;
    }

    char logmsg[512];
    if(enableLogging) {
        snprintf(logmsg, sizeof(logmsg), "Run: hostApi='%s', model='%s', prompt='%s', printRaw=%d", hostApi, model, prompt, printRaw);
        write_log(logmsg);
    }

    // Prepare JSON body
    char body[1024];
    snprintf(body, sizeof(body), "{\"model\":\"%s\",\"prompt\":\"%s\"}", model, prompt);

    CURL *curl = curl_easy_init();
    if(!curl) {
        fprintf(stderr, "Failed to init curl\n");
        return 1;
    }

    char url[512];
    snprintf(url, sizeof(url), "%s/api/generate", hostApi);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    MemoryStruct chunk = {0};
    chunk.memory = malloc(1);
    chunk.size = 0;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

    CURLcode res = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    printf("STATUS CODE: %ld\n", status);
    if(enableLogging) {
        snprintf(logmsg, sizeof(logmsg), "STATUS CODE: %ld", status);
        write_log(logmsg);
    }

    if(res != CURLE_OK) {
        fprintf(stderr, "ERROR: %s\n", curl_easy_strerror(res));
        if(enableLogging) {
            snprintf(logmsg, sizeof(logmsg), "ERROR: %s", curl_easy_strerror(res));
            write_log(logmsg);
        }
    } else {
        if(printRaw) {
            printf("%s\n", chunk.memory);
            if(enableLogging) {
                snprintf(logmsg, sizeof(logmsg), "RAW RESPONSE: %.200s...", chunk.memory);
                write_log(logmsg);
            }
        } else {
            // Split by newline, parse each line as JSON, concatenate "response"
            char *saveptr, *line = strtok_r(chunk.memory, "\n", &saveptr);
            char fullResponse[8192] = {0};
            while(line) {
                cJSON *json = cJSON_Parse(line);
                if(json) {
                    cJSON *resp = cJSON_GetObjectItem(json, "response");
                    if(resp && cJSON_IsString(resp)) {
                        strncat(fullResponse, resp->valuestring, sizeof(fullResponse)-strlen(fullResponse)-1);
                    }
                    cJSON_Delete(json);
                }
                line = strtok_r(NULL, "\n", &saveptr);
            }
            printf("%s\n", fullResponse);
            if(enableLogging) {
                snprintf(logmsg, sizeof(logmsg), "FULL RESPONSE: %.200s...", fullResponse);
                write_log(logmsg);
            }
        }
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    free(chunk.memory);
    return 0;
}
