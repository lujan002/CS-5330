#include "model_download.hpp"

#include <curl/curl.h>
#include <zip.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

const char* kHost = "https://www.models-resource.com";
const char* kGamePath = "/3ds/pokemonxy/";
const char* kAssetPrefix = "/3ds/pokemonxy/asset/";
const char* kUserAgent =
    "Mozilla/5.0 (X11; Linux x86_64) ar_card/1.0 (CS5330 course project)";

std::size_t appendToString(void* data, std::size_t size, std::size_t nmemb, void* userp) {
    const std::size_t total = size * nmemb;
    static_cast<std::string*>(userp)->append(static_cast<char*>(data), total);
    return total;
}

bool httpGet(const std::string& url, std::string& body, std::string& error) {
    body.clear();
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        error = "curl_easy_init failed";
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, appendToString);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, kUserAgent);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    const CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        error = std::string("GET ") + url + " failed: " + curl_easy_strerror(rc);
        return false;
    }
    if (body.empty()) {
        error = std::string("GET ") + url + " returned an empty body (HTTP " +
                std::to_string(status) + ")";
        return false;
    }
    return true;
}

// Comparison key: lowercase, punctuation and spaces removed, so "Mr. Mime",
// "mr mime" and "MrMime" all match. Non-ASCII bytes are kept so that the
// Nidoran forms stay distinguishable.
std::string normalizeName(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        if (std::isalnum(c) != 0) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c >= 0x80) {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

// The Models Resource HTML stores apostrophes as &#039; (and similar). Without
// decoding, "Farfetch'd" becomes the key "farfetch039d" and never matches.
std::string decodeBasicEntities(std::string value) {
    auto replace_all = [&](const std::string& from, const std::string& to) {
        for (std::size_t at = 0; (at = value.find(from, at)) != std::string::npos; ) {
            value.replace(at, from.size(), to);
            at += to.size();
        }
    };
    replace_all("&amp;", "&");
    replace_all("&lt;", "<");
    replace_all("&gt;", ">");
    replace_all("&quot;", "\"");
    replace_all("&#39;", "'");
    replace_all("&#039;", "'");
    replace_all("&apos;", "'");
    // Numeric decimal entities: &#NNNN;
    for (std::size_t at = 0; (at = value.find("&#", at)) != std::string::npos; ) {
        const std::size_t end = value.find(';', at + 2);
        if (end == std::string::npos) {
            break;
        }
        const std::string num = value.substr(at + 2, end - (at + 2));
        if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) {
            at = end + 1;
            continue;
        }
        const int code = std::stoi(num);
        if (code <= 0 || code > 127) {
            at = end + 1;
            continue;
        }
        value.replace(at, end - at + 1, 1, static_cast<char>(code));
        at += 1;
    }
    return value;
}

// "#0003 Venusaur" -> "Venusaur"
std::string stripDexNumber(const std::string& display) {
    if (display.empty() || display[0] != '#') {
        return display;
    }
    const std::size_t space = display.find(' ');
    return space == std::string::npos ? display : display.substr(space + 1);
}

// "#0003 Venusaur" / "#6 Charizard" -> national dex, or -1 if unparseable.
int parseDexNumber(const std::string& display) {
    if (display.empty() || display[0] != '#') {
        return -1;
    }
    std::size_t end = 1;
    while (end < display.size() &&
           std::isdigit(static_cast<unsigned char>(display[end]))) {
        end++;
    }
    if (end == 1) {
        return -1;
    }
    return std::stoi(display.substr(1, end - 1));
}

std::string attributeAfter(const std::string& html, std::size_t from, const std::string& attr) {
    const std::size_t at = html.find(attr, from);
    if (at == std::string::npos) {
        return "";
    }
    const std::size_t start = at + attr.size();
    const std::size_t end = html.find('"', start);
    return end == std::string::npos ? "" : html.substr(start, end - start);
}

fs::path makeScratchDir() {
    // tmpfs keeps downloads in RAM so nothing lands on the user's disk.
    std::error_code ec;
    static int counter = 0;
    const std::string leaf =
        "ar_card_" + std::to_string(getpid()) + "_" + std::to_string(counter++);

    for (const fs::path base : {fs::path("/dev/shm"), fs::temp_directory_path()}) {
        const fs::path dir = base / leaf;
        if (fs::create_directories(dir, ec)) {
            return dir;
        }
    }
    return {};
}

bool extractZip(const std::string& bytes, const fs::path& dest, std::string& error) {
    zip_error_t zip_err;
    zip_error_init(&zip_err);
    zip_source_t* source =
        zip_source_buffer_create(bytes.data(), bytes.size(), 0, &zip_err);
    if (source == nullptr) {
        error = std::string("zip_source_buffer_create: ") + zip_error_strerror(&zip_err);
        zip_error_fini(&zip_err);
        return false;
    }

    zip_t* archive = zip_open_from_source(source, ZIP_RDONLY, &zip_err);
    if (archive == nullptr) {
        error = std::string("not a readable zip archive: ") + zip_error_strerror(&zip_err);
        zip_source_free(source);
        zip_error_fini(&zip_err);
        return false;
    }
    zip_error_fini(&zip_err);

    const zip_int64_t count = zip_get_num_entries(archive, 0);
    std::vector<char> buffer(64 * 1024);
    bool ok = true;

    for (zip_int64_t i = 0; i < count && ok; ++i) {
        const char* raw_name = zip_get_name(archive, i, 0);
        if (raw_name == nullptr) {
            continue;
        }
        std::string name = raw_name;
        std::replace(name.begin(), name.end(), '\\', '/');
        // Refuse anything that would escape the scratch directory.
        if (name.empty() || name.front() == '/' || name.find("..") != std::string::npos) {
            continue;
        }

        const fs::path target = dest / name;
        std::error_code ec;
        if (name.back() == '/') {
            fs::create_directories(target, ec);
            continue;
        }
        fs::create_directories(target.parent_path(), ec);

        zip_file_t* file = zip_fopen_index(archive, i, 0);
        if (file == nullptr) {
            error = "could not read archive entry: " + name;
            ok = false;
            break;
        }
        std::ofstream out(target, std::ios::binary);
        if (!out) {
            zip_fclose(file);
            error = "could not write " + target.string();
            ok = false;
            break;
        }
        for (;;) {
            const zip_int64_t got = zip_fread(file, buffer.data(), buffer.size());
            if (got < 0) {
                error = "corrupt archive entry: " + name;
                ok = false;
                break;
            }
            if (got == 0) {
                break;
            }
            out.write(buffer.data(), got);
        }
        zip_fclose(file);
    }

    zip_close(archive);
    return ok;
}

}  // namespace

ModelDownloader::~ModelDownloader() {
    std::error_code ec;
    for (const std::string& dir : temp_dirs_) {
        fs::remove_all(dir, ec);
    }
}

bool ModelDownloader::fetchIndex(std::string& error) {
    if (!entries_.empty()) {
        return true;
    }

    std::string html;
    if (!httpGet(std::string(kHost) + kGamePath, html, error)) {
        return false;
    }

    // Each tile is: <a href="/3ds/pokemonxy/asset/292107/" ...>
    //                 ... <div class="iconheader" title="#0003 Venusaur">
    std::vector<std::string> seen;
    std::size_t pos = 0;
    while ((pos = html.find(kAssetPrefix, pos)) != std::string::npos) {
        const std::size_t id_start = pos + std::strlen(kAssetPrefix);
        std::size_t id_end = id_start;
        while (id_end < html.size() && std::isdigit(static_cast<unsigned char>(html[id_end]))) {
            id_end++;
        }
        if (id_end == id_start) {
            pos = id_start;
            continue;
        }
        const long asset_id = std::stol(html.substr(id_start, id_end - id_start));
        const std::string title = attributeAfter(html, id_end, "title=\"");
        pos = id_end;
        if (title.empty()) {
            continue;
        }

        IndexEntry entry;
        entry.display = decodeBasicEntities(title);
        entry.clean = stripDexNumber(entry.display);
        entry.asset_id = asset_id;

        const std::string key = normalizeName(entry.clean);
        if (key.empty() ||
            std::find(seen.begin(), seen.end(), key) != seen.end()) {
            continue;
        }
        seen.push_back(key);
        entries_.push_back(entry);
    }

    if (entries_.empty()) {
        error = "could not find any assets on the index page (site layout may have changed)";
        return false;
    }
    return true;
}

std::vector<std::string> ModelDownloader::suggest(
    const std::string& query, std::size_t limit) const {
    const std::string key = normalizeName(query);
    std::vector<std::string> hits;
    for (const IndexEntry& entry : entries_) {
        if (normalizeName(entry.clean).find(key) != std::string::npos) {
            hits.push_back(entry.clean);
            if (hits.size() >= limit) {
                break;
            }
        }
    }
    return hits;
}

bool ModelDownloader::downloadByDex(
    int dex,
    std::vector<ModelEntry>& out_models,
    std::string& error) {
    out_models.clear();
    if (dex <= 0) {
        error = "national dex must be a positive integer";
        return false;
    }
    if (!fetchIndex(error)) {
        return false;
    }

    // Several tiles can share a dex (base form, Mega, costume). Prefer the
    // shortest non-Mega name so "#0006 Charizard" wins over Mega Charizard X.
    const IndexEntry* best = nullptr;
    for (const IndexEntry& entry : entries_) {
        if (parseDexNumber(entry.display) != dex) {
            continue;
        }
        const std::string key = normalizeName(entry.clean);
        const bool mega = key.find("mega") != std::string::npos;
        if (best == nullptr) {
            best = &entry;
            continue;
        }
        const std::string best_key = normalizeName(best->clean);
        const bool best_mega = best_key.find("mega") != std::string::npos;
        if (best_mega && !mega) {
            best = &entry;
        } else if (best_mega == mega && key.size() < best_key.size()) {
            best = &entry;
        }
    }
    if (best == nullptr) {
        error = "no Pokemon X/Y model for national dex #" + std::to_string(dex);
        return false;
    }
    return download(best->clean, out_models, error);
}

bool ModelDownloader::download(
    const std::string& pokemon_name,
    std::vector<ModelEntry>& out_models,
    std::string& error) {
    out_models.clear();
    if (!fetchIndex(error)) {
        return false;
    }

    const std::string key = normalizeName(pokemon_name);
    if (key.empty()) {
        error = "empty Pokemon name";
        return false;
    }

    const IndexEntry* match = nullptr;
    for (const IndexEntry& entry : entries_) {
        if (normalizeName(entry.clean) == key) {
            match = &entry;
            break;
        }
    }
    if (match == nullptr) {  // fall back to a unique prefix match
        for (const IndexEntry& entry : entries_) {
            if (normalizeName(entry.clean).rfind(key, 0) == 0) {
                if (match != nullptr) {
                    match = nullptr;
                    break;
                }
                match = &entry;
            }
        }
    }
    if (match == nullptr) {
        error = "\"" + pokemon_name + "\" is not in the Pokemon X/Y model index";
        const std::vector<std::string> hints = suggest(pokemon_name);
        if (!hints.empty()) {
            error += " (did you mean: ";
            for (std::size_t i = 0; i < hints.size(); i++) {
                error += (i != 0 ? ", " : "") + hints[i];
            }
            error += "?)";
        }
        return false;
    }

    const std::string asset_url =
        std::string(kHost) + kAssetPrefix + std::to_string(match->asset_id) + "/";
    printf("Fetching %s from %s\n", match->clean.c_str(), asset_url.c_str());

    std::string asset_page;
    if (!httpGet(asset_url, asset_page, error)) {
        return false;
    }

    // The download button carries the archive path, e.g.
    // data-file="/media/assets/289/292107.zip?updated=1755498324"
    std::string zip_path = attributeAfter(asset_page, 0, "data-file=\"");
    if (zip_path.find("/media/assets/") == std::string::npos) {
        const std::size_t at = asset_page.find("/media/assets/");
        if (at == std::string::npos) {
            error = "no archive link on the asset page for " + match->clean;
            return false;
        }
        const std::size_t end = asset_page.find('"', at);
        zip_path = asset_page.substr(at, end - at);
    }

    std::string archive;
    if (!httpGet(kHost + zip_path, archive, error)) {
        return false;
    }
    printf("Downloaded %.1f MB, unpacking to RAM\n", archive.size() / (1024.0 * 1024.0));

    const fs::path scratch = makeScratchDir();
    if (scratch.empty()) {
        error = "could not create a scratch directory";
        return false;
    }
    temp_dirs_.push_back(scratch.string());

    if (!extractZip(archive, scratch, error)) {
        return false;
    }

    // Reuse the normal discovery rules so downloaded archives get the same
    // ColladaMax preference and Mega filtering as the bundled assets.
    out_models = discoverModels({scratch.string()});
    if (out_models.empty()) {
        error = "no loadable model found inside the " + match->clean + " archive";
        return false;
    }

    // Some archives name their file after the internal asset id ("pm0081_00"), so
    // fall back to the Pokedex name for anything that does not mention it.
    const int dex = parseDexNumber(match->display);
    const std::string clean_key = normalizeName(match->clean);
    for (std::size_t i = 0; i < out_models.size(); i++) {
        const std::string entry_key = normalizeName(out_models[i].name);
        if (entry_key == clean_key) {
            out_models[i].name = match->clean;  // fix casing, e.g. "snorlax"
        } else if (entry_key.find(clean_key) == std::string::npos) {
            out_models[i].name = out_models.size() > 1
                                     ? match->clean + "_" + std::to_string(i + 1)
                                     : match->clean;
        }
        if (dex > 0) {
            out_models[i].dex = dex;
        } else {
            resolveModelDex(out_models[i]);
        }
    }
    return true;
}
