#include "poke3d_download.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char* kAssetRoot =
    "https://raw.githubusercontent.com/Pokemon-3D-api/assets/main/models/opt/";
const char* kPokeApi = "https://pokeapi.co/api/v2/pokemon/";
const char* kUserAgent =
    "Mozilla/5.0 (X11; Linux x86_64) ar_card/1.0 (CS5330 course project)";

struct AssetCandidate {
    std::string category;  // regular, multiform, alolan, ...
    std::string file;      // 479.glb or RotomHeat.glb
    std::string label;     // display hint, e.g. "Heat Rotom"
};

std::size_t appendToString(void* data, std::size_t size, std::size_t nmemb, void* userp) {
    const std::size_t total = size * nmemb;
    static_cast<std::string*>(userp)->append(static_cast<char*>(data), total);
    return total;
}

bool httpGet(const std::string& url, std::string& body, long& status, std::string& error) {
    body.clear();
    status = 0;
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
    // Capture 404s ourselves so callers can try gendered / form fallbacks.
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0L);

    const CURLcode rc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        error = std::string("GET ") + url + " failed: " + curl_easy_strerror(rc);
        return false;
    }
    if (status < 200 || status >= 300) {
        error = std::string("GET ") + url + " returned HTTP " + std::to_string(status);
        return false;
    }
    if (body.empty()) {
        error = std::string("GET ") + url + " returned an empty body";
        return false;
    }
    return true;
}

std::string slugifyName(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        if (std::isalnum(c) != 0) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == ' ' || c == '_' || c == '-') {
            out.push_back('-');
        }
    }
    // Collapse repeated dashes.
    std::string collapsed;
    for (char c : out) {
        if (c == '-' && !collapsed.empty() && collapsed.back() == '-') {
            continue;
        }
        collapsed.push_back(c);
    }
    while (!collapsed.empty() && collapsed.back() == '-') {
        collapsed.pop_back();
    }
    return collapsed;
}

std::string titleCaseSlug(const std::string& slug) {
    std::string out;
    bool cap = true;
    for (char c : slug) {
        if (c == '-' || c == '_') {
            out.push_back(' ');
            cap = true;
            continue;
        }
        if (cap && std::isalpha(static_cast<unsigned char>(c))) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            cap = false;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string toLower(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool containsToken(const std::string& haystack_lower, const char* token) {
    return haystack_lower.find(token) != std::string::npos;
}

// Pull the species national dex from a PokeAPI JSON blob. Form endpoints put a
// form id first; the species object is unique: "species":{"name":"...","url":
// ".../pokemon-species/NNN/"}.
int parsePokeApiSpeciesDex(const std::string& json) {
    const std::string key = "pokemon-species/";
    const std::size_t at = json.find(key);
    if (at == std::string::npos) {
        return -1;
    }
    std::size_t i = at + key.size();
    std::size_t end = i;
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        end++;
    }
    if (end == i) {
        return -1;
    }
    return std::stoi(json.substr(i, end - i));
}

std::string parsePokeApiName(const std::string& json) {
    // Root "name" is not the first "name" in the blob (abilities come first).
    // The species object is unique: "species":{"name":"charizard",...}
    const std::string key = "\"species\":{\"name\":\"";
    const std::size_t at = json.find(key);
    if (at == std::string::npos) {
        return "";
    }
    const std::size_t start = at + key.size();
    const std::size_t end = json.find('"', start);
    if (end == std::string::npos) {
        return "";
    }
    return json.substr(start, end - start);
}

fs::path makeScratchDir() {
    std::error_code ec;
    static int counter = 0;
    const std::string leaf =
        "poke3d_" + std::to_string(getpid()) + "_" + std::to_string(counter++);

    for (const fs::path base : {fs::path("/dev/shm"), fs::temp_directory_path()}) {
        const fs::path dir = base / leaf;
        if (fs::create_directories(dir, ec)) {
            return dir;
        }
    }
    return {};
}

std::string findPrepScript() {
    const char* candidates[] = {
        "tools/poke3d_prep/prep_glb.mjs",
        "../tools/poke3d_prep/prep_glb.mjs",
        "../../tools/poke3d_prep/prep_glb.mjs",
    };
    for (const char* path : candidates) {
        std::ifstream probe(path);
        if (probe.is_open()) {
            return path;
        }
    }
    return "";
}

bool runPrep(const fs::path& input, const fs::path& output, std::string& error) {
    const std::string script = findPrepScript();
    if (script.empty()) {
        error = "could not find tools/poke3d_prep/prep_glb.mjs "
                "(run from project_final/ or project_final/build/)";
        return false;
    }

    // Ensure node_modules exist; surface a clear message if not.
    const fs::path script_dir = fs::path(script).parent_path();
    if (!fs::exists(script_dir / "node_modules")) {
        error = "poke-3D prep deps missing — run: "
                "cd tools/poke3d_prep && npm install";
        return false;
    }

    const std::string cmd =
        "node \"" + script + "\" \"" + input.string() + "\" \"" + output.string() + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        error = "prep_glb.mjs failed (exit " + std::to_string(rc) +
                "). Need node + npm deps in tools/poke3d_prep";
        return false;
    }
    if (!fs::exists(output)) {
        error = "prep_glb.mjs did not write " + output.string();
        return false;
    }
    return true;
}

bool writeBytes(const fs::path& path, const std::string& bytes, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "could not write " + path.string();
        return false;
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        error = "failed while writing " + path.string();
        return false;
    }
    return true;
}

void pushUnique(std::vector<AssetCandidate>& out, AssetCandidate c) {
    for (const AssetCandidate& existing : out) {
        if (existing.category == c.category && existing.file == c.file) {
            return;
        }
    }
    out.push_back(std::move(c));
}

// Map a TCG card name + national dex to poke-3D asset paths, preferred first.
// Falls back to regular/<dex>.glb (+ gendered -M) when no form matches.
std::vector<AssetCandidate> formCandidates(int dex, const std::string& card_name) {
    std::vector<AssetCandidate> out;
    const std::string lower = toLower(card_name);
    const std::string dex_file = std::to_string(dex) + ".glb";

    auto regional = [&](const char* token, const char* category, const char* label) {
        if (containsToken(lower, token)) {
            pushUnique(out, {category, dex_file, label});
        }
    };
    regional("alolan", "alolan", "Alolan");
    regional("hisuian", "hisuian", "Hisuian");
    regional("galarian", "galar", "Galarian");
    regional("paldean", "paldea", "Paldean");

    if (containsToken(lower, "origin")) {
        pushUnique(out, {"origin", dex_file, "Origin"});
    }
    if (containsToken(lower, "primal")) {
        pushUnique(out, {"primal", dex_file, "Primal"});
    }
    if (containsToken(lower, "gigantamax") || containsToken(lower, "gmax")) {
        pushUnique(out, {"gmax", dex_file, "Gigantamax"});
    }

    // Mega X / Mega Y before plain Mega.
    if (containsToken(lower, "mega") &&
        (lower.find(" x") != std::string::npos || lower.rfind(" x") == lower.size() - 2 ||
         containsToken(lower, " mega x") || lower.find("-x") != std::string::npos)) {
        pushUnique(out, {"x", dex_file, "Mega X"});
    }
    if (containsToken(lower, "mega") &&
        (lower.find(" y") != std::string::npos ||
         containsToken(lower, " mega y") || lower.find("-y") != std::string::npos)) {
        pushUnique(out, {"y", dex_file, "Mega Y"});
    }
    if (containsToken(lower, "mega")) {
        pushUnique(out, {"mega", dex_file, "Mega"});
    }

    // Named multiforms (Rotom appliances, Shaymin Sky, etc.).
    if (dex == 479) {
        if (containsToken(lower, "heat")) {
            pushUnique(out, {"multiform", "RotomHeat.glb", "Heat Rotom"});
        }
        if (containsToken(lower, "wash")) {
            pushUnique(out, {"multiform", "RotomWash.glb", "Wash Rotom"});
        }
        if (containsToken(lower, "frost")) {
            pushUnique(out, {"multiform", "RotomFrost.glb", "Frost Rotom"});
        }
        if (containsToken(lower, "fan")) {
            pushUnique(out, {"multiform", "RotomFan.glb", "Fan Rotom"});
        }
        if (containsToken(lower, "mow")) {
            pushUnique(out, {"multiform", "RotomMow.glb", "Mow Rotom"});
        }
    }
    if (dex == 492 &&
        (containsToken(lower, "sky forme") || containsToken(lower, "sky form"))) {
        pushUnique(out, {"multiform", "ShayminSky.glb", "Shaymin Sky"});
    }
    if (dex == 905 && containsToken(lower, "therian")) {
        pushUnique(out, {"multiform", "EnamorusTherian.glb", "Enamorus Therian"});
    }
    if (dex == 745) {
        if (containsToken(lower, "midnight")) {
            pushUnique(out, {"multiform", "LycanrocMidnightForm.glb", "Lycanroc Midnight"});
        }
        if (containsToken(lower, "dusk")) {
            pushUnique(out, {"multiform", "LycanrocDuskForm.glb", "Lycanroc Dusk"});
        }
    }
    if (dex == 746 && containsToken(lower, "school")) {
        pushUnique(out, {"multiform", "WishiwashiSchool.glb", "Wishiwashi School"});
    }
    if (dex == 550 &&
        (containsToken(lower, "blue-stripe") || containsToken(lower, "blue stripe"))) {
        pushUnique(out, {"multiform", "BasculinBlueStripe.glb", "Basculin Blue-Stripe"});
    }

    // Always fall back to the regular (and male) asset.
    pushUnique(out, {"regular", dex_file, ""});
    pushUnique(out, {"regular", std::to_string(dex) + "-M.glb", ""});
    return out;
}

}  // namespace

Poke3dDownloader::~Poke3dDownloader() {
    std::error_code ec;
    for (const std::string& dir : temp_dirs_) {
        fs::remove_all(dir, ec);
    }
}

bool Poke3dDownloader::downloadByDex(
    int dex,
    std::vector<ModelEntry>& out_models,
    std::string& error,
    const std::string& card_name) {
    out_models.clear();
    if (dex <= 0) {
        error = "national dex must be a positive integer";
        return false;
    }

    const fs::path scratch = makeScratchDir();
    if (scratch.empty()) {
        error = "could not create a scratch directory";
        return false;
    }
    temp_dirs_.push_back(scratch.string());

    const std::vector<AssetCandidate> candidates = formCandidates(dex, card_name);
    std::string glb_bytes;
    AssetCandidate used;
    std::string last_error;
    for (const AssetCandidate& cand : candidates) {
        const std::string url =
            std::string(kAssetRoot) + cand.category + "/" + cand.file;
        long status = 0;
        std::string try_error;
        printf("Fetching poke-3D model %s\n", url.c_str());
        if (httpGet(url, glb_bytes, status, try_error)) {
            used = cand;
            last_error.clear();
            break;
        }
        last_error = try_error;
        glb_bytes.clear();
        if (status != 404) {
            error = try_error;
            return false;
        }
    }
    if (glb_bytes.empty()) {
        error = "no poke-3D model for national dex #" + std::to_string(dex);
        if (!card_name.empty()) {
            error += " (\"" + card_name + "\")";
        }
        if (!last_error.empty()) {
            error += " (" + last_error + ")";
        }
        return false;
    }

    printf("Downloaded %.1f KB (%s/%s), preparing for Assimp\n",
           glb_bytes.size() / 1024.0, used.category.c_str(), used.file.c_str());

    const fs::path raw_path = scratch / (used.file + "_raw.glb");
    const fs::path prep_path = scratch / used.file;
    if (!writeBytes(raw_path, glb_bytes, error)) {
        return false;
    }
    if (!runPrep(raw_path, prep_path, error)) {
        return false;
    }

    // Drop the raw download; keep only the Assimp-friendly file.
    std::error_code ec;
    fs::remove(raw_path, ec);

    std::string display = "Dex" + std::to_string(dex);
    {
        std::string json;
        long status = 0;
        std::string ignore;
        if (httpGet(std::string(kPokeApi) + std::to_string(dex), json, status, ignore)) {
            const std::string slug = parsePokeApiName(json);
            if (!slug.empty()) {
                display = titleCaseSlug(slug);
            }
        }
    }
    if (used.category == "multiform" && !used.label.empty()) {
        display = used.label;
    } else if (used.category != "regular" && !used.label.empty()) {
        // "Alolan" + "Raichu" → "Alolan Raichu"
        if (display.rfind(used.label, 0) != 0) {
            display = used.label + " " + display;
        }
    } else if (!card_name.empty() && used.category != "regular") {
        display = card_name;
    }

    ModelEntry entry;
    entry.name = display;
    entry.path = prep_path.string();
    entry.dex = dex;
    out_models.push_back(entry);
    return true;
}

bool Poke3dDownloader::download(
    const std::string& pokemon_name,
    std::vector<ModelEntry>& out_models,
    std::string& error) {
    out_models.clear();
    const std::string slug = slugifyName(pokemon_name);
    if (slug.empty()) {
        error = "empty Pokemon name";
        return false;
    }

    std::string json;
    long status = 0;
    // Try the full slug first (rotom-heat, etc.), then strip form prefixes.
    std::vector<std::string> try_slugs = {slug};
    // "heat-rotom" / "alolan-raichu" — also try the trailing species token.
    const std::size_t dash = slug.rfind('-');
    if (dash != std::string::npos && dash + 1 < slug.size()) {
        try_slugs.push_back(slug.substr(dash + 1));
    }
    // "alolan-raichu" — species is last; "rotom-heat" — species is first.
    if (dash != std::string::npos && dash > 0) {
        try_slugs.push_back(slug.substr(0, dash));
    }

    int dex = -1;
    std::string last_error;
    for (const std::string& try_slug : try_slugs) {
        const std::string url = std::string(kPokeApi) + try_slug;
        printf("Resolving \"%s\" via PokeAPI\n", try_slug.c_str());
        if (!httpGet(url, json, status, last_error)) {
            continue;
        }
        dex = parsePokeApiSpeciesDex(json);
        if (dex <= 0) {
            // Fall back to the first "id": for plain species endpoints.
            const std::string key = "\"id\":";
            const std::size_t at = json.find(key);
            if (at != std::string::npos) {
                std::size_t i = at + key.size();
                while (i < json.size() &&
                       std::isspace(static_cast<unsigned char>(json[i]))) {
                    i++;
                }
                std::size_t end = i;
                while (end < json.size() &&
                       std::isdigit(static_cast<unsigned char>(json[end]))) {
                    end++;
                }
                if (end > i) {
                    dex = std::stoi(json.substr(i, end - i));
                }
            }
        }
        if (dex > 0) {
            break;
        }
    }
    if (dex <= 0) {
        error = "\"" + pokemon_name + "\" could not be resolved to a national dex";
        if (!last_error.empty()) {
            error += " (" + last_error + ")";
        }
        return false;
    }
    return downloadByDex(dex, out_models, error, pokemon_name);
}
