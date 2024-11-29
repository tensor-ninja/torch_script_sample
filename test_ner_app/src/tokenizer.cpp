#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

BertTokenizer::BertTokenizer(const std::string& vocab_path) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Could not open vocab file: " + vocab_path);
    }

    std::string line;
    int index = 0;  // Auto-increment index
    while (std::getline(vocab_file, line)) {
        // Remove any trailing whitespace
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), line.end());
        
        if (!line.empty()) {
            vocab[line] = index++;
            if (line.substr(0, 2) == "##") {
                subword_tokens.insert(line);
            }
        }
    }

    // Verify essential tokens exist
    std::vector<std::string> required_tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    for (const auto& token : required_tokens) {
        if (vocab.find(token) == vocab.end()) {
            std::cerr << "Warning: Required token '" << token << "' not found in vocabulary\n";
        }
    }
}

std::vector<torch::Tensor> BertTokenizer::encode(const std::string& text, int max_length) {
    #ifdef DEBUG
    std::cout << "\nEncoding text: '" << text << "'\n";
    #endif
    
    // Split into words but preserve case
    std::vector<std::string> words = basic_tokenize(text);
    #ifdef DEBUG
    std::cout << "Basic tokenization: ";
    for (const auto& word : words) std::cout << "'" << word << "' ";
    std::cout << "\n";
    #endif
    
    // Apply WordPiece tokenization
    std::vector<std::string> tokens = {"[CLS]"};
    for (const auto& word : words) {
        // First try the exact word with original case
        if (vocab.find(word) != vocab.end()) {
            tokens.push_back(word);
            #ifdef DEBUG
            std::cout << "Word '" << word << "' -> kept as-is\n";
            #endif
            continue;
        }
        
        // If not found, try WordPiece tokenization with original case
        auto word_tokens = wordpiece_tokenize(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        #ifdef DEBUG
        std::cout << "Word '" << word << "' -> tokens: ";
        for (const auto& t : word_tokens) std::cout << "'" << t << "' ";
        std::cout << "\n";
        #endif
    }
    tokens.push_back("[SEP]");
    
    // Convert tokens to ids and create attention mask
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    
    #ifdef DEBUG
    std::cout << "\nToken to ID conversion:\n";
    #endif
    
    for (const auto& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            input_ids.push_back(vocab[token]);
            #ifdef DEBUG
            std::cout << "'" << token << "' -> " << vocab[token] << "\n";
            #endif
        } else {
            input_ids.push_back(vocab["[UNK]"]);
            #ifdef DEBUG
            std::cout << "'" << token << "' -> [UNK](" << vocab["[UNK]"] << ")\n";
            #endif
        }
        attention_mask.push_back(1);
    }
    
    // Pad if necessary
    while (input_ids.size() < max_length) {
        input_ids.push_back(vocab["[PAD]"]);
        attention_mask.push_back(0);
    }
    
    // Create tensors
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor input_ids_tensor = torch::from_blob(input_ids.data(), {1, max_length}, options).clone();
    torch::Tensor attention_mask_tensor = torch::from_blob(attention_mask.data(), {1, max_length}, options).clone();
    
    return {input_ids_tensor, attention_mask_tensor};
}

std::vector<std::string> BertTokenizer::tokenize_for_test(const std::string& text) {
    std::string processed_text = preprocess_text(text);
    std::vector<std::string> words = basic_tokenize(processed_text);
    
    std::vector<std::string> final_tokens;
    for (const auto& word : words) {
        auto word_tokens = wordpiece_tokenize(word);
        final_tokens.insert(final_tokens.end(), word_tokens.begin(), word_tokens.end());
        
        #ifdef DEBUG
        std::cout << "Word: " << word << " -> Tokens: ";
        for (const auto& token : word_tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n";
        #endif
    }
    
    return final_tokens;
}

std::string BertTokenizer::preprocess_text(const std::string& text) {
    std::string processed = text;
    // Convert to lowercase
    std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
    // Add spaces around punctuation
    for (size_t i = 0; i < processed.length(); i++) {
        if (std::ispunct(processed[i])) {
            processed.insert(i, " ");
            processed.insert(i + 2, " ");
            i += 2;
        }
    }
    return processed;
}

std::vector<std::string> BertTokenizer::basic_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<std::string> BertTokenizer::wordpiece_tokenize(const std::string& word) {
    if (vocab.find(word) != vocab.end()) {
        return {word};
    }
    
    std::vector<std::string> tokens;
    size_t start = 0;
    bool is_first = true;
    
    while (start < word.length()) {
        size_t end = word.length();
        std::string curr_substr;
        
        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (!is_first) {
                substr = "##" + substr;
            }
            
            if (vocab.find(substr) != vocab.end()) {
                curr_substr = substr;
                break;
            }
            end--;
        }
        
        if (curr_substr.empty()) {
            return {"[UNK]"};
        }
        
        tokens.push_back(curr_substr);
        start += (is_first ? curr_substr.length() : curr_substr.length() - 2);
        is_first = false;
    }
    
    return tokens;
} 