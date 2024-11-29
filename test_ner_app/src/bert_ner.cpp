#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

class BertTokenizer {
public:
    BertTokenizer(const std::string& vocab_path) {
        std::ifstream vocab_file(vocab_path);
        std::string line;
        while (std::getline(vocab_file, line)) {
            size_t space_pos = line.find('\t');
            if (space_pos != std::string::npos) {
                std::string token = line.substr(0, space_pos);
                int id = std::stoi(line.substr(space_pos + 1));
                vocab[token] = id;
                if (token.substr(0, 2) == "##") {
                    subword_tokens.insert(token);
                }
            }
        }
    }
    
    std::vector<torch::Tensor> encode(const std::string& text, int max_length = 128) {
        // Perform basic preprocessing
        std::string processed_text = preprocess_text(text);
        
        // Tokenize into words first
        std::vector<std::string> words = basic_tokenize(processed_text);
        
        // Apply WordPiece tokenization
        std::vector<std::string> tokens = {"[CLS]"};
        for (const auto& word : words) {
            auto word_tokens = wordpiece_tokenize(word);
            tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        }
        tokens.push_back("[SEP]");
        
        // Convert tokens to ids and create attention mask
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        
        for (const auto& token : tokens) {
            if (vocab.find(token) != vocab.end()) {
                input_ids.push_back(vocab[token]);
            } else {
                input_ids.push_back(vocab["[UNK]"]);
            }
            attention_mask.push_back(1);
        }
        
        // Truncate if necessary
        if (input_ids.size() > max_length) {
            input_ids.resize(max_length);
            attention_mask.resize(max_length);
        }
        
        // Pad if necessary
        while (input_ids.size() < max_length) {
            input_ids.push_back(0);  // padding token id
            attention_mask.push_back(0);
        }
        
        // Convert to tensors
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        torch::Tensor input_ids_tensor = torch::from_blob(input_ids.data(), {1, max_length}, options).clone();
        torch::Tensor attention_mask_tensor = torch::from_blob(attention_mask.data(), {1, max_length}, options).clone();
        
        return {input_ids_tensor, attention_mask_tensor};
    }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_set<std::string> subword_tokens;
    
    std::string preprocess_text(const std::string& text) {
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
    
    std::vector<std::string> basic_tokenize(const std::string& text) {
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
    
    std::vector<std::string> wordpiece_tokenize(const std::string& word) {
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
};

class NERModel {
public:
    NERModel(const std::string& model_path, 
             const std::string& vocab_path,
             const std::string& label_map_path) {
        // Load the TorchScript model
        model = torch::jit::load(model_path);
        tokenizer = std::make_unique<BertTokenizer>(vocab_path);
        
        // Load label map
        std::ifstream label_file(label_map_path);
        label_map = json::parse(label_file);
    }
    
    std::vector<std::pair<std::string, std::string>> predict(const std::string& text) {
        // Tokenize input
        auto encoded = tokenizer->encode(text);
        auto tokens = tokenize(text);
        
        // Create a vector of inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(encoded[0]);
        inputs.push_back(encoded[1]);
        
        // Run inference
        torch::Tensor output = model.forward(inputs).toTensor();
        
        // Process predictions
        auto predictions = output.argmax(-1);
        auto predictions_0 = predictions[0];
        auto pred_accessor = predictions_0.accessor<int64_t, 1>();
        
        // Convert predictions to labels
        std::vector<std::pair<std::string, std::string>> entities;
        
        // Skip [CLS] token (i=1) and stop before [SEP] token
        for (int i = 1; i < tokens.size() + 1; i++) {
            int label_id = pred_accessor[i];
            std::string label = label_map[std::to_string(label_id)];
            if (label != "O") {  // Not "Outside" label
                entities.push_back({tokens[i-1], label});
            }
        }
        
        return entities;
    }
    
private:
    torch::jit::script::Module model;
    std::unique_ptr<BertTokenizer> tokenizer;
    json label_map;
    
    std::vector<std::string> tokenize(const std::string& text) {
        // Basic tokenization - you might want to improve this
        std::vector<std::string> tokens;
        size_t start = 0, end = 0;
        while ((end = text.find(' ', start)) != std::string::npos) {
            tokens.push_back(text.substr(start, end - start));
            start = end + 1;
        }
        tokens.push_back(text.substr(start));
        return tokens;
    }
};

int main() {
    try {
        NERModel ner("../exported_model/traced_model.pt",
                    "../exported_model/vocab.txt",
                    "../exported_model/label_map.json");
        
        std::string text = "John works at Microsoft in Seattle";
        auto entities = ner.predict(text);
        
        std::cout << "Found entities:\n";
        for (const auto& entity : entities) {
            std::cout << entity.first << ": " << entity.second << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}