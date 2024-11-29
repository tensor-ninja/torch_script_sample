#pragma once
#include <torch/script.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class BertTokenizer {
public:
    BertTokenizer(const std::string& vocab_path);
    std::vector<torch::Tensor> encode(const std::string& text, int max_length = 128);
    // Add for testing purposes
    std::vector<std::string> tokenize_for_test(const std::string& text);

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_set<std::string> subword_tokens;
    
    std::string preprocess_text(const std::string& text);
    std::vector<std::string> basic_tokenize(const std::string& text);
    std::vector<std::string> wordpiece_tokenize(const std::string& word);
};