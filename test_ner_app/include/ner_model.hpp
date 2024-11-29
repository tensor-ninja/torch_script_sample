#pragma once
#include <torch/script.h>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include "tokenizer.hpp"

class NERModel {
public:
    NERModel(const std::string& model_path, 
             const std::string& vocab_path,
             const std::string& label_map_path);
             
    std::vector<std::pair<std::string, std::string>> predict(const std::string& text);

private:
    torch::jit::script::Module model;
    std::unique_ptr<BertTokenizer> tokenizer;
    nlohmann::json label_map;
    
    std::vector<std::string> tokenize(const std::string& text);
};