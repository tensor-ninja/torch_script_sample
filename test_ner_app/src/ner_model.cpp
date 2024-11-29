#include "ner_model.hpp"
#include <fstream>

using json = nlohmann::json;

NERModel::NERModel(const std::string& model_path, 
                   const std::string& vocab_path,
                   const std::string& label_map_path) {
    #ifdef DEBUG
    std::cout << "Loading NER model...\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Vocab path: " << vocab_path << "\n";
    std::cout << "Label map path: " << label_map_path << "\n";
    #endif

    // Load the TorchScript model
    try {
        model = torch::jit::load(model_path);
        #ifdef DEBUG
        std::cout << "Model loaded successfully\n";
        #endif
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading the model: " + std::string(e.what()));
    }

    tokenizer = std::make_unique<BertTokenizer>(vocab_path);
    
    // Load label map
    std::ifstream label_file(label_map_path);
    if (!label_file.is_open()) {
        throw std::runtime_error("Could not open label map file: " + label_map_path);
    }
    
    label_map = json::parse(label_file);
    
    #ifdef DEBUG
    std::cout << "Label map loaded with " << label_map.size() << " labels:\n";
    for (auto it = label_map.begin(); it != label_map.end(); ++it) {
        std::cout << "ID " << it.key() << " -> " << it.value() << "\n";
    }
    #endif
}

std::vector<std::pair<std::string, std::string>> NERModel::predict(const std::string& text) {
    #ifdef DEBUG
    std::cout << "\nNER Prediction for: '" << text << "'\n";
    #endif

    // Get the WordPiece tokens and their IDs
    auto encoded = tokenizer->encode(text);
    auto original_tokens = tokenize(text);  // Space-based tokenization for final output
    
    #ifdef DEBUG
    std::cout << "Original tokens: ";
    for (const auto& t : original_tokens) {
        std::cout << "'" << t << "' ";
    }
    std::cout << "\n";
    #endif

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(encoded[0]);
    inputs.push_back(encoded[1]);
    
    #ifdef DEBUG
    std::cout << "Running model inference...\n";
    #endif

    // Run inference
    torch::Tensor output = model.forward(inputs).toTensor();
    auto predictions = output.argmax(-1);
    auto predictions_0 = predictions[0];
    auto pred_accessor = predictions_0.accessor<int64_t, 1>();
    
    #ifdef DEBUG
    std::cout << "Model output shape: " << output.sizes() << "\n";
    std::cout << "Predictions shape: " << predictions.sizes() << "\n";
    #endif

    // Convert predictions to labels
    std::vector<std::pair<std::string, std::string>> entities;
    std::string current_entity;
    std::string current_label;
    int token_idx = 0;
    
    // Skip [CLS] token (i=1) and stop before [SEP] token
    for (size_t i = 0; i < original_tokens.size(); i++) {
        int label_id = pred_accessor[i + 1];  // +1 to skip [CLS]
        std::string label = label_map[std::to_string(label_id)];
        
        #ifdef DEBUG
        std::cout << "Token '" << original_tokens[i] << "' -> Label '" << label << "' (ID: " << label_id << ")\n";
        #endif

        if (label[0] == 'B') {  // Beginning of entity
            if (!current_entity.empty()) {
                entities.push_back({current_entity, current_label});
            }
            current_entity = original_tokens[i];
            current_label = label.substr(2);  // Remove "B-"
        }
        else if (label[0] == 'I') {  // Inside entity
            if (!current_entity.empty()) {
                current_entity += " " + original_tokens[i];
            }
        }
        else {  // Outside (O)
            if (!current_entity.empty()) {
                entities.push_back({current_entity, current_label});
                current_entity.clear();
                current_label.clear();
            }
        }
    }
    
    // Add final entity if exists
    if (!current_entity.empty()) {
        entities.push_back({current_entity, current_label});
    }
    
    #ifdef DEBUG
    std::cout << "\nFound " << entities.size() << " entities\n";
    for (const auto& entity : entities) {
        std::cout << "Entity: '" << entity.first << "' -> Type: '" << entity.second << "'\n";
    }
    #endif

    return entities;
}

std::vector<std::string> NERModel::tokenize(const std::string& text) {
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